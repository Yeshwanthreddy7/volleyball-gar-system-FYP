"""
train_mamba.py – Training Pipeline for the Mamba Volleyball Classifier.

Loads labelled CSV files from a directory (produced by prepare_training_data.py
or label_clips.py), trains MambaClassifier with data augmentation, and saves a
checkpoint.

Features:
  • Stratified train / val / test split
  • Data augmentation: horizontal flip, time-jitter, Gaussian noise
  • Weighted cross-entropy (handles class imbalance)
  • Label smoothing
  • AdamW + cosine-annealing LR scheduler
  • Gradient clipping
  • Early stopping on val macro-F1
  • Per-class F1, precision, recall on test set
  • Formatted confusion matrix with class names

Usage:
  python train_mamba.py ./training_csv \\
      --epochs 60 --batch_size 32 --lr 1e-3 \\
      --checkpoint mamba_checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mamba_model import (
    FEATURE_COLS, INPUT_DIM, LABEL_NAMES, NUM_CLASSES,
    MambaClassifier, SEQ_LEN,
)

MIN_STD = 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset with augmentation
# ──────────────────────────────────────────────────────────────────────────────

class VolleyballDataset(Dataset):
    """
    Loads all labelled CSVs from a directory.

    Each CSV: columns = frame_id, ball_x, ball_y, p1_x, p1_y, …, p6_x, p6_y, target_label
              rows    = exactly SEQ_LEN (29) frames per sequence.
    """

    def __init__(self, directory: str) -> None:
        self.samples: list[tuple[torch.Tensor, int]] = []
        self._label_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}

        csv_files = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in '{directory}'.")

        skipped = 0
        for fname in sorted(csv_files):
            fpath = os.path.join(directory, fname)
            try:
                df = pd.read_csv(fpath)
            except Exception as exc:
                print(f"[SKIP] {fname}: {exc}")
                skipped += 1
                continue

            if "target_label" not in df.columns:
                print(f"[SKIP] {fname}: no 'target_label' column.")
                skipped += 1
                continue

            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                print(f"[SKIP] {fname}: missing feature cols {missing}.")
                skipped += 1
                continue

            label_str = df["target_label"].iloc[0]
            if label_str not in self._label_to_idx:
                print(f"[SKIP] {fname}: unknown label '{label_str}'.")
                skipped += 1
                continue

            label_idx = self._label_to_idx[label_str]
            features  = df[FEATURE_COLS].values.astype(np.float32)   # (T, 14)

            # Pad / truncate to exactly SEQ_LEN
            if len(features) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(features), INPUT_DIM), np.float32)
                features = np.vstack([features, pad])
            else:
                features = features[:SEQ_LEN]

            self.samples.append((torch.from_numpy(features), label_idx))

        print(f"Loaded {len(self.samples)} sequences ({skipped} skipped).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.samples[idx]


class AugmentedWrapper(Dataset):
    """
    Wraps VolleyballDataset with z-score normalisation and optional augmentations.

    Augmentations (applied randomly during training):
      • Horizontal flip  – mirror X coordinates of ball and all players.
                           Simulates mirror-image tactic patterns.
      • Time jitter      – randomly shift start by ±jitter_frames.
      • Gaussian noise   – add small noise to all features.
    """

    def __init__(
        self,
        base:        VolleyballDataset,
        indices:     list[int],
        mean:        torch.Tensor,
        std:         torch.Tensor,
        augment:     bool  = False,
        flip_prob:   float = 0.5,
        jitter_max:  int   = 3,
        noise_std:   float = 0.01,
    ) -> None:
        self.base       = base
        self.indices    = indices
        self.mean       = mean
        self.std        = std
        self.augment    = augment
        self.flip_prob  = flip_prob
        self.jitter_max = jitter_max
        self.noise_std  = noise_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        seq, label = self.base.samples[self.indices[i]]   # (29, 14)
        seq = seq.clone()

        if self.augment:
            # ── Time jitter ──────────────────────────────────────────────────
            if self.jitter_max > 0:
                j = random.randint(-self.jitter_max, self.jitter_max)
                if j != 0:
                    seq = torch.roll(seq, shifts=j, dims=0)
                    # Zero-out the wrapped rows to avoid temporal leakage
                    if j > 0:
                        seq[:j] = 0.0
                    else:
                        seq[j:] = 0.0

            # ── Horizontal flip (mirror X coordinates) ───────────────────────
            # Feature layout: [ball_x, ball_y, p1_x, p1_y, …, p6_x, p6_y]
            # X columns: 0, 2, 4, 6, 8, 10, 12  (0-indexed)
            # Flip: new_x = COURT_WIDTH_CM - old_x
            if random.random() < self.flip_prob:
                from mamba_model import COURT_WIDTH_CM
                x_cols = [0, 2, 4, 6, 8, 10, 12]
                for c in x_cols:
                    # Only flip non-zero entries (zero = undetected player)
                    mask     = seq[:, c] != 0
                    seq[mask, c] = COURT_WIDTH_CM - seq[mask, c]

            # ── Gaussian noise ───────────────────────────────────────────────
            if self.noise_std > 0:
                noise = torch.randn_like(seq) * self.noise_std
                seq   = seq + noise

        # Z-score normalisation
        seq = (seq - self.mean) / self.std
        return seq, label


# ──────────────────────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_normalization(
    dataset: VolleyballDataset, train_indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = torch.stack([dataset.samples[i][0] for i in train_indices])  # (N, 29, 14)
    mean = seqs.mean(dim=(0, 1))
    std  = seqs.std(dim=(0, 1)).clamp(min=MIN_STD)
    return mean, std


def compute_class_weights(
    dataset: VolleyballDataset, train_indices: list[int], device: torch.device
) -> torch.Tensor:
    counts = torch.zeros(NUM_CLASSES)
    for i in train_indices:
        _, lbl = dataset.samples[i]
        counts[lbl] += 1
    weights = NUM_CLASSES / (counts + 1.0)
    return (weights / weights.mean()).to(device)


def stratified_split(
    dataset: VolleyballDataset, val_split: float, test_split: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, (_, lbl) in enumerate(dataset.samples):
        by_label[int(lbl)].append(i)

    train_idx = val_idx = test_idx = []
    train_idx, val_idx, test_idx = [], [], []

    for lbl, indices in by_label.items():
        rng.shuffle(indices)
        n      = len(indices)
        n_test = max(1, int(np.floor(n * test_split))) if test_split > 0 and n >= 3 else 0
        n_val  = max(1, int(np.floor(n * val_split)))  if val_split  > 0 and n >= 3 else 0
        while n - n_val - n_test < 1 and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0: n_val -= 1
            elif n_test > 0:                  n_test -= 1

        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test: n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def split_distribution(dataset: VolleyballDataset, indices: list[int]) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_NAMES}
    for i in indices:
        _, lbl = dataset.samples[i]
        counts[LABEL_NAMES[int(lbl)]] += 1
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def confusion_metrics(cm: torch.Tensor) -> dict:
    """
    Compute per-class precision, recall, F1, and macro averages
    from a (num_classes × num_classes) confusion matrix (rows=true, cols=pred).
    """
    total   = float(cm.sum().item())
    correct = float(torch.diag(cm).sum().item())
    acc     = correct / total if total > 0 else 0.0

    tp   = torch.diag(cm).float()
    sup  = cm.sum(dim=1).float()
    pred = cm.sum(dim=0).float()

    prec = tp / pred.clamp(min=1.0)
    rec  = tp / sup.clamp(min=1.0)
    f1   = 2.0 * prec * rec / (prec + rec).clamp(min=1e-12)

    return {
        "accuracy":     acc,
        "macro_f1":     float(f1.mean().item()),
        "balanced_acc": float(rec.mean().item()),
        "per_class":    {
            LABEL_NAMES[i]: {
                "precision": float(prec[i].item()),
                "recall":    float(rec[i].item()),
                "f1":        float(f1[i].item()),
                "support":   int(sup[i].item()),
            }
            for i in range(NUM_CLASSES)
        },
    }


def print_confusion_matrix(cm: torch.Tensor, title: str = "Confusion Matrix") -> None:
    """Print a labelled confusion matrix."""
    short = ["SpBk", "DlSp", "CoAt", "CoDf"]   # 4-char abbreviations
    print(f"\n  {title}  (rows=True, cols=Pred)")
    header = "         " + "  ".join(f"{s:>5}" for s in short)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, row_name in enumerate(short):
        row = cm[i].tolist()
        vals = "  ".join(f"{int(v):5d}" for v in row)
        label_full = LABEL_NAMES[i]
        print(f"  {row_name}  | {vals}   ← {label_full}")
    print()


def save_confusion_heatmap(cm: torch.Tensor, path: str) -> None:
    """Save a matplotlib confusion matrix heatmap."""
    try:
        import seaborn as sns
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    short = ["SpBk", "DlSp", "CoAt", "CoDf"]
    data  = cm.numpy().astype(float)
    sns.heatmap(data, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Confusion matrix heatmap saved -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: MambaClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = correct = total = 0

    with torch.set_grad_enabled(training):
        for seqs, labels in loader:
            seqs   = seqs.to(device)
            labels = labels.to(device)
            logits = model(seqs)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(-1) == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


def evaluate(
    model: MambaClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = total = 0
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    with torch.no_grad():
        for seqs, labels in loader:
            seqs   = seqs.to(device)
            labels = labels.to(device)
            logits = model(seqs)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(-1)

            total_loss += loss.item() * len(labels)
            total      += len(labels)
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                cm[int(t), int(p)] += 1

    metrics = confusion_metrics(cm)
    metrics["loss"]      = total_loss / total if total > 0 else 0.0
    metrics["confusion"] = cm
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main training script
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = VolleyballDataset(args.csv_dir)
    if len(dataset) == 0:
        print("[ERROR] Dataset empty."); return

    train_idx, val_idx, test_idx = stratified_split(
        dataset, args.val_split, args.test_split, args.seed)
    if not train_idx:
        print("[ERROR] Train split empty."); return

    mean, std = compute_normalization(dataset, train_idx)

    train_ds = AugmentedWrapper(dataset, train_idx, mean, std,
                                augment=True,
                                flip_prob=0.5, jitter_max=3, noise_std=0.01)
    val_ds   = AugmentedWrapper(dataset, val_idx,   mean, std, augment=False) if val_idx  else None
    test_ds  = AugmentedWrapper(dataset, test_idx,  mean, std, augment=False) if test_idx else None

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw) if val_ds   else None
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw) if test_ds  else None

    print(f"\nSplit  ->  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print("Class distribution:")
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        dist = split_distribution(dataset, indices)
        row  = "  ".join(f"{LABEL_NAMES[i][:6]}={dist[LABEL_NAMES[i]]}" for i in range(NUM_CLASSES))
        print(f"  {split_name:5s}: {row}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MambaClassifier(
        input_dim=INPUT_DIM, d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, d_conv=args.d_conv,
        num_classes=NUM_CLASSES, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMambaClassifier  |  params: {n_params:,}  |  d_model={args.d_model}"
          f"  n_layers={args.n_layers}  d_state={args.d_state}")

    # ── Loss / optimiser / scheduler ──────────────────────────────────────────
    class_weights = compute_class_weights(dataset, train_idx, device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1 = -1.0; best_epoch = 0; patience_ctr = 0
    history: list[dict] = []

    print(f"\n{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  {'VlLoss':>8}  {'VlAcc':>7}  {'VlF1':>7}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader:
            vl = evaluate(model, val_loader, criterion, device)
        else:
            vl = {"loss": tr_loss, "accuracy": tr_acc, "macro_f1": tr_acc, "confusion": None}

        scheduler.step()

        vl_f1 = float(vl["macro_f1"])
        print(f"{epoch:6d}  {tr_loss:8.4f}  {tr_acc:7.3f}  "
              f"{vl['loss']:8.4f}  {vl['accuracy']:7.3f}  {vl_f1:7.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl["loss"], "val_acc": vl["accuracy"], "val_f1": vl_f1,
        })

        if vl_f1 >= best_f1:
            best_f1 = vl_f1; best_epoch = epoch; patience_ctr = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "norm_mean": mean, "norm_std": std,
                "args": vars(args), "label_names": LABEL_NAMES,
                "best_val_macro_f1": best_f1,
            }, args.checkpoint)
        else:
            patience_ctr += 1
            if args.patience > 0 and patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
                break

    print(f"\nBest val macro-F1 = {best_f1:.4f}  at epoch {best_epoch}")
    print(f"Checkpoint saved  -> {args.checkpoint}")

    # ── Save training history ─────────────────────────────────────────────────
    if args.history_csv:
        pd.DataFrame(history).to_csv(args.history_csv, index=False)
        print(f"Training history  -> {args.history_csv}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    if test_loader:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        tm = evaluate(model, test_loader, criterion, device)

        print("\n" + "=" * 58)
        print("  HELD-OUT TEST SET RESULTS")
        print("=" * 58)
        print(f"  Accuracy     : {tm['accuracy']:.4f}")
        print(f"  Macro F1     : {tm['macro_f1']:.4f}")
        print(f"  Balanced Acc : {tm['balanced_acc']:.4f}")

        print("\n  Per-Class Metrics:")
        print(f"  {'Class':<22}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'N':>5}")
        print("  " + "-" * 50)
        for name, m in tm["per_class"].items():
            print(f"  {name:<22}  {m['precision']:6.3f}  {m['recall']:6.3f}"
                  f"  {m['f1']:6.3f}  {m['support']:5d}")

        print_confusion_matrix(tm["confusion"])
        save_confusion_heatmap(tm["confusion"], args.checkpoint.replace(".pt", "_cm.png"))
        print("=" * 58)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Mamba volleyball tactic classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv_dir", help="Directory of labelled CSV sequences.")
    p.add_argument("--val_split",        type=float, default=0.15)
    p.add_argument("--test_split",       type=float, default=0.10)
    p.add_argument("--d_model",          type=int,   default=64)
    p.add_argument("--n_layers",         type=int,   default=4)
    p.add_argument("--d_state",          type=int,   default=16)
    p.add_argument("--d_conv",           type=int,   default=4)
    p.add_argument("--dropout",          type=float, default=0.2)
    p.add_argument("--epochs",           type=int,   default=60)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--label_smoothing",  type=float, default=0.1)
    p.add_argument("--num_workers",      type=int,   default=0)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--patience",         type=int,   default=12,
                   help="Early-stopping patience (0 = disabled).")
    p.add_argument("--checkpoint",       default="mamba_checkpoint.pt")
    p.add_argument("--history_csv",      default="training_history.csv")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
