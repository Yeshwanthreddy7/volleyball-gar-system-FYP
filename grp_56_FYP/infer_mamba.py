"""
infer_mamba.py – Standalone Mamba Inference on a Single Feature-CSV or Video.

Two modes:
  1. CSV mode  : classify a single 29-row feature CSV  (quick model test)
  2. Video mode: full pipeline on a match video         (same as pipeline.py
                 but with simplified output)

Usage (CSV):
  python infer_mamba.py --csv clip_001.csv --checkpoint mamba_checkpoint.pt

Usage (Video):
  python infer_mamba.py --video match.mp4 --checkpoint mamba_checkpoint.pt \\
      --court-corners 42,18 1238,18 1238,702 42,702
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mamba_model import (
    FEATURE_COLS, INPUT_DIM, LABEL_NAMES, NUM_CLASSES,
    LABEL_TO_IDX, SEQ_LEN, MambaClassifier,
)


# ──────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sa   = ckpt.get("args", {})
    model = MambaClassifier(
        input_dim=sa.get("input_dim", INPUT_DIM),
        d_model=sa.get("d_model", 64),
        n_layers=sa.get("n_layers", 4),
        d_state=sa.get("d_state", 16),
        d_conv=sa.get("d_conv", 4),
        num_classes=NUM_CLASSES,
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    mean = ckpt["norm_mean"].to(device)
    std  = ckpt["norm_std"].to(device)
    print(f"Checkpoint loaded : {path}")
    print(f"  Best val macro-F1 = {ckpt.get('best_val_macro_f1', 'N/A')}")
    return model, mean, std


# ──────────────────────────────────────────────────────────────────────────────
# CSV mode
# ──────────────────────────────────────────────────────────────────────────────

def infer_csv(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_checkpoint(args.checkpoint, device)

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] CSV missing columns: {missing}")
        sys.exit(1)

    features = df[FEATURE_COLS].values.astype(np.float32)
    if len(features) < SEQ_LEN:
        pad      = np.zeros((SEQ_LEN - len(features), INPUT_DIM), np.float32)
        features = np.vstack([features, pad])
    else:
        features = features[:SEQ_LEN]

    t     = torch.from_numpy(features).to(device)
    t     = (t - mean) / std
    with torch.no_grad():
        probs = F.softmax(model(t.unsqueeze(0)), dim=-1)[0]

    pred_idx = int(probs.argmax().item())
    label    = LABEL_NAMES[pred_idx]
    conf     = float(probs[pred_idx].item())

    print(f"\n{'='*50}")
    print(f"  Input  : {args.csv}")
    print(f"  Result : [{LABEL_TO_IDX[label]}] {label}")
    print(f"  Conf   : {conf:.4f}")
    print(f"\n  Class probabilities:")
    for i, name in enumerate(LABEL_NAMES):
        bar = "█" * int(probs[i].item() * 30)
        print(f"  {name:<24s} {probs[i].item():6.4f}  {bar}")
    print(f"{'='*50}")


# ──────────────────────────────────────────────────────────────────────────────
# Video mode (delegates to pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

def infer_video(args: argparse.Namespace) -> None:
    from pipeline import run_pipeline
    # Build a namespace matching pipeline.py's expected args
    pipe_args = argparse.Namespace(
        video=args.video,
        checkpoint=args.checkpoint,
        yolo_model=args.yolo_model,
        conf_threshold=0.35,
        court_corners=args.court_corners,
        stride=args.stride,
        output_video=args.output_video,
        output_csv=args.output_csv,
    )
    run_pipeline(pipe_args)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

class _CornersAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            pairs = [tuple(float(v) for v in s.split(",")) for s in values]
            if len(pairs) != 4 or any(len(p) != 2 for p in pairs):
                raise ValueError
        except ValueError:
            parser.error("Need 4 x,y pairs e.g. 42,18 1238,18 1238,702 42,702")
        setattr(namespace, self.dest, pairs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone Mamba inference (CSV or video).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",   help="Single 29-row feature CSV to classify.")
    group.add_argument("--video", help="Match video for full pipeline inference.")

    p.add_argument("--checkpoint",     required=True,
                   help="Trained Mamba checkpoint (.pt).")
    p.add_argument("--yolo-model",     default="yolo11n.pt")
    p.add_argument("--court-corners",  nargs=4, metavar=("TL","TR","BR","BL"),
                   action=_CornersAction, default=None)
    p.add_argument("--stride",         type=int, default=15)
    p.add_argument("--output-video",   default="annotated.mp4")
    p.add_argument("--output-csv",     default="predictions.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if args.csv:
        infer_csv(args)
    else:
        infer_video(args)
