"""
extract_clips.py – Extract labeled rally clips from full match videos.

Reads labels.csv which you create manually by watching the 12 match videos
and writing down the anchor time (moment of ball contact) and tactic for
each rally you want to annotate.

labels.csv format (comma-separated):
  video_file,anchor_time,tactic_class
  match1.mp4,00:02:14.5,coordinated_attack
  match1.mp4,00:05:33.0,spacing_breakdown
  ...

Supported tactic_class values (case-insensitive, spaces or underscores):
  coordinated_attack | coordinated_defense | delayed_support | spacing_breakdown

Each clip is 2 seconds (60 frames at 30 FPS) starting 0.3 s before the anchor,
giving frames for both the training window (1–29) and evaluation window (30–41).

Output folder structure:
  dataset/
    Coordinated_Attack/
    Coordinated_Defense/
    Delayed_Support/
    Spacing_Breakdown/

Usage:
  python extract_clips.py                    # reads labels.csv in current dir
  python extract_clips.py --labels path/to/labels.csv
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
from datetime import timedelta

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Canonical folder names (used for dataset directory structure)
CANONICAL_CLASSES: dict[str, str] = {
    "coordinated_attack":   "Coordinated_Attack",
    "coordinated_defense":  "Coordinated_Defense",
    "coordinated_defence":  "Coordinated_Defense",   # British spelling alias
    "delayed_support":      "Delayed_Support",
    "spacing_breakdown":    "Spacing_Breakdown",
}

CLIP_DURATION_SEC = 2.0     # total clip duration in seconds (≥ 41 frames @ 30 FPS)
PRE_ANCHOR_SEC    = 0.3     # seconds before anchor to start clip

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_anchor(anchor_str: str) -> float:
    """Parse HH:MM:SS.ss or MM:SS.ss into total seconds."""
    parts = anchor_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), float(parts[1])
    else:
        raise ValueError(f"Cannot parse time '{anchor_str}'. Use HH:MM:SS.ss or MM:SS.ss")
    return h * 3600 + m * 60 + s


def _to_ffmpeg_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for ffmpeg -ss / -t arguments."""
    seconds = max(0.0, seconds)
    td = timedelta(seconds=seconds)
    total = td.total_seconds()
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _normalize_tactic(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    return CANONICAL_CLASSES.get(key, "")


def _find_ffmpeg() -> str | None:
    if exe := shutil.which("ffmpeg"):
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _find_video(video_file: str, root_dir: str) -> str | None:
    """Try the path as-is, then relative to root_dir, then scan root_dir."""
    candidates = [
        video_file,
        os.path.join(root_dir, video_file),
        os.path.join(root_dir, os.path.basename(video_file)),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    # Last resort: search root_dir by basename
    base = os.path.basename(video_file)
    for fname in os.listdir(root_dir):
        if fname == base:
            return os.path.abspath(os.path.join(root_dir, fname))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(labels_path: str) -> None:
    print("=" * 60)
    print("  Volleyball Dataset Extraction Pipeline")
    print("=" * 60)

    labels_path = os.path.abspath(labels_path)
    if not os.path.isfile(labels_path):
        print(f"[ERROR] labels.csv not found: {labels_path}")
        return

    root_dir    = os.path.dirname(labels_path)
    dataset_dir = os.path.join(root_dir, "dataset")

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        print("[ERROR] ffmpeg not found. Install ffmpeg or run: pip install imageio-ffmpeg")
        return

    df = pd.read_csv(labels_path)
    required = {"video_file", "anchor_time", "tactic_class"}
    if missing := required - set(df.columns):
        print(f"[ERROR] labels.csv is missing columns: {sorted(missing)}")
        return

    # Create output folders
    for class_dir in CANONICAL_CLASSES.values():
        os.makedirs(os.path.join(dataset_dir, class_dir), exist_ok=True)
    print(f"Output directory : {dataset_dir}\n")

    success = skipped = 0
    clip_duration_str = _to_ffmpeg_time(CLIP_DURATION_SEC)

    for idx, row in df.iterrows():
        raw_video  = str(row["video_file"]).strip()
        anchor_str = str(row["anchor_time"]).strip()
        raw_tactic = str(row["tactic_class"]).strip()

        tactic = _normalize_tactic(raw_tactic)
        if not tactic:
            print(f"[SKIP] Row {idx+1}: Unknown tactic '{raw_tactic}'.")
            skipped += 1
            continue

        video_path = _find_video(raw_video, root_dir)
        if not video_path:
            print(f"[SKIP] Row {idx+1}: Video not found '{raw_video}'.")
            skipped += 1
            continue

        try:
            anchor_sec = _parse_anchor(anchor_str)
        except ValueError as exc:
            print(f"[SKIP] Row {idx+1}: {exc}")
            skipped += 1
            continue

        # Start clip PRE_ANCHOR_SEC before the ball contact event
        start_sec = max(0.0, anchor_sec - PRE_ANCHOR_SEC)
        start_str = _to_ffmpeg_time(start_sec)

        stem      = os.path.splitext(os.path.basename(video_path))[0]
        clip_name = f"{stem}_clip_{idx+1:04d}.mp4"
        clip_path = os.path.join(dataset_dir, tactic, clip_name)

        cmd = [
            ffmpeg, "-y",
            "-ss", start_str,
            "-i", video_path,
            "-t", clip_duration_str,
            "-c:v", "libx264",       # re-encode for accurate frame alignment
            "-an",                    # no audio needed
            "-crf", "18",
            clip_path,
        ]

        print(f"[{idx+1:03d}/{len(df):03d}] {tactic:<22s} @ {start_str} -> {clip_name}")
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            success += 1
        else:
            print(f"         !! ffmpeg failed on row {idx+1}. Check video path/time.")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"  Done.  Extracted: {success}   Skipped: {skipped}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract labeled rally clips from full match videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--labels",
        default="labels.csv",
        help="Path to labels.csv annotation file.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(args.labels)
