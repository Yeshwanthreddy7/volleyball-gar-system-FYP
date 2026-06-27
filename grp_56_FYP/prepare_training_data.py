"""
prepare_training_data.py – Build Mamba Training CSVs from Labeled Clips.

Input layout (from extract_clips.py):
  dataset/
    Coordinated_Attack/    ← *.mp4 clips
    Coordinated_Defense/
    Delayed_Support/
    Spacing_Breakdown/

Processing per clip (back-center camera):
  1. YOLOv11 detects players + ball per frame.
  2. Near-team players filtered: court_y ≥ NET_Y_CM (≥ 900 cm).
  3. ByteTrack assigns persistent player IDs.
  4. Homography maps pixel centroids → court (X, Y) in cm.
  5. 14-feature vector built: [ball_x, ball_y, p1_x, p1_y, …, p6_x, p6_y].
  6. Sliding window (size=29, stride=5) generates multiple CSVs per clip.

Output:
  training_csv/
    Coordinated_Attack__clip_0001_seq00.csv
    ...

Usage:
  python prepare_training_data.py dataset --output-dir training_csv \\
      --yolo-model yolo11n.pt \\
      --court-corners 42,18 1238,18 1238,702 42,702
"""

from __future__ import annotations
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

from mamba_model import (
    COURT_WIDTH_CM, COURT_LENGTH_CM, NET_Y_CM,
    N_PLAYERS, SEQ_LEN, INPUT_DIM, FEATURE_COLS,
    NEAR_TEAM_MIN_Y, NEAR_TEAM_MAX_Y,
)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import supervision as sv
    HAS_SV = True
except ImportError:
    HAS_SV = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FOLDER_TO_LABEL: dict[str, str] = {
    "Coordinated_Attack":  "Coordinated Attack",
    "Coordinated_Defense": "Coordinated Defense",
    "Delayed_Support":     "Delayed Support",
    "Spacing_Breakdown":   "Spacing Breakdown",
}
VIDEO_EXTS   = (".mp4", ".mov", ".avi", ".mkv", ".webm")
STRIDE       = 5     # sliding window step (frames)
PERSON_CLS   = 0     # YOLO COCO class for person
BALL_CLS     = 32    # YOLO COCO class for sports ball
CONF_THRESH  = 0.35  # detection confidence threshold


# ──────────────────────────────────────────────────────────────────────────────
# Homography
# ──────────────────────────────────────────────────────────────────────────────

def build_homography(corners: list[tuple[float, float]]) -> np.ndarray:
    """
    Compute perspective homography from 4 pixel corners → court cm.

    corners order: TL, TR, BR, BL as seen by the back-center camera.
    TL pixel → court (0, 0)              far-left corner (Team B end)
    TR pixel → court (COURT_WIDTH_CM, 0) far-right corner
    BR pixel → court (COURT_WIDTH_CM, COURT_LENGTH_CM)  near-right (Team A end)
    BL pixel → court (0, COURT_LENGTH_CM)               near-left
    """
    src = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0,               0              ],
        [COURT_WIDTH_CM,  0              ],
        [COURT_WIDTH_CM,  COURT_LENGTH_CM],
        [0,               COURT_LENGTH_CM],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def pixel_to_court(
    px: float, py: float,
    frame_w: int, frame_h: int,
    H: np.ndarray | None,
) -> tuple[float, float]:
    """Map a pixel position to court coordinates (cm)."""
    if H is not None:
        pt  = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    # Linear fallback (no homography supplied)
    cx = px / frame_w * COURT_WIDTH_CM
    cy = py / frame_h * COURT_LENGTH_CM
    return cx, cy


# ──────────────────────────────────────────────────────────────────────────────
# Detection & Tracking helpers
# ──────────────────────────────────────────────────────────────────────────────

def detect_frame(
    frame: np.ndarray,
    yolo: "YOLO",
    conf: float = CONF_THRESH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Run YOLOv11 on one frame.
    Returns:
      person_xyxy : (N,4)  bounding boxes of detected people
      person_conf : (N,)   confidence scores
      ball_center : (2,)   pixel center of highest-conf ball, or None
    """
    results  = yolo(frame, verbose=False, conf=conf)[0]
    boxes    = results.boxes
    cls_ids  = boxes.cls.int().cpu().numpy()
    xyxy     = boxes.xyxy.cpu().numpy()
    confs    = boxes.conf.cpu().numpy()

    p_mask   = cls_ids == PERSON_CLS
    p_xyxy   = xyxy[p_mask]
    p_conf   = confs[p_mask]

    ball_center: np.ndarray | None = None
    b_mask = cls_ids == BALL_CLS
    if b_mask.any():
        b_boxes = xyxy[b_mask]
        b_confs = confs[b_mask]
        best    = int(np.argmax(b_confs))
        x1, y1, x2, y2 = b_boxes[best]
        ball_center = np.array([(x1+x2)/2, (y1+y2)/2], dtype=float)

    return p_xyxy, p_conf, ball_center


def filter_near_team(
    xyxy: np.ndarray, conf: np.ndarray,
    frame_w: int, frame_h: int,
    H: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only players physically inside Team A's court half.

    Uses foot point (bottom-centre of bbox) so a setter at the net is not
    incorrectly dropped.  X/Y court bounds exclude referees and bench staff.
    """
    if len(xyxy) == 0:
        return xyxy, conf
    keep = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        foot_x = (x1 + x2) / 2
        foot_y = y2
        cx, cy = pixel_to_court(foot_x, foot_y, frame_w, frame_h, H)
        if (cy >= NEAR_TEAM_MIN_Y and cy <= COURT_LENGTH_CM
                and 0.0 <= cx <= COURT_WIDTH_CM):
            keep.append(i)
    if not keep:
        return np.empty((0,4), dtype=xyxy.dtype), np.empty((0,), dtype=conf.dtype)
    idx = np.array(keep, dtype=int)
    return xyxy[idx], conf[idx]


def extract_positions(
    tracks: "sv.Detections",
    ball_px: np.ndarray | None,
    frame_w: int, frame_h: int,
    H: np.ndarray | None,
    track_age: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ByteTrack detections → court positions.
    Returns:
      player_pos : (N_PLAYERS, 2) court cm  (centroid-filled when fewer than 6 detected)
      ball_pos   : (2,) court cm            (zero if ball unseen)
    """
    if tracks.tracker_id is not None:
        for tid in tracks.tracker_id:
            track_age[int(tid)] = track_age.get(int(tid), 0) + 1

    centers: dict[int, tuple[float, float]] = {}
    if tracks.tracker_id is not None and len(tracks.xyxy) > 0:
        for tid, (x1,y1,x2,y2) in zip(tracks.tracker_id, tracks.xyxy):
            centers[int(tid)] = ((x1+x2)/2, (y1+y2)/2)

    chosen = sorted(centers, key=lambda t: track_age.get(t,0), reverse=True)[:N_PLAYERS]

    player_pos = np.zeros((N_PLAYERS, 2), dtype=float)
    for slot, tid in enumerate(chosen):
        px, py = centers[tid]
        cx, cy = pixel_to_court(px, py, frame_w, frame_h, H)
        player_pos[slot] = [cx, cy]

    # Fill undetected slots with team centroid to avoid phantom players at (0,0).
    n_detected = len(chosen)
    if 0 < n_detected < N_PLAYERS:
        player_pos[n_detected:] = player_pos[:n_detected].mean(axis=0)

    if ball_px is not None:
        bx, by = pixel_to_court(float(ball_px[0]), float(ball_px[1]), frame_w, frame_h, H)
    else:
        bx, by = 0.0, 0.0

    return player_pos, np.array([bx, by], dtype=float)


def build_feature_row(player_pos: np.ndarray, ball_pos: np.ndarray) -> np.ndarray:
    """Build 14-element feature vector [ball_x, ball_y, p1_x, p1_y, …, p6_x, p6_y]."""
    row = np.empty(INPUT_DIM, dtype=np.float32)
    row[0] = ball_pos[0]
    row[1] = ball_pos[1]
    for i in range(N_PLAYERS):
        row[2 + i*2]     = player_pos[i, 0]
        row[2 + i*2 + 1] = player_pos[i, 1]
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Clip → multiple CSVs (sliding window)
# ──────────────────────────────────────────────────────────────────────────────

def process_clip(
    clip_path: str,
    label: str,
    yolo: "YOLO",
    H: np.ndarray | None,
    out_dir: str,
    clip_stem: str,
    stride: int = STRIDE,
) -> int:
    """
    Extract all 29-frame windows (stride=stride) from one clip.
    Returns the number of CSV files written.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return 0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker   = sv.ByteTrack()
    track_age: dict[int, int] = {}
    all_rows:  list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p_xyxy, p_conf, ball_px = detect_frame(frame, yolo)
        p_xyxy, p_conf          = filter_near_team(p_xyxy, p_conf, frame_w, frame_h, H)

        dets = sv.Detections(
            xyxy=p_xyxy,
            confidence=p_conf,
            class_id=np.zeros(len(p_xyxy), dtype=int),
        ) if len(p_xyxy) > 0 else sv.Detections.empty()
        tracks    = tracker.update_with_detections(dets)

        player_pos, ball_pos = extract_positions(
            tracks, ball_px, frame_w, frame_h, H, track_age)
        all_rows.append(build_feature_row(player_pos, ball_pos))

    cap.release()

    if len(all_rows) < SEQ_LEN:
        return 0

    frames = np.stack(all_rows)   # (total_frames, 14)
    n_written = 0

    for start in range(0, len(frames) - SEQ_LEN + 1, stride):
        window = frames[start: start + SEQ_LEN]   # (29, 14)
        df = pd.DataFrame(window, columns=FEATURE_COLS)
        df.insert(0, "frame_id", np.arange(1, SEQ_LEN + 1, dtype=int))
        df["target_label"] = label

        seq_idx  = n_written
        out_name = f"{clip_stem}_seq{seq_idx:03d}.csv"
        df.to_csv(os.path.join(out_dir, out_name), index=False)
        n_written += 1

    return n_written


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if not HAS_YOLO:
        print("[ERROR] ultralytics not installed.  pip install ultralytics>=8.3.0")
        sys.exit(1)
    if not HAS_SV:
        print("[ERROR] supervision not installed.  pip install supervision>=0.21.0")
        sys.exit(1)

    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir     = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.clean_output:
        removed = sum(1 for f in os.listdir(out_dir)
                      if f.lower().endswith(".csv")
                      and not os.remove(os.path.join(out_dir, f)))
        print(f"Cleaned {removed} old CSV files.")

    # Build homography
    H: np.ndarray | None = None
    if args.court_corners:
        H = build_homography(args.court_corners)
        print("Homography: using supplied court corner points.")
    else:
        print("Homography: linear pixel scaling (no corners supplied).")

    # Collect clips
    clips: list[tuple[str, str]] = []     # (clip_path, label_name)
    for folder, label in FOLDER_TO_LABEL.items():
        class_dir = os.path.join(dataset_dir, folder)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(VIDEO_EXTS):
                clips.append((os.path.join(class_dir, fname), label))

    if not clips:
        print(f"[ERROR] No video clips found under {dataset_dir}")
        sys.exit(1)

    if args.max_clips > 0:
        clips = clips[:args.max_clips]

    print(f"\nLoading YOLOv11: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    print(f"Processing {len(clips)} clips → {out_dir}\n")
    total_csvs = total_skipped = 0

    for i, (clip_path, label) in enumerate(clips, start=1):
        stem       = os.path.splitext(os.path.basename(clip_path))[0]
        folder     = os.path.basename(os.path.dirname(clip_path))
        clip_stem  = f"{folder}__{stem}"

        n = process_clip(clip_path, label, yolo, H, out_dir, clip_stem, args.stride)
        if n == 0:
            print(f"[{i:3d}/{len(clips)}] SKIP {os.path.basename(clip_path)}")
            total_skipped += 1
        else:
            print(f"[{i:3d}/{len(clips)}] OK   {os.path.basename(clip_path)}  → {n} sequences")
            total_csvs += n

    print(f"\nDone.  CSV sequences written: {total_csvs}  |  Clips skipped: {total_skipped}")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

class _CornersAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            pairs = [tuple(float(v) for v in s.split(",")) for s in values]
            if len(pairs) != 4 or any(len(p) != 2 for p in pairs):
                raise ValueError
        except ValueError:
            parser.error("--court-corners needs 4 x,y pairs e.g. 42,18 1238,18 1238,702 42,702")
        setattr(namespace, self.dest, pairs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare Mamba training CSVs from labeled volleyball clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset_dir", help="Root directory with class sub-folders of clips.")
    p.add_argument("--output-dir",   default="training_csv")
    p.add_argument("--yolo-model",   default="yolo11n.pt",
                   help="YOLOv11 weights (downloads if not present).")
    p.add_argument("--court-corners", nargs=4, metavar=("TL","TR","BR","BL"),
                   action=_CornersAction, default=None,
                   help="4 pixel corners TL TR BR BL for homography.")
    p.add_argument("--stride",       type=int, default=STRIDE,
                   help="Sliding window stride (frames).")
    p.add_argument("--clean-output", action="store_true")
    p.add_argument("--max-clips",    type=int, default=0,
                   help="Limit clips processed (0 = all).")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
