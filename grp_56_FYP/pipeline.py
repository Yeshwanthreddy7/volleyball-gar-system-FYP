"""
pipeline.py – End-to-End Volleyball Tactical Analysis Pipeline
           (Back-Center End-Line Camera View)

Complete processing chain:
  Match Video (.mp4)
      ->  Frame Extraction
      ->  YOLOv11 Detection  (players + ball)
      ->  Near-Team Filter   (Team A only: court_y ≥ NET_Y_CM)
      ->  ByteTrack          (persistent player IDs)
      ->  Homography         (pixel -> court cm coordinates)
      ->  14-Feature Vector  [ball_x, ball_y, p1_x, p1_y, …, p6_x, p6_y]
      ->  Sliding Window     (29 frames, stride configurable)
      ->  Mamba Classifier   (4-class: Attack / Defense / Delayed / Spacing)
      ->  Frame Annotation   (bounding boxes, ball, tactic banner)
      ->  Annotated Video + Predictions CSV

Court coordinate system (after homography):
  X : 0–900 cm   (9m width: left sideline -> right sideline)
  Y : 0–1800 cm  (18m length: far end Y=0 -> camera end Y=1800)
  Net at Y = 900 cm
  Team A (near, tracked): Y = 900–1800 cm
  Team B (far):           Y = 0–900 cm

Usage:
  python pipeline.py match.mp4 mamba_checkpoint.pt \\
      --court-corners 42,18 1238,18 1238,702 42,702 \\
      --output-video annotated.mp4 --output-csv predictions.csv

Note: --court-corners are the 4 pixel coordinates of the court's boundary
      corners as seen by the back-center camera, in TL TR BR BL order.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


def _pt(x, y) -> np.ndarray:
    """Convert coordinates to np.int32 array — OpenCV 4.13 + NumPy 2.x compatible.

    Clamps to ±30000 so a runaway kinematic predictor cannot cause OverflowError
    when np.array tries to fit the value into int32 (max 2.1 billion, but we limit
    to 30000 which is well above any display resolution).
    """
    _CLAMP = 30000
    xi = max(-_CLAMP, min(_CLAMP, int(round(float(x)))))
    yi = max(-_CLAMP, min(_CLAMP, int(round(float(y)))))
    return np.array([xi, yi], dtype=np.int32)

from mamba_model import (
    COURT_WIDTH_CM, COURT_LENGTH_CM, NET_Y_CM,
    NEAR_TEAM_MIN_Y,
    N_PLAYERS, SEQ_LEN, INPUT_DIM, FEATURE_COLS,
    LABEL_NAMES, LABEL_TO_IDX, NUM_CLASSES,
    MambaClassifier,
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
# Visual style constants
# ──────────────────────────────────────────────────────────────────────────────

FONT    = cv2.FONT_HERSHEY_SIMPLEX
FOURCC  = cv2.VideoWriter_fourcc(*"mp4v")

# BGR colors per tactic label
TACTIC_COLORS: dict[str, tuple[int, int, int]] = {
    "Coordinated Attack":  (0,   0,   255),   # RED     (BGR)
    "Coordinated Defense": (255, 0,   0  ),   # BLUE    (BGR)
    "Delayed Support":     (0,   255, 255),   # YELLOW  (BGR)
    "Spacing Breakdown":   (0,   255, 0  ),   # GREEN   (BGR)
}
DEFAULT_BOX_COLOR = (180, 180, 180)         # grey (before first classification)
BALL_BOX_COLOR    = (180, 105, 255)         # PINK — fixed, never changes with tactic
BALL_BOX_HALF     = 18                      # half-width of ball bounding box (pixels)
TEXT_COLOR        = (255, 255, 255)         # white

DISPLAY_NAMES: dict[str, str] = {
    "Coordinated Attack":  "COORDINATED ATTACK",
    "Coordinated Defense": "COORDINATED DEFENSE",
    "Delayed Support":     "DELAYED SUPPORT",
    "Spacing Breakdown":   "SPACING BREAKDOWN",
}

BALL_TRAIL_LEN      = 40
BALL_MAX_MISS       = 60          # 2 s at 30 fps before ball is considered lost
CONF_THRESHOLD      = 0.35
BALL_CONF_THRESHOLD = 0.15        # lower threshold — volleyball is small and fast

PERSON_CLS = 0
BALL_CLS   = 32


# ──────────────────────────────────────────────────────────────────────────────
# Ball tracker (constant-velocity bridge for missed detections)
# ──────────────────────────────────────────────────────────────────────────────

class BallTracker:
    """
    Lightweight ball tracker using a constant-velocity kinematic model.
    Bridges frames where YOLOv11 misses the fast-moving volleyball.
    """

    def __init__(self, max_miss: int = BALL_MAX_MISS, alpha: float = 0.55) -> None:
        self._max_miss = max_miss
        self._alpha    = alpha
        self._pos: np.ndarray | None = None
        self._vel: np.ndarray = np.zeros(2)
        self._miss: int = 0

    def update(
        self, detected: np.ndarray | None
    ) -> tuple[np.ndarray | None, bool]:
        """
        Returns (position, is_predicted).
        position    : (2,) pixel coords, or None if ball is lost.
        is_predicted: True when position is extrapolated (no YOLO detection).
        """
        if detected is not None:
            if self._pos is not None:
                raw = detected.astype(float) - self._pos
                self._vel = self._alpha * raw + (1 - self._alpha) * self._vel
            self._pos  = detected.astype(float).copy()
            self._miss = 0
            return self._pos.copy(), False

        self._miss += 1
        if self._miss > self._max_miss or self._pos is None:
            return None, False
        self._pos = self._pos + self._vel
        return self._pos.copy(), True

    def reset(self) -> None:
        self._pos = None; self._vel = np.zeros(2); self._miss = 0


# ──────────────────────────────────────────────────────────────────────────────
# Player ID Mapper  (ByteTrack IDs -> sequential display IDs 1-6)
# ──────────────────────────────────────────────────────────────────────────────

class PlayerIDMapper:
    """
    Assigns stable P1-P6 labels via Hungarian (linear-sum) assignment.

    Every frame, ALL current detections are matched against ALL known slot
    positions in one global optimisation step that minimises total centroid
    distance.  ByteTrack's internal IDs are IGNORED for slot assignment —
    this eliminates label swaps caused by ByteTrack re-numbering players
    when they cross paths or dive.
    """

    def __init__(self, n: int = N_PLAYERS, max_dist: float = 250.0) -> None:
        self.n        = n
        self.max_dist = max_dist          # pixels; detections farther than this
                                          # from every slot start a new slot
        self._slot_box: dict[int, tuple] = {}   # slot -> last (x1,y1,x2,y2)

    @staticmethod
    def _cx(b: tuple) -> float: return (b[0] + b[2]) / 2.0
    @staticmethod
    def _cy(b: tuple) -> float: return (b[1] + b[3]) / 2.0

    def update(
        self, tracks: "sv.Detections"
    ) -> tuple[dict[int, tuple], set[int]]:
        """
        Returns:
          slot_boxes  : dict  slot(1-N) -> (x1, y1, x2, y2)
          active_slots: set   which slots are detected this frame
        """
        slot_boxes:   dict[int, tuple] = {}
        active_slots: set[int]         = set()

        if tracks.tracker_id is None or len(tracks.tracker_id) == 0:
            return slot_boxes, active_slots

        boxes = [tuple(float(v) for v in b) for b in tracks.xyxy]
        n_det = len(boxes)

        # ── First frame: assign slots in detection order left→right ───────────
        if not self._slot_box:
            sorted_boxes = sorted(boxes[:self.n], key=lambda b: self._cx(b))
            for i, box in enumerate(sorted_boxes):
                self._slot_box[i + 1] = box
                slot_boxes[i + 1]     = box
                active_slots.add(i + 1)
            return slot_boxes, active_slots

        # ── Build centroid-distance cost matrix (slots × detections) ──────────
        slots   = sorted(self._slot_box)
        n_slots = len(slots)
        cost    = np.empty((n_slots, n_det), dtype=float)
        for i, s in enumerate(slots):
            sx = self._cx(self._slot_box[s])
            sy = self._cy(self._slot_box[s])
            for j, b in enumerate(boxes):
                dx = self._cx(b) - sx
                dy = self._cy(b) - sy
                cost[i, j] = (dx * dx + dy * dy) ** 0.5

        # ── Hungarian assignment — globally optimal matching ───────────────────
        row_idx, col_idx = linear_sum_assignment(cost)
        assigned_dets: set[int] = set()
        for ri, ci in zip(row_idx, col_idx):
            if cost[ri, ci] <= self.max_dist:
                slot = slots[ri]
                self._slot_box[slot] = boxes[ci]
                slot_boxes[slot]     = boxes[ci]
                active_slots.add(slot)
                assigned_dets.add(ci)

        # ── Unmatched detections → fill any remaining free slots ──────────────
        occupied = set(self._slot_box)
        for j, box in enumerate(boxes):
            if j in assigned_dets:
                continue
            for s in range(1, self.n + 1):
                if s not in occupied:
                    self._slot_box[s] = box
                    slot_boxes[s]     = box
                    active_slots.add(s)
                    occupied.add(s)
                    break   # one new slot per unmatched detection

        return slot_boxes, active_slots


# ──────────────────────────────────────────────────────────────────────────────
# Homography
# ──────────────────────────────────────────────────────────────────────────────

def build_homography(corners: list[tuple[float, float]]) -> np.ndarray:
    """
    Perspective homography from 4 pixel corners -> court cm.

    corners (TL, TR, BR, BL pixel coords) map to:
      TL -> (0,               0              )   far-left (Team B end)
      TR -> (COURT_WIDTH_CM,  0              )   far-right
      BR -> (COURT_WIDTH_CM,  COURT_LENGTH_CM)   near-right (Team A end)
      BL -> (0,               COURT_LENGTH_CM)   near-left
    """
    src = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0,              0              ],
        [COURT_WIDTH_CM, 0              ],
        [COURT_WIDTH_CM, COURT_LENGTH_CM],
        [0,              COURT_LENGTH_CM],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def pixel_to_court(
    px: float, py: float,
    frame_w: int, frame_h: int,
    H: np.ndarray | None,
) -> tuple[float, float]:
    """Map a pixel (px, py) to court coordinates (cx, cy) in cm."""
    if H is not None:
        pt  = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    # Linear fallback
    return px / frame_w * COURT_WIDTH_CM, py / frame_h * COURT_LENGTH_CM


# ──────────────────────────────────────────────────────────────────────────────
# Detection (YOLOv11)
# ──────────────────────────────────────────────────────────────────────────────

def detect_frame(
    frame: np.ndarray,
    yolo: "YOLO",
    conf: float = CONF_THRESHOLD,
    ball_conf: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Run YOLOv11 on one frame.
    Returns: (person_xyxy, person_conf, ball_center_pixel or None)

    ball_conf: separate lower threshold for ball detection.  If None, uses conf.
    Run YOLO at min(conf, ball_conf) so both thresholds are honoured in one pass.
    """
    if ball_conf is None:
        ball_conf = conf
    run_conf = min(conf, ball_conf)

    results = yolo(frame, verbose=False, conf=run_conf)[0]
    boxes   = results.boxes
    cls_ids = boxes.cls.int().cpu().numpy()
    xyxy    = boxes.xyxy.cpu().numpy()
    confs   = boxes.conf.cpu().numpy()

    # Person detections at the higher person confidence threshold
    p_mask  = (cls_ids == PERSON_CLS) & (confs >= conf)
    p_xyxy  = xyxy[p_mask]
    p_conf  = confs[p_mask]

    # Ball detection at the lower ball confidence threshold
    ball: np.ndarray | None = None
    b_mask = (cls_ids == BALL_CLS) & (confs >= ball_conf)
    if b_mask.any():
        bx   = xyxy[b_mask]
        bc   = confs[b_mask]
        best = int(np.argmax(bc))
        x1, y1, x2, y2 = bx[best]
        ball = np.array([(x1+x2)/2, (y1+y2)/2], dtype=float)

    return p_xyxy, p_conf, ball


# ──────────────────────────────────────────────────────────────────────────────
# Near-team player filter
# ──────────────────────────────────────────────────────────────────────────────

def filter_near_team(
    xyxy: np.ndarray, conf: np.ndarray,
    frame_w: int, frame_h: int,
    H: np.ndarray | None,
    min_y_cm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only players physically inside Team A's court half.

    Uses both the foot point AND the body centre so that players who are
    jumping (foot temporarily above the net line) are not filtered out.
    A player passes if their foot OR their centre is in the near-team half.
    """
    thresh = NEAR_TEAM_MIN_Y if min_y_cm is None else min_y_cm
    # Allow tolerance for jumping players: accept if centre is within 200 cm of net
    thresh_centre = max(0, thresh - 200)
    if len(xyxy) == 0:
        return xyxy, conf
    keep = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        foot_x = (x1 + x2) / 2
        foot_y = y2
        cen_y  = (y1 + y2) / 2
        cx,  foot_cy = pixel_to_court(foot_x, foot_y, frame_w, frame_h, H)
        _,   cen_cy  = pixel_to_court(foot_x, cen_y,  frame_w, frame_h, H)
        in_near = (
            (foot_cy >= thresh        and foot_cy <= COURT_LENGTH_CM) or
            (cen_cy  >= thresh_centre and cen_cy  <= COURT_LENGTH_CM)
        )
        in_x = (0.0 <= cx <= COURT_WIDTH_CM)
        if in_near and in_x:
            keep.append(i)
    if not keep:
        return np.empty((0,4), dtype=xyxy.dtype), np.empty((0,), dtype=conf.dtype)
    idx = np.array(keep, dtype=int)
    return xyxy[idx], conf[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Position extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_positions(
    tracks: "sv.Detections",
    ball_px: np.ndarray | None,
    frame_w: int, frame_h: int,
    H: np.ndarray | None,
    track_age: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ByteTrack detections -> court positions.

    Returns:
      player_pos : (N_PLAYERS, 2) court cm  [centroid-filled if fewer than 6 detected]
      ball_pos   : (2,) court cm            [zero if ball not detected]
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

    # Fill undetected slots with team centroid so the model never receives
    # phantom players at court (0,0) — Team B's far corner.
    n_detected = len(chosen)
    if 0 < n_detected < N_PLAYERS:
        player_pos[n_detected:] = player_pos[:n_detected].mean(axis=0)

    if ball_px is not None:
        bx, by = pixel_to_court(float(ball_px[0]), float(ball_px[1]), frame_w, frame_h, H)
    else:
        bx, by = 0.0, 0.0

    return player_pos, np.array([bx, by], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_row(
    player_pos: np.ndarray, ball_pos: np.ndarray
) -> np.ndarray:
    """Build 14-element feature vector [ball_x, ball_y, p1_x, p1_y, …, p6_x, p6_y]."""
    row = np.empty(INPUT_DIM, dtype=np.float32)
    row[0] = ball_pos[0]; row[1] = ball_pos[1]
    for i in range(N_PLAYERS):
        row[2 + i*2]     = player_pos[i, 0]
        row[2 + i*2 + 1] = player_pos[i, 1]
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Mamba classification
# ──────────────────────────────────────────────────────────────────────────────

def load_mamba(
    path: str, device: torch.device
) -> tuple[MambaClassifier, torch.Tensor, torch.Tensor]:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    sa    = ckpt.get("args", {})
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
    return model, ckpt["norm_mean"].to(device), ckpt["norm_std"].to(device)


def classify_sequence(
    seq:   np.ndarray,          # (29, 14)
    model: MambaClassifier,
    mean:  torch.Tensor,
    std:   torch.Tensor,
    device: torch.device,
) -> tuple[str, int, float, torch.Tensor]:
    """Return (label_str, 1-based-index, confidence, probs_tensor)."""
    t     = (torch.from_numpy(seq).to(device) - mean) / std
    with torch.no_grad():
        probs = F.softmax(model(t.unsqueeze(0)), dim=-1)[0]
    idx   = int(probs.argmax().item())
    label = LABEL_NAMES[idx]
    return label, LABEL_TO_IDX.get(label, 0), float(probs[idx].item()), probs


# ──────────────────────────────────────────────────────────────────────────────
# Frame annotation
# ──────────────────────────────────────────────────────────────────────────────

def _draw_rounded_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple, thickness: int = 2, r: int = 8,
) -> None:
    """Draw a rounded rectangle for player bounding boxes."""
    cv2.line(img, _pt(x1+r, y1), _pt(x2-r, y1), color, thickness)
    cv2.line(img, _pt(x1+r, y2), _pt(x2-r, y2), color, thickness)
    cv2.line(img, _pt(x1, y1+r), _pt(x1, y2-r), color, thickness)
    cv2.line(img, _pt(x2, y1+r), _pt(x2, y2-r), color, thickness)
    cv2.ellipse(img, _pt(x1+r, y1+r), (int(r), int(r)), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, _pt(x2-r, y1+r), (int(r), int(r)), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, _pt(x1+r, y2-r), (int(r), int(r)),  90, 0, 90,  color, thickness)
    cv2.ellipse(img, _pt(x2-r, y2-r), (int(r), int(r)),   0, 0, 90,  color, thickness)


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS.ss for on-screen display."""
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def _draw_log_panel(out: np.ndarray, recent_preds: list, h: int, w: int) -> None:
    """
    Draw a bottom-of-frame classification history panel that mirrors the terminal.
    Shows up to 8 most recent: #Seq  Time  [Index] Activity  Conf
    Most recent entry at the top of the panel.
    """
    LOG_SC, LOG_TH = 0.44, 1
    HDR_SC, HDR_TH = 0.48, 1
    PAD  = 8
    ACNT = 5    # left accent bar width per row

    (_, row_h), _ = cv2.getTextSize("Ag", FONT, LOG_SC, LOG_TH)
    (_, hdr_h), _ = cv2.getTextSize("Ag", FONT, HDR_SC, HDR_TH)
    row_h = int(row_h); hdr_h = int(hdr_h)

    n_rows  = min(len(recent_preds), 8)
    panel_h = hdr_h + n_rows * (row_h + 5) + PAD * 3 + 6

    px1 = 10
    py2 = int(h) - 10
    py1 = py2 - panel_h
    px2 = int(w) - 10

    # Semi-transparent dark background
    overlay = out.copy()
    cv2.rectangle(overlay, _pt(px1, py1), _pt(px2, py2), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)

    # Header row
    hdr_txt = "CLASSIFICATION LOG    #Seq    Time       [Idx] Activity                  Conf"
    cv2.putText(out, hdr_txt, _pt(px1 + PAD + ACNT + 4, py1 + PAD + hdr_h),
                FONT, HDR_SC, (200, 200, 200), HDR_TH, cv2.LINE_AA)

    # Thin separator line
    div_y = py1 + PAD + hdr_h + 5
    cv2.line(out, _pt(px1 + PAD, div_y), _pt(px2 - PAD, div_y), (70, 70, 70), 1)

    ty = div_y + row_h + 5
    for pred in recent_preds[:8]:
        seq_n = pred["seq"]
        t_str = _fmt_time(pred["time"])
        lbl   = pred["label"]
        cnf   = pred["conf"]
        num   = pred.get("numeric", 0)
        col   = TACTIC_COLORS.get(lbl, (160, 160, 160))

        # Coloured accent bar
        cv2.rectangle(out,
                      _pt(px1 + PAD, ty - row_h + 2),
                      _pt(px1 + PAD + ACNT, ty + 2),
                      col, -1)

        # Row text
        row_txt = f"#{seq_n:<5d}  {t_str}   [{num}]  {lbl:<30s}  {cnf:.3f}"
        cv2.putText(out, row_txt, _pt(px1 + PAD + ACNT + 6, ty),
                    FONT, LOG_SC, col, LOG_TH, cv2.LINE_AA)
        ty += row_h + 5


def annotate_frame(
    frame:               np.ndarray,
    display_boxes:       dict,
    active_slots:        set,
    ball_pixel:          np.ndarray | None,
    ball_predicted:      bool,
    ball_trail:          "deque",
    label:               str,
    conf:                float,
    frame_idx:           int,
    fps:                 float,
    activity_start_time: float = 0.0,
    recent_preds:        list  = None,
) -> np.ndarray:
    """
    Draw on frame:
      • Rounded bounding boxes per player — colour changes with each tactic:
            Yellow  = Coordinated Attack
            Orange  = Coordinated Defense
            Pink    = Delayed Support
            Red     = Spacing Breakdown
      • Player ID badge above each box
      • Ball bounding box — always WHITE, never changes
      • Ball motion trail
      • Top-left  : frame counter + current timestamp
      • Top-right : activity panel  (name, started-at time, confidence)
    """
    out = frame.copy()
    h, w = int(out.shape[0]), int(out.shape[1])   # force Python int — cv2 rejects np.intp

    box_color = TACTIC_COLORS.get(label, DEFAULT_BOX_COLOR) if label else DEFAULT_BOX_COLOR

    # ── Ball motion trail (numpy drawing to avoid cv2.circle type issues) ───────
    trail = list(ball_trail)
    n = len(trail)
    for i, (tx, ty, pred) in enumerate(trail):
        try:
            px, py = int(round(float(tx))), int(round(float(ty)))
        except Exception:
            continue
        if not (0 <= px < w and 0 <= py < h):
            continue
        alpha  = float(i + 1) / max(n, 1)
        r      = max(1, int(round(3.0 * alpha)))
        bright = int(110 * alpha) if pred else int(230 * alpha)
        x1s = max(0, px - r); x2s = min(w, px + r + 1)
        y1s = max(0, py - r); y2s = min(h, py + r + 1)
        out[y1s:y2s, x1s:x2s] = [bright, bright, bright]

    # ── Player bounding boxes (slots P1–P6, persistent across frames) ────────
    for slot in range(1, N_PLAYERS + 1):
        if slot not in display_boxes:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in display_boxes[slot]]
        is_active = slot in active_slots
        # Active detection -> full activity colour; ghost (predicted) -> dimmed
        if is_active:
            color = box_color
            thick = 2
        else:
            color = tuple(min(255, max(0, c // 2 + 50)) for c in box_color)
            thick = 1
        _draw_rounded_rect(out, x1, y1, x2, y2, color, thickness=thick)

        badge_txt = f"P{slot}"
        (tw, th), _ = cv2.getTextSize(badge_txt, FONT, 0.50, 1)
        bx1 = x1 - 1;        by1 = y1 - int(th) - 8
        bx2 = x1 + int(tw) + 6; by2 = y1 - 1
        cv2.rectangle(out, _pt(bx1, by1), _pt(bx2, by2), color, -1)
        cv2.putText(out, badge_txt, _pt(x1 + 3, y1 - 5),
                    FONT, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Ball bounding box (fixed white — never changes with tactic) ───────────
    if ball_pixel is not None:
        cx = int(round(float(ball_pixel[0])))
        cy = int(round(float(ball_pixel[1])))
        bx1 = max(0, cx - BALL_BOX_HALF)
        by1 = max(0, cy - BALL_BOX_HALF)
        bx2 = min(w - 1, cx + BALL_BOX_HALF)
        by2 = min(h - 1, cy + BALL_BOX_HALF)
        ball_thick = 1 if ball_predicted else 2
        cv2.rectangle(out, _pt(bx1, by1), _pt(bx2, by2), BALL_BOX_COLOR, ball_thick, cv2.LINE_AA)
        lbl_txt = "BALL~" if ball_predicted else "BALL"
        cv2.putText(out, lbl_txt, _pt(bx1, by1 - 5),
                    FONT, 0.45, BALL_BOX_COLOR, 1, cv2.LINE_AA)

    # ── Top-left: frame counter + timestamp ──────────────────────────────────
    ts_now = frame_idx / fps
    cv2.putText(out, f"Frame {frame_idx:05d}   {_fmt_time(ts_now)}",
                _pt(10, 30), FONT, 0.65, TEXT_COLOR, 2, cv2.LINE_AA)

    # ── Top-right: activity panel ─────────────────────────────────────────────
    PAD   = 14
    ACCENT = 6   # width of the coloured left-side accent bar

    if label:
        tac_color = TACTIC_COLORS.get(label, DEFAULT_BOX_COLOR)
        disp_name = DISPLAY_NAMES.get(label, label)
        numeric   = LABEL_TO_IDX.get(label, 0)

        line1 = disp_name                               # e.g. "ATTACKING"
        line2 = f"Activity : {label}"                  # full tactic name
        line3 = f"Started  : {_fmt_time(activity_start_time)}"
        line4 = f"Conf     : {conf:.2f}"

        sc1, th1 = 0.80, 2
        sc2, th2 = 0.52, 1

        sizes = [
            cv2.getTextSize(line1, FONT, sc1, th1)[0],
            cv2.getTextSize(line2, FONT, sc2, th2)[0],
            cv2.getTextSize(line3, FONT, sc2, th2)[0],
            cv2.getTextSize(line4, FONT, sc2, th2)[0],
        ]
        panel_w = max(int(s[0]) for s in sizes) + PAD * 2 + ACCENT + 4
        panel_h = sum(int(s[1]) for s in sizes) + PAD * 5 + 4

        px1 = w - panel_w - 10
        py1 = 10
        px2 = px1 + panel_w
        py2 = py1 + panel_h

        # Semi-transparent dark background
        overlay = out.copy()
        cv2.rectangle(overlay, _pt(px1, py1), _pt(px2, py2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.80, out, 0.20, 0, out)

        # Coloured accent bar on left edge
        cv2.rectangle(out, _pt(px1, py1), _pt(px1 + ACCENT, py2), tac_color, -1)

        # Text lines
        tx = px1 + ACCENT + PAD
        ty = py1 + PAD + int(sizes[0][1])
        cv2.putText(out, line1, _pt(tx, ty), FONT, sc1, tac_color, th1, cv2.LINE_AA)
        ty += int(sizes[1][1]) + PAD
        cv2.putText(out, line2, _pt(tx, ty), FONT, sc2, TEXT_COLOR, th2, cv2.LINE_AA)
        ty += int(sizes[2][1]) + 6
        cv2.putText(out, line3, _pt(tx, ty), FONT, sc2, TEXT_COLOR, th2, cv2.LINE_AA)
        ty += int(sizes[3][1]) + 6
        cv2.putText(out, line4, _pt(tx, ty), FONT, sc2, TEXT_COLOR, th2, cv2.LINE_AA)

    else:
        # Waiting for enough frames to classify
        msg = "CLASSIFYING..."
        (mw, mh), _ = cv2.getTextSize(msg, FONT, 0.65, 1)
        px1 = w - int(mw) - 24
        py1 = 10
        overlay = out.copy()
        cv2.rectangle(overlay, _pt(px1 - 8, py1), _pt(w - 10, py1 + int(mh) + 16),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.putText(out, msg, _pt(px1, py1 + int(mh) + 8),
                    FONT, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Bottom: classification log panel (mirrors terminal output) ────────────
    if recent_preds:
        _draw_log_panel(out, recent_preds, h, w)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth CSV loader
# ──────────────────────────────────────────────────────────────────────────────

# Maps the CSV spelling (British + underscores) to internal label names
_CSV_TO_LABEL: dict[str, str] = {
    "Spacing_Breakdown":   "Spacing Breakdown",
    "Coordinated_Attack":  "Coordinated Attack",
    "Coordinated_Defence": "Coordinated Defense",
    "Coordinated_Defense": "Coordinated Defense",
    "Delayed_Support":     "Delayed Support",
}

def load_ground_truth(csv_path: str) -> list[tuple[float, str]]:
    """
    Parse the project annotation CSV (video_file  anchor_time  tactic_class).
    Returns a list of (seconds, label) sorted by time.
    """
    df = pd.read_csv(csv_path, sep=r"\s+", engine="python")
    ann: list[tuple[float, str]] = []
    for _, row in df.iterrows():
        t_str = str(row["anchor_time"]).strip()
        try:
            m, s = t_str.split(":")
            secs = float(m) * 60 + float(s)
        except Exception:
            continue
        label = _CSV_TO_LABEL.get(str(row["tactic_class"]).strip(), "")
        if label:
            ann.append((secs, label))
    ann.sort(key=lambda x: x[0])
    return ann


def gt_label_at(ann: list[tuple[float, str]], t: float, hold: float = 30.0) -> str:
    """
    Return the most recent annotation at or before time t, held for up to
    `hold` seconds.  Returns "" outside all annotation windows so dead-ball
    periods (celebrations, between-rally breaks) show no label.
    """
    label = ""
    for at, lbl in ann:
        if at <= t:
            if t - at <= hold:
                label = lbl
        else:
            break
    return label


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:

    # ── Dependency checks ─────────────────────────────────────────────────────
    if not HAS_YOLO:
        print("[ERROR] Install ultralytics:  pip install ultralytics>=8.3.0", file=sys.stderr)
        sys.exit(1)
    if not HAS_SV:
        print("[ERROR] Install supervision:  pip install supervision>=0.21.0", file=sys.stderr)
        sys.exit(1)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    # ── Load YOLOv11 ─────────────────────────────────────────────────────────
    print(f"Loading YOLOv11 : {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    # ── Initialise ByteTrack ─────────────────────────────────────────────────
    print("Initialising ByteTrack tracker")
    tracker = sv.ByteTrack()

    # ── Load Mamba checkpoint ─────────────────────────────────────────────────
    model = norm_mean = norm_std = None
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading Mamba checkpoint : {args.checkpoint}  (device: {device})")
        model, norm_mean, norm_std = load_mamba(args.checkpoint, device)
    else:
        print("No Mamba checkpoint – using rule-based fallback classification.")
        from label_clips import label_clip, LABEL_TO_INDEX as RULE_IDX

    # ── Ground-truth CSV (overrides Mamba predictions when supplied) ─────────
    gt_ann: list[tuple[float, str]] = []
    if getattr(args, "ground_truth_csv", None):
        gt_ann = load_ground_truth(args.ground_truth_csv)
        print(f"Ground-truth CSV loaded: {len(gt_ann)} annotations "
              f"from '{args.ground_truth_csv}'")

    # ── Homography ────────────────────────────────────────────────────────────
    H: np.ndarray | None = None
    if args.court_corners:
        H = build_homography(args.court_corners)
        print("Homography: court corners supplied ✓")
    else:
        print("Homography: no corners supplied — auto-detecting from video …")
        try:
            from auto_corners import auto_detect_court_corners
            auto_c = auto_detect_court_corners(args.video,
                                               debug_path="corners_debug.jpg")
            if auto_c:
                H = build_homography(auto_c)
                print("Homography: auto-detected court corners ✓  (see corners_debug.jpg)")
            else:
                print(f"[WARN] Auto-detection failed -> "
                      f"linear scaling to {COURT_WIDTH_CM}×{COURT_LENGTH_CM} cm.")
        except ImportError:
            print(f"Homography: linear scaling to {COURT_WIDTH_CM}×{COURT_LENGTH_CM} cm.")

    # ── Open video ────────────────────────────────────────────────────────────
    print(f"\nOpening: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {frame_w}×{frame_h}  {fps:.1f} FPS  ~{n_frames} frames")

    # ── Output video writers ──────────────────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    if args.output_video:
        writer = cv2.VideoWriter(args.output_video, FOURCC, fps, (frame_w, frame_h))
        if not writer.isOpened():
            print(f"[WARNING] Cannot create .mp4 '{args.output_video}'.", file=sys.stderr)
            writer = None
        else:
            print(f"  Output video (.mp4) : {args.output_video}")

    writer2: cv2.VideoWriter | None = None
    if getattr(args, "output_avi", None):
        fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
        writer2 = cv2.VideoWriter(args.output_avi, fourcc_avi, fps, (frame_w, frame_h))
        if not writer2.isOpened():
            print(f"[WARNING] Cannot create .avi '{args.output_avi}'.", file=sys.stderr)
            writer2 = None
        else:
            print(f"  Output video (.avi) : {args.output_avi}")

    # ── Auto-training data directory ──────────────────────────────────────────
    train_dir: str | None = getattr(args, "save_training_data", None) or None
    train_saved = 0
    if train_dir:
        os.makedirs(train_dir, exist_ok=True)
        print(f"  Training data dir   : {train_dir}  (auto-labeled while processing)")

    # ── State ─────────────────────────────────────────────────────────────────
    ring_buf: list[np.ndarray] = []     # rolling buffer of feature rows
    track_age: dict[int, int]  = {}
    ball_tracker  = BallTracker()
    ball_trail: deque = deque(maxlen=BALL_TRAIL_LEN)
    id_mapper        = PlayerIDMapper(N_PLAYERS)
    persistent_boxes: dict[int, tuple] = {}   # slot -> last known (x1,y1,x2,y2)
    active_slots:     set[int]         = set()

    current_label        = ""
    current_conf         = 0.0
    activity_start_time  = 0.0   # video timestamp (seconds) when current activity began
    seq_idx = frame_idx = 0
    frames_since_last_classify = 0
    predictions:   list[dict] = []
    recent_preds:  deque      = deque(maxlen=8)   # log panel — last 8 classifications

    print("\nProcessing frames …\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frames_since_last_classify += 1

        # ── YOLOv11 detection ─────────────────────────────────────────────────
        ball_conf_thr = getattr(args, "ball_conf_threshold", BALL_CONF_THRESHOLD)
        p_xyxy, p_conf, ball_px = detect_frame(
            frame, yolo, args.conf_threshold, ball_conf=ball_conf_thr)

        # ── Near-team filter (Team A only, back-center camera) ────────────────
        p_xyxy, p_conf = filter_near_team(p_xyxy, p_conf, frame_w, frame_h, H)

        # ── Cap to exactly N_PLAYERS by highest YOLO confidence ───────────────
        # Prevents coaches/referees near the back line from stealing a slot.
        if len(p_xyxy) > N_PLAYERS:
            top_idx = np.argsort(p_conf)[::-1][:N_PLAYERS]
            p_xyxy  = p_xyxy[top_idx]
            p_conf  = p_conf[top_idx]

        # ── ByteTrack update ──────────────────────────────────────────────────
        dets = sv.Detections(
            xyxy=p_xyxy,
            confidence=p_conf,
            class_id=np.zeros(len(p_xyxy), dtype=int),
        ) if len(p_xyxy) > 0 else sv.Detections.empty()
        tracks = tracker.update_with_detections(dets)

        # ── Map ByteTrack IDs -> sequential display slots P1–P6 ───────────────
        slot_boxes, active_slots = id_mapper.update(tracks)
        persistent_boxes.update(slot_boxes)   # keep last known bbox per slot

        # ── Ball tracker ──────────────────────────────────────────────────────
        ball_tracked, ball_pred = ball_tracker.update(ball_px)

        # ── Court positions + 14-feature vector ───────────────────────────────
        player_pos, ball_pos = extract_positions(
            tracks, ball_tracked, frame_w, frame_h, H, track_age)
        feat_row = build_feature_row(player_pos, ball_pos)

        # ── Sliding-window buffer ─────────────────────────────────────────────
        ring_buf.append(feat_row)
        if len(ring_buf) > SEQ_LEN:
            ring_buf.pop(0)

        # ── Ground-truth label override (uses CSV annotations when supplied) ────
        if gt_ann:
            t_now = frame_idx / fps
            gt_lbl = gt_label_at(gt_ann, t_now, hold=30.0)
            if gt_lbl != current_label:
                activity_start_time = frame_idx / fps
            current_label = gt_lbl
            current_conf  = 1.0

        # Classify only when ball is on near team's half (court_y >= 900 cm).
        # No activity label is shown when ball is on opponent side / undetected.
        ball_in_near_half = (ball_pos[1] >= NET_Y_CM) if ball_pos[1] > 0 else False

        # Classify every `stride` frames once buffer is full AND ball is on near side
        # (skipped when ground-truth CSV is supplied — GT takes priority)
        if (not gt_ann
                and len(ring_buf) == SEQ_LEN
                and frames_since_last_classify >= args.stride
                and ball_in_near_half):

            seq_arr = np.stack(ring_buf)   # (29, 14)

            if model is not None:
                label, numeric, conf, probs = classify_sequence(
                    seq_arr, model, norm_mean, norm_std, device)
                probs_dict = {f"p_{n}": round(float(probs[i]),4)
                              for i, n in enumerate(LABEL_NAMES)}
            else:
                # Rule-based fallback
                df_seq = pd.DataFrame(seq_arr, columns=FEATURE_COLS)
                df_seq.insert(0, "frame_id", range(1, SEQ_LEN + 1))
                label   = label_clip(df_seq)
                numeric = RULE_IDX.get(label, 0)
                conf    = 1.0
                probs_dict = {}

            if label != current_label:
                activity_start_time = frame_idx / fps
            current_label = label
            current_conf  = conf
            frames_since_last_classify = 0
            seq_idx += 1

            # ── Auto-save training CSV (rule-based label, skip Unclassified) ──
            if train_dir and label in set(LABEL_NAMES):
                df_tr = pd.DataFrame(seq_arr, columns=FEATURE_COLS)
                df_tr.insert(0, "frame_id", range(1, SEQ_LEN + 1))
                df_tr["target_label"] = label
                df_tr.to_csv(
                    os.path.join(train_dir, f"auto_{seq_idx:06d}.csv"),
                    index=False)
                train_saved += 1

            entry = {
                "sequence":    seq_idx,
                "frame_start": frame_idx - SEQ_LEN + 1,
                "frame_end":   frame_idx,
                "label":       label,
                "label_index": numeric,
                "confidence":  round(conf, 4),
                **probs_dict,
            }
            predictions.append(entry)
            recent_preds.appendleft({
                "seq":     seq_idx,
                "time":    (frame_idx - SEQ_LEN + 1) / fps,
                "label":   label,
                "conf":    round(conf, 4),
                "numeric": numeric,
            })

            print(f"  Seq {seq_idx:4d}  frames {frame_idx-SEQ_LEN+1:5d}–{frame_idx:5d}"
                  f"  ->  [{numeric}] {DISPLAY_NAMES.get(label,label):<20s}  conf={conf:.3f}")

        # ── Ball trail update ─────────────────────────────────────────────────
        if ball_tracked is not None:
            ball_trail.append((int(ball_tracked[0]), int(ball_tracked[1]), ball_pred))

        # ── Annotate frame ────────────────────────────────────────────────────
        if writer is not None or writer2 is not None:
            annotated = annotate_frame(
                frame, persistent_boxes, active_slots,
                ball_tracked, ball_pred,
                ball_trail, current_label, current_conf, frame_idx, fps,
                activity_start_time=activity_start_time,
                recent_preds=list(recent_preds))
            if writer is not None:
                writer.write(annotated)
            if writer2 is not None:
                writer2.write(annotated)

        if frame_idx % 300 == 0:
            pct = frame_idx / max(n_frames, 1) * 100
            print(f"  … {frame_idx}/{n_frames} ({pct:.0f}%)")

    # ── Wrap up ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"\nAnnotated video (.mp4) -> '{args.output_video}'")
    if writer2:
        writer2.release()
        print(f"Annotated video (.avi) -> '{args.output_avi}'")

    if predictions and args.output_csv:
        pd.DataFrame(predictions).to_csv(args.output_csv, index=False)
        print(f"Predictions CSV -> '{args.output_csv}'")

    print(f"\nTotal frames : {frame_idx}  |  Sequences classified : {seq_idx}")
    if train_dir:
        print(f"Training CSVs saved  : {train_saved}  ->  {train_dir}/")

    # Summary counts
    if predictions:
        from collections import Counter
        counts = Counter(r["label"] for r in predictions)
        total  = sum(counts.values())
        print("\nTactic distribution:")
        for lbl in LABEL_NAMES:
            n = counts.get(lbl, 0)
            print(f"  {lbl:<24s}: {n:4d}  ({n/total*100:5.1f}%)")


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
            parser.error("--court-corners: need 4 x,y pairs e.g. 42,18 1238,18 1238,702 42,702")
        setattr(namespace, self.dest, pairs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Volleyball tactical analysis pipeline (back-center camera).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video",          help="Input match video (.mp4).")
    p.add_argument("checkpoint",     nargs="?", default=None,
                   help="Trained Mamba checkpoint (.pt). Uses rule-based if omitted.")
    p.add_argument("--yolo-model",   default="yolo11n.pt",
                   help="YOLOv11 weights. Downloads automatically if not present.")
    p.add_argument("--conf-threshold",      type=float, default=CONF_THRESHOLD)
    p.add_argument("--ball-conf-threshold", type=float, default=BALL_CONF_THRESHOLD,
                   help="Separate (lower) YOLO confidence for ball detection.")
    p.add_argument("--court-corners",  nargs=4, metavar=("TL","TR","BR","BL"),
                   action=_CornersAction, default=None,
                   help="4 pixel corners of court boundary (TL TR BR BL).")
    p.add_argument("--stride",       type=int, default=15,
                   help="Frames between successive classifications (sliding window step).")
    p.add_argument("--save-training-data", default=None, metavar="DIR",
                   help="Auto-save per-sequence training CSVs to DIR while processing.")
    p.add_argument("--output-video", default="annotated.mp4")
    p.add_argument("--output-avi",   default="annotated.avi",
                   help="Second output in AVI/XVID format. Pass empty string to disable.")
    p.add_argument("--output-csv",   default="predictions.csv")
    p.add_argument("--ground-truth-csv", default=None, metavar="CSV",
                   help="Annotation CSV (video_file anchor_time tactic_class). "
                        "When supplied, overrides Mamba predictions with ground-truth labels.")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(_parse_args())
