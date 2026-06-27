"""
annotate_from_csv.py — Annotate a volleyball match video using a pre-labeled
CSV activity timeline (no Mamba model required).

Reads a tab- or comma-separated file with columns:
  video_file | anchor_time | tactic_class

Outputs:
  • Annotated video  — player bounding boxes coloured by current tactic,
                        ball bounding box + motion trail, activity panel,
                        classification log panel at the bottom.
  • Activity report CSV — per-event summary table.
  • Console summary    — tactic distribution and duration breakdown.

Colour legend (same as pipeline.py):
  Coordinated Attack  → RED    (0, 0, 255) BGR
  Coordinated Defense → BLUE   (255, 0, 0) BGR
  Delayed Support     → YELLOW (0, 255, 255) BGR
  Spacing Breakdown   → GREEN  (0, 255, 0) BGR

Usage:
  cd grp_56_FYP
  python annotate_from_csv.py \\
      --video "dataset/videoplayback (1).mp4" \\
      --csv "dataset/csv_video(1).txt" \\
      --output annotated_csv.mp4 \\
      [--yolo-model yolo11n.pt] [--conf 0.35] [--display-window 8] [--stride 2] \\
      [--max-players 12] [--track-all]

Key flags:
  --max-players N   Maximum number of players to assign persistent IDs (default 12).
                    IDs 1-N are assigned on first appearance and never change.
  --track-all       Disable near-team filtering so ALL detected players receive IDs,
                    including both teams, referees, and bench staff.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque, Counter

import cv2
import numpy as np
import pandas as pd


def _pt(x, y) -> np.ndarray:
    """Convert any coordinates to np.int32 array so OpenCV 4.13 + NumPy 2.x accepts them.

    Clamps to ±30000 before casting to int32 so a run-away kinematic prediction
    (ball tracker diverging off-screen) never causes OverflowError in np.array.
    OpenCV's C binding parses 1-D int32 arrays as points without the per-element
    type checking that breaks with NumPy 2.x np.intp values in Python tuples.
    """
    _CLAMP = 30000
    xi = max(-_CLAMP, min(_CLAMP, int(round(float(x)))))
    yi = max(-_CLAMP, min(_CLAMP, int(round(float(y)))))
    return np.array([xi, yi], dtype=np.int32)


# ── reuse shared visual constants + helpers from pipeline.py ──────────────────
from pipeline import (
    TACTIC_COLORS, DEFAULT_BOX_COLOR, BALL_BOX_COLOR, BALL_BOX_HALF,
    TEXT_COLOR, DISPLAY_NAMES, BALL_TRAIL_LEN, BALL_MAX_MISS,
    FONT, FOURCC, PERSON_CLS, BALL_CLS,
    BallTracker, PlayerIDMapper,
    detect_frame, filter_near_team, build_homography,
    _draw_rounded_rect, _fmt_time,
)
from mamba_model import (
    N_PLAYERS, COURT_WIDTH_CM, COURT_LENGTH_CM, NET_Y_CM, NEAR_TEAM_MIN_Y,
    LABEL_TO_IDX,
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
# Tactic name normalisation  (CSV uses underscores + British spelling)
# ──────────────────────────────────────────────────────────────────────────────

_TACTIC_ALIASES: dict[str, str] = {
    "coordinated_attack":  "Coordinated Attack",
    "coordinated_defense": "Coordinated Defense",
    "coordinated_defence": "Coordinated Defense",   # British spelling
    "delayed_support":     "Delayed Support",
    "spacing_breakdown":   "Spacing Breakdown",
}


def _normalize_tactic(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    return _TACTIC_ALIASES.get(key, raw.strip())


# ──────────────────────────────────────────────────────────────────────────────
# CSV loading
# ──────────────────────────────────────────────────────────────────────────────

def _parse_anchor_time(s: str) -> float:
    """Parse MM:SS.s  or  HH:MM:SS.s  into total seconds."""
    parts = s.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(s)


def load_events(csv_path: str) -> list[tuple[float, str]]:
    """
    Load the annotation CSV and return a list of (time_seconds, tactic_name)
    sorted by time.  Handles both tab- and comma-separated files.
    """
    with open(csv_path, encoding="utf-8") as fh:
        first = fh.readline()
    sep = "\t" if "\t" in first else ","
    df = pd.read_csv(csv_path, sep=sep)

    required = {"anchor_time", "tactic_class"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    events: list[tuple[float, str]] = []
    for _, row in df.iterrows():
        t      = _parse_anchor_time(str(row["anchor_time"]))
        tactic = _normalize_tactic(str(row["tactic_class"]))
        events.append((t, tactic))

    events.sort(key=lambda x: x[0])
    return events


# ──────────────────────────────────────────────────────────────────────────────
# Active-tactic lookup at a given video timestamp
# ──────────────────────────────────────────────────────────────────────────────

def get_active_tactic(
    events:  list[tuple[float, str]],
    t:       float,
    window:  float = 8.0,
    pre:     float = 0.5,
) -> tuple[str | None, float | None]:
    """
    Return (tactic, anchor_time) for the most recent CSV event that is:
      • Within `pre` seconds in the future of  t  (early banner)
      • OR  up to `window` seconds in the past of  t  (persistence)

    Returns (None, None) when no event falls inside that range.
    """
    active_tactic: str | None  = None
    active_time:   float | None = None

    for ev_t, ev_tac in events:
        if ev_t <= t + pre:                # event has fired (with early-start offset)
            if t - ev_t < window:          # still within display window
                active_tactic = ev_tac
                active_time   = ev_t
        else:
            break                          # events are sorted; no need to continue

    return active_tactic, active_time


# ──────────────────────────────────────────────────────────────────────────────
# Bottom log panel  (shows recent CSV events rather than model predictions)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_csv_log_panel(
    out:          np.ndarray,
    recent_evs:   list[dict],
    h:            int,
    w:            int,
) -> None:
    """
    Draw a bottom-of-frame panel listing the last ≤8 CSV events.
    Each row:  Event #  AnchorTime  [Idx] Tactic
    """
    LOG_SC, LOG_TH = 0.44, 1
    HDR_SC, HDR_TH = 0.48, 1
    PAD  = 8
    ACNT = 5

    (_, row_h), _ = cv2.getTextSize("Ag", FONT, LOG_SC, LOG_TH)
    (_, hdr_h), _ = cv2.getTextSize("Ag", FONT, HDR_SC, HDR_TH)
    row_h = int(row_h); hdr_h = int(hdr_h)

    n_rows  = min(len(recent_evs), 8)
    if n_rows == 0:
        return
    panel_h = hdr_h + n_rows * (row_h + 5) + PAD * 3 + 6

    px1 = 10
    py2 = int(h) - 10
    py1 = py2 - panel_h
    px2 = int(w) - 10

    overlay = out.copy()
    cv2.rectangle(overlay, _pt(px1, py1), _pt(px2, py2), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)

    hdr = "CSV EVENT LOG    #Evt   AnchorTime    [Idx] Tactic"
    cv2.putText(out, hdr, _pt(px1 + PAD + ACNT + 4, py1 + PAD + hdr_h),
                FONT, HDR_SC, (200, 200, 200), HDR_TH, cv2.LINE_AA)

    div_y = py1 + PAD + hdr_h + 5
    cv2.line(out, _pt(px1 + PAD, div_y), _pt(px2 - PAD, div_y), (70, 70, 70), 1)

    ty = div_y + row_h + 5
    for ev in recent_evs[:8]:
        lbl = ev["tactic"]
        col = TACTIC_COLORS.get(lbl, (160, 160, 160))
        num = LABEL_TO_IDX.get(lbl, 0)

        cv2.rectangle(out,
                      _pt(px1 + PAD, ty - row_h + 2),
                      _pt(px1 + PAD + ACNT, ty + 2),
                      col, -1)

        row_txt = (f"#{ev['idx']:<5d}  {_fmt_time(ev['anchor'])}   "
                   f"[{num}]  {lbl}")
        cv2.putText(out, row_txt, _pt(px1 + PAD + ACNT + 6, ty),
                    FONT, LOG_SC, col, LOG_TH, cv2.LINE_AA)
        ty += row_h + 5


# ──────────────────────────────────────────────────────────────────────────────
# Frame annotation
# ──────────────────────────────────────────────────────────────────────────────

def annotate_frame_csv(
    frame:           np.ndarray,
    display_boxes:   dict[int, tuple],
    active_slots:    set[int],
    ball_pixel:      np.ndarray | None,
    ball_predicted:  bool,
    ball_trail:      deque,
    label:           str | None,
    anchor_time:     float | None,
    frame_idx:       int,
    fps:             float,
    recent_evs:      list[dict],
    n_players:       int = 12,
    slot_last_seen:  dict | None = None,
    max_ghost:       int = 5,
) -> np.ndarray:
    """
    Draw on a single frame:
      • Ball motion trail
      • Player bounding boxes  (colour = current CSV tactic)
      • Player ID badge (P1–Pn) — ID assigned on first appearance, never changes
      • Ball bounding box  (pink, fixed)
      • Top-left  : frame counter + video timestamp
      • Top-right : activity panel (tactic, anchor time)
      • Bottom    : CSV event log panel
    """
    out = frame.copy()
    h, w = int(out.shape[0]), int(out.shape[1])   # force Python int — cv2 rejects np.intp

    box_color = TACTIC_COLORS.get(label, DEFAULT_BOX_COLOR) if label else DEFAULT_BOX_COLOR

    # ── Ball motion trail (drawn with numpy to avoid cv2.circle type issues) ──
    trail = list(ball_trail)
    n     = len(trail)
    for i, (tx, ty, pred) in enumerate(trail):
        try:
            x, y = int(round(float(tx))), int(round(float(ty)))
        except Exception:
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue
        alpha  = float(i + 1) / max(n, 1)
        r      = max(1, int(round(3.0 * alpha)))
        bright = int(110 * alpha) if pred else int(230 * alpha)
        x1 = max(0, x - r); x2 = min(w, x + r + 1)
        y1 = max(0, y - r); y2 = min(h, y + r + 1)
        out[y1:y2, x1:x2] = [bright, bright, bright]

    # ── Player bounding boxes — only draw if detected recently ───────────────
    for slot in range(1, n_players + 1):
        if slot not in display_boxes:
            continue
        # Skip stale boxes: if a slot hasn't been seen within max_ghost frames
        # it means the player left the frame; don't draw a ghost box.
        if slot_last_seen is not None:
            last = slot_last_seen.get(slot, 0)
            if frame_idx - last > max_ghost:
                continue
        x1, y1, x2, y2 = [int(v) for v in display_boxes[slot]]
        # Clamp to frame so partially off-screen detections still render cleanly
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        is_active = slot in active_slots
        if is_active:
            color, thick = box_color, 2
        else:
            # Brief-persistence ghost (≤ max_ghost frames): thin, dimmed
            color = tuple(min(255, max(0, c // 2 + 60)) for c in box_color)
            thick = 1
        _draw_rounded_rect(out, x1, y1, x2, y2, color, thickness=thick)

        badge_txt = f"P{slot}"
        (tw, th), _ = cv2.getTextSize(badge_txt, FONT, 0.50, 1)
        bx1 = x1 - 1;       by1 = y1 - int(th) - 8
        bx2 = x1 + int(tw) + 6; by2 = y1 - 1
        cv2.rectangle(out, _pt(bx1, by1), _pt(bx2, by2), color, -1)
        cv2.putText(out, badge_txt, _pt(x1 + 3, y1 - 5),
                    FONT, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Ball bounding box ─────────────────────────────────────────────────────
    if ball_pixel is not None:
        cx = max(-30000, min(30000, int(round(float(ball_pixel[0])))))
        cy = max(-30000, min(30000, int(round(float(ball_pixel[1])))))
        # Only render when the predicted position is within (or just outside) the frame;
        # if the kinematic model has diverged the coordinates will be thousands of pixels
        # off-screen and the box is meaningless.
        if -BALL_BOX_HALF <= cx <= w + BALL_BOX_HALF and -BALL_BOX_HALF <= cy <= h + BALL_BOX_HALF:
            bx1 = max(0, cx - BALL_BOX_HALF)
            by1 = max(0, cy - BALL_BOX_HALF)
            bx2 = min(w - 1, cx + BALL_BOX_HALF)
            by2 = min(h - 1, cy + BALL_BOX_HALF)
            ball_thick = 1 if ball_predicted else 2
            cv2.rectangle(out, _pt(bx1, by1), _pt(bx2, by2),
                          BALL_BOX_COLOR, ball_thick, cv2.LINE_AA)
            lbl_txt = "BALL~" if ball_predicted else "BALL"
            cv2.putText(out, lbl_txt, _pt(bx1, by1 - 5),
                        FONT, 0.45, BALL_BOX_COLOR, 1, cv2.LINE_AA)

    # ── Top-left: frame counter + video timestamp ─────────────────────────────
    ts_now = frame_idx / fps
    cv2.putText(out, f"Frame {frame_idx:05d}   {_fmt_time(ts_now)}",
                _pt(10, 30), FONT, 0.65, TEXT_COLOR, 2, cv2.LINE_AA)

    # ── Top-right: activity panel ─────────────────────────────────────────────
    PAD    = 14
    ACCENT = 6

    if label:
        tac_color = TACTIC_COLORS.get(label, DEFAULT_BOX_COLOR)
        disp_name = DISPLAY_NAMES.get(label, label.upper())
        numeric   = LABEL_TO_IDX.get(label, 0)

        line1 = disp_name
        line2 = f"Tactic   : {label}"
        line3 = f"Anchor @ : {_fmt_time(anchor_time) if anchor_time is not None else '--'}"
        line4 = f"Source   : CSV label [{numeric}]"

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

        overlay = out.copy()
        cv2.rectangle(overlay, _pt(px1, py1), _pt(px2, py2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.80, out, 0.20, 0, out)
        cv2.rectangle(out, _pt(px1, py1), _pt(px1 + ACCENT, py2), tac_color, -1)

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
        msg = "NO ACTIVITY"
        (mw, mh), _ = cv2.getTextSize(msg, FONT, 0.65, 1)
        px1 = w - int(mw) - 24
        py1 = 10
        overlay = out.copy()
        cv2.rectangle(overlay, _pt(px1 - 8, py1), _pt(w - 10, py1 + int(mh) + 16),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.putText(out, msg, _pt(px1, py1 + int(mh) + 8),
                    FONT, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Bottom: CSV event log panel ───────────────────────────────────────────
    if recent_evs:
        _draw_csv_log_panel(out, recent_evs, h, w)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if not HAS_YOLO:
        print("[ERROR] ultralytics not installed.  pip install ultralytics>=8.3.0")
        sys.exit(1)
    if not HAS_SV:
        print("[ERROR] supervision not installed.  pip install supervision>=0.21.0")
        sys.exit(1)

    # ── Load CSV events ───────────────────────────────────────────────────────
    print(f"Loading events from : {args.csv}")
    events = load_events(args.csv)
    print(f"  {len(events)} annotated events loaded")
    tactic_counts: Counter = Counter(tac for _, tac in events)
    for tac, cnt in sorted(tactic_counts.items(), key=lambda x: -x[1]):
        print(f"    {tac:<28s} : {cnt}")

    # ── Load YOLO ─────────────────────────────────────────────────────────────
    print(f"\nLoading YOLOv11 : {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    # ── ByteTrack ─────────────────────────────────────────────────────────────
    # lost_track_buffer=150 keeps track IDs alive for 5 s @ 30 fps so players
    # that are temporarily occluded come back with the same ID.
    tracker = sv.ByteTrack(lost_track_buffer=150)

    # ── Homography ────────────────────────────────────────────────────────────
    H: np.ndarray | None = None
    if args.court_corners:
        H = build_homography(args.court_corners)
        print("Homography: court corners supplied ✓")
    else:
        print("Homography: no corners — attempting auto-detection …")
        try:
            from auto_corners import auto_detect_court_corners
            auto_c = auto_detect_court_corners(args.video, debug_path="corners_debug.jpg")
            if auto_c:
                H = build_homography(auto_c)
                print("Homography: auto-detected ✓  (see corners_debug.jpg)")
            else:
                print(f"  Auto-detection failed → "
                      f"linear scaling to {COURT_WIDTH_CM}×{COURT_LENGTH_CM} cm.")
        except ImportError:
            print(f"  Linear scaling to {COURT_WIDTH_CM}×{COURT_LENGTH_CM} cm.")

    # ── Open video ────────────────────────────────────────────────────────────
    print(f"\nOpening : {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {frame_w}×{frame_h}  {fps:.1f} FPS  ~{n_frames} frames "
          f"({n_frames/fps/60:.1f} min)")

    # ── Validate auto-detected homography ─────────────────────────────────────
    # If no corners were manually provided, check whether the auto-detected
    # corners are degenerate (essentially = frame borders).  This happens when
    # the floor segmentation captures the whole image instead of just the court.
    # A degenerate H is worse than no H: it passes far-team players through the
    # near-team filter.  Reject it and fall back to the pixel-Y threshold.
    if H is not None and not args.court_corners:
        try:
            H_inv = np.linalg.inv(H)
            # Where does court-Y = 900 cm (the net) map to in pixel space?
            mid_x_court = float(COURT_WIDTH_CM) / 2.0
            net_px = cv2.perspectiveTransform(
                np.array([[[mid_x_court, float(NET_Y_CM)]]], dtype=np.float32), H_inv)
            net_pixel_y = float(net_px[0, 0, 1])
            if not (frame_h * 0.15 < net_pixel_y < frame_h * 0.75):
                print(f"  [WARN] Auto-homography places net at pixel y={net_pixel_y:.0f} "
                      f"(outside 15–75 % of frame height {frame_h}) → rejected.")
                H = None
            else:
                print(f"  Homography validated: net at pixel y≈{net_pixel_y:.0f}  "
                      f"({net_pixel_y/frame_h*100:.0f} % from top)")
        except Exception:
            H = None

    # ── Pixel-Y fallback when no valid homography ─────────────────────────────
    # For back-centre endline cameras the near team (between camera and net)
    # always occupies the LOWER portion of the frame.  Use --near-team-y to set
    # the fraction of frame height below which a player must stand to be kept.
    # Default 0.50 = keep players whose foot (bbox bottom) is below the midline.
    if H is None and not args.track_all:
        y_frac = args.near_team_y if args.near_team_y > 0.0 else 0.50
        print(f"  Pixel-Y team filter active: keeping foot y > {y_frac*100:.0f}% "
              f"of frame height ({int(y_frac * frame_h)} px)")

    # ── Output video writer ───────────────────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    if args.output:
        writer = cv2.VideoWriter(args.output, FOURCC, fps, (frame_w, frame_h))
        if writer.isOpened():
            print(f"  Output video : {args.output}")
        else:
            print(f"  [WARNING] Cannot create '{args.output}'.")
            writer = None

    # ── State ─────────────────────────────────────────────────────────────────
    max_players        = args.max_players
    ball_tracker       = BallTracker(max_miss=BALL_MAX_MISS)
    ball_trail: deque  = deque(maxlen=BALL_TRAIL_LEN)
    id_mapper          = PlayerIDMapper(max_players)
    persistent_boxes:  dict[int, tuple] = {}
    active_slots:      set[int]         = set()
    track_age:         dict[int, int]   = {}
    slot_last_seen:    dict[int, int]   = {}   # slot → frame_idx of last detection

    # Report accumulators
    report_rows:  list[dict]  = []
    recent_evs:   deque       = deque(maxlen=8)   # log panel
    seen_anchors: set[float]  = set()             # track which events we've logged

    # Activity duration counters (seconds per tactic)
    tac_duration: dict[str, float] = {t: 0.0 for _, t in events}

    prev_label:  str | None  = None
    frame_idx                = 0

    # Pre-compute which events we'll log for the panel (most recent at top)
    all_events_for_log = [
        {"idx": i + 1, "anchor": t, "tactic": tac}
        for i, (t, tac) in enumerate(events)
    ]

    print("\nProcessing frames …\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_now = frame_idx / fps

        # Process only every `stride` frames for YOLO (still write every frame)
        if (frame_idx - 1) % args.stride == 0:
            p_xyxy, p_conf, ball_px = detect_frame(
                frame, yolo, args.conf,
                ball_conf=getattr(args, "ball_conf", args.conf))

            # ── Team filter ───────────────────────────────────────────────────
            if not args.track_all and len(p_xyxy) > 0:
                if H is not None:
                    # Homography-based: exact court-coordinate filter
                    p_xyxy, p_conf = filter_near_team(
                        p_xyxy, p_conf, frame_w, frame_h, H)
                else:
                    # Pixel-Y fallback: near team has feet in lower frame
                    y_frac  = args.near_team_y if args.near_team_y > 0.0 else 0.50
                    y_thresh = frame_h * y_frac
                    mask = p_xyxy[:, 3] >= y_thresh   # y2 (foot) ≥ threshold
                    p_xyxy = p_xyxy[mask]
                    p_conf = p_conf[mask]

            dets = (sv.Detections(
                        xyxy=p_xyxy,
                        confidence=p_conf,
                        class_id=np.zeros(len(p_xyxy), dtype=int))
                    if len(p_xyxy) > 0
                    else sv.Detections.empty())
            tracks = tracker.update_with_detections(dets)

            if tracks.tracker_id is not None:
                for tid in tracks.tracker_id:
                    track_age[int(tid)] = track_age.get(int(tid), 0) + 1

            slot_boxes, active_slots = id_mapper.update(tracks)
            persistent_boxes.update(slot_boxes)
            # Record when each slot was last seen for ghost-box suppression
            for s in active_slots:
                slot_last_seen[s] = frame_idx

            ball_tracked, ball_pred = ball_tracker.update(ball_px)
        else:
            # Between YOLO strides: propagate ball with kinematic model
            ball_tracked, ball_pred = ball_tracker.update(None)

        if ball_tracked is not None:
            bx = float(ball_tracked[0].item() if hasattr(ball_tracked[0], 'item') else ball_tracked[0])
            by = float(ball_tracked[1].item() if hasattr(ball_tracked[1], 'item') else ball_tracked[1])
            if math.isfinite(bx) and math.isfinite(by):
                # Reset tracker if kinematic prediction has diverged way outside the frame
                # (e.g. a false-positive detection gave a huge velocity)
                if -frame_w <= bx <= 2 * frame_w and -frame_h <= by <= 2 * frame_h:
                    ball_trail.append((int(round(bx)), int(round(by)), bool(ball_pred)))
                else:
                    ball_tracked = None
                    ball_tracker = BallTracker(max_miss=BALL_MAX_MISS)
            else:
                ball_tracked = None
                ball_tracker = BallTracker(max_miss=BALL_MAX_MISS)

        # ── Determine active tactic from CSV ──────────────────────────────────
        label, anchor_time = get_active_tactic(events, t_now, args.display_window)

        # Accumulate tactic duration
        if label:
            tac_duration[label] = tac_duration.get(label, 0.0) + 1.0 / fps

        # Log event to panel when a new anchor first becomes active
        if anchor_time is not None and anchor_time not in seen_anchors:
            seen_anchors.add(anchor_time)
            ev_idx = next(
                (i + 1 for i, (t, _) in enumerate(events) if t == anchor_time),
                len(seen_anchors))
            recent_evs.appendleft({
                "idx":    ev_idx,
                "anchor": anchor_time,
                "tactic": label,
            })
            print(f"  Event {ev_idx:3d}  @{_fmt_time(anchor_time)}  "
                  f"→  {label}  (frame {frame_idx})")

        # Build report row
        report_rows.append({
            "frame":       frame_idx,
            "time_sec":    round(t_now, 3),
            "time_str":    _fmt_time(t_now),
            "tactic":      label or "",
            "anchor_time": round(anchor_time, 3) if anchor_time else "",
        })

        # ── Annotate and write frame ──────────────────────────────────────────
        if writer is not None:
            annotated = annotate_frame_csv(
                frame, persistent_boxes, active_slots,
                ball_tracked, ball_pred, ball_trail,
                label, anchor_time, frame_idx, fps,
                list(recent_evs),
                n_players=max_players,
                slot_last_seen=slot_last_seen,
                max_ghost=5)
            writer.write(annotated)

        if frame_idx % 500 == 0:
            pct = frame_idx / max(n_frames, 1) * 100
            print(f"  … {frame_idx}/{n_frames} ({pct:.0f}%)")

    # ── Wrap up ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"\nAnnotated video → '{args.output}'")

    # Activity report CSV
    report_path = os.path.splitext(args.output)[0] + "_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"Activity report  → '{report_path}'")

    # Console summary
    total_sec = frame_idx / fps
    print(f"\n{'='*60}")
    print(f"  Video summary")
    print(f"{'='*60}")
    print(f"  Total frames  : {frame_idx}")
    print(f"  Duration      : {_fmt_time(total_sec)}")
    print(f"  CSV events    : {len(events)}")
    print(f"\n  Tactic distribution (by event count):")
    for tac, cnt in sorted(tactic_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(events) * 100
        print(f"    {tac:<28s}  {cnt:3d} events  ({pct:5.1f}%)")

    print(f"\n  Tactic duration (seconds active):")
    total_active = sum(tac_duration.values())
    for tac in sorted(tac_duration, key=lambda x: -tac_duration[x]):
        sec = tac_duration[tac]
        pct = sec / total_sec * 100
        print(f"    {tac:<28s}  {sec:6.1f}s  ({pct:5.1f}% of video)")
    print(f"{'='*60}\n")


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
            parser.error("--court-corners: 4 x,y pairs  e.g.  42,18 1238,18 1238,702 42,702")
        setattr(namespace, self.dest, pairs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate a volleyball video from a CSV activity timeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",          required=True,
                   help="Input match video (.mp4).")
    p.add_argument("--csv",            required=True,
                   help="Tab/comma-separated annotation file "
                        "(video_file | anchor_time | tactic_class).")
    p.add_argument("--output",         default="annotated_csv.mp4",
                   help="Output annotated video path.")
    p.add_argument("--yolo-model",     default="yolo11n.pt",
                   help="YOLOv11 weights file.  Auto-downloaded if absent.")
    p.add_argument("--conf",           type=float, default=0.20,
                   help="YOLO person detection confidence threshold.")
    p.add_argument("--ball-conf",      type=float, default=0.08,
                   help="YOLO ball detection confidence threshold (lower than --conf "
                        "because the ball is small and often partially occluded).")
    p.add_argument("--stride",         type=int, default=1,
                   help="Run YOLO every N frames (1 = every frame, "
                        "higher = faster but less smooth boxes).")
    p.add_argument("--display-window", type=float, default=8.0,
                   help="Seconds to keep showing a tactic after its anchor time.")
    p.add_argument("--court-corners",  nargs=4,
                   metavar=("TL", "TR", "BR", "BL"),
                   action=_CornersAction, default=None,
                   help="4 pixel corners of court boundary (TL TR BR BL) for "
                        "near-team player filtering via homography.")
    p.add_argument("--near-team-y",    type=float, default=0.0,
                   help="Pixel-Y fraction threshold (0–1) for the near-team filter "
                        "when no homography is available.  Players whose bounding-box "
                        "foot (y2) is BELOW this fraction of frame height are kept; "
                        "the rest (far team, referees above the net) are dropped.  "
                        "For a back-centre endline camera, 0.40–0.50 works well.  "
                        "0.0 = disabled (default).")
    p.add_argument("--max-players",    type=int, default=6,
                   help="Maximum number of players to assign persistent IDs (1–N). "
                        "Default 6 = standard volleyball team.  Each new detection "
                        "gets the next free ID; once all N slots are filled, new "
                        "detections are matched by proximity to the nearest existing slot.")
    p.add_argument("--track-all",      action="store_true", default=False,
                   help="Disable near-team court filtering so ALL detected persons "
                        "(both teams, referees, coaches) receive persistent player IDs. "
                        "Ignored when no homography is available (already tracks all).")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
