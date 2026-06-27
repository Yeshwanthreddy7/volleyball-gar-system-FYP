"""
auto_corners.py — Automatic volleyball court corner detection from a video frame.

Detects the 4 court corners (TL, TR, BR, BL) using HSV color segmentation of
the wood-floor court surface.  Used by pipeline.py when --court-corners is not
provided — zero manual work required.

Outputs:
  corners_debug.jpg — visualisation of detected corners for verification.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


# ── corner ordering ───────────────────────────────────────────────────────────

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 corners ordered as [TL, TR, BR, BL]."""
    pts  = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.stack([
        pts[np.argmin(s)],    # TL  (smallest x+y)
        pts[np.argmin(diff)], # TR  (smallest y-x)
        pts[np.argmax(s)],    # BR  (largest  x+y)
        pts[np.argmax(diff)], # BL  (largest  y-x)
    ])


# ── floor segmentation ────────────────────────────────────────────────────────

def _segment_floor(frame: np.ndarray) -> np.ndarray:
    """
    Return a binary mask that is white where the court floor is and
    black everywhere else (walls, audience, players, equipment).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # ── Strategy A: warm wood floor (yellow-orange, common in volleyball halls) ──
    for lo, hi in [
        ((8,  12,  85), (38, 210, 250)),   # bright warm wood
        ((5,   8,  75), (42, 230, 255)),   # very bright / over-exposed
        ((10, 18, 100), (35, 180, 240)),   # normal indoor lighting
    ]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

    # ── Strategy B: light gray / parquet (low-saturation bright surface) ────────
    gray_mask = cv2.inRange(hsv,
                            np.array([0,   0, 140], np.uint8),
                            np.array([180, 50, 255], np.uint8))

    # Use whichever gives more coverage
    if int(np.sum(gray_mask)) > int(np.sum(mask)) * 1.4:
        mask = gray_mask
    else:
        mask = mask | (gray_mask & cv2.inRange(hsv,
                                               np.array([0,   0, 150], np.uint8),
                                               np.array([180, 30, 255], np.uint8)))

    # ── Remove top strip (scoreboard overlay / ceiling) ─────────────────────────
    mask[:int(h * 0.10), :] = 0

    # ── Morphological clean-up ───────────────────────────────────────────────────
    k_big  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    k_med  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_big)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_med)
    return mask


# ── quadrilateral fitting ─────────────────────────────────────────────────────

def _fit_quad(contour: np.ndarray) -> np.ndarray:
    """Approximate contour as a quadrilateral; fall back to min-area-rect."""
    peri = cv2.arcLength(contour, True)
    for eps in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04,
                0.05, 0.06, 0.07, 0.08, 0.10, 0.12]:
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    # Last resort
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).reshape(4, 2).astype(np.float32)


# ── off-screen corner extrapolation ──────────────────────────────────────────

def _extend_sidelines(corners: np.ndarray, frame_h: int) -> np.ndarray:
    """
    If the near end-line (BL, BR) is partially or fully off-screen because
    the camera is very close to Japan's end line, extrapolate the sidelines
    to the bottom of the frame (or 10 % below it).
    """
    tl, tr, br, bl = corners.astype(np.float64)
    target_y = float(frame_h) * 1.05   # 5 % below frame bottom

    def at_y(p1, p2, y):
        dy = p2[1] - p1[1]
        if abs(dy) < 1.0:
            return p2[0]
        return p1[0] + (y - p1[1]) / dy * (p2[0] - p1[0])

    if bl[1] < frame_h * 0.90:
        bl = np.array([at_y(tl, bl, target_y), target_y])
    if br[1] < frame_h * 0.90:
        br = np.array([at_y(tr, br, target_y), target_y])

    return np.stack([tl, tr, br, bl]).astype(np.float32)


# ── main detector ─────────────────────────────────────────────────────────────

def detect_court_corners(
    frame: np.ndarray,
    debug_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Detect 4 court corners from a single BGR frame.
    Returns (4, 2) float32 array ordered [TL, TR, BR, BL], or None on failure.
    """
    h, w = frame.shape[:2]
    mask = _segment_floor(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep only large regions (court must be > 8 % of frame area)
    min_area = w * h * 0.08
    large = [c for c in contours if cv2.contourArea(c) > min_area]
    if not large:
        return None

    court_contour = max(large, key=cv2.contourArea)
    raw_quad = _fit_quad(court_contour)
    ordered  = _order_corners(raw_quad)
    ordered  = _extend_sidelines(ordered, h)

    if debug_path:
        _save_debug(frame, mask, ordered, debug_path)

    return ordered


def _save_debug(frame, mask, corners, path):
    vis = frame.copy()
    h, w = vis.shape[:2]
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 255, 255), (0, 165, 255), (0, 0, 255), (255, 0, 0)]

    for (pt, lbl, col) in zip(corners, labels, colors):
        px = int(np.clip(pt[0], 0, w - 1))
        py = int(np.clip(pt[1], 0, h - 1))
        cv2.circle(vis, (px, py), 10, col, -1)
        cv2.putText(vis, lbl, (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

    clipped = np.array(
        [[int(np.clip(p[0], 0, w-1)), int(np.clip(p[1], 0, h-1))]
         for p in corners], dtype=np.int32)
    cv2.polylines(vis, [clipped], True, (0, 255, 0), 2)

    # Side-by-side: original with corners  |  floor mask
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([vis, mask_rgb])
    if combined.shape[1] > 1920:
        scale = 1920 / combined.shape[1]
        combined = cv2.resize(combined, None, fx=scale, fy=scale)
    cv2.imwrite(path, combined)


# ── video entry-point ─────────────────────────────────────────────────────────

def auto_detect_court_corners(
    video_path: str,
    debug_path: Optional[str] = "corners_debug.jpg",
) -> Optional[list]:
    """
    Open *video_path*, try several early frames, return the first successful
    corner detection as a list [[x,y], [x,y], [x,y], [x,y]] (TL,TR,BR,BL),
    or None if all attempts fail.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[AutoCorner] Cannot open: {video_path}")
        return None

    try_frames = [0, 10, 30, 60, 90, 120, 180, 300, 450, 600]
    for fi in try_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        dp = debug_path if fi == try_frames[0] else None
        corners = detect_court_corners(frame, debug_path=dp)
        if corners is not None:
            cap.release()
            result = corners.tolist()
            tl, tr, br, bl = result
            print(f"[AutoCorner] Detected on frame {fi}:")
            print(f"  TL = ({int(tl[0])}, {int(tl[1])})")
            print(f"  TR = ({int(tr[0])}, {int(tr[1])})")
            print(f"  BR = ({int(br[0])}, {int(br[1])})")
            print(f"  BL = ({int(bl[0])}, {int(bl[1])})")
            if debug_path:
                print(f"  Debug image → {debug_path}")
            return result

    cap.release()
    print("[AutoCorner] All frames failed — auto-detection unsuccessful.")
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "videoplayback (1).mp4"
    corners = auto_detect_court_corners(path, debug_path="corners_debug.jpg")
    if corners:
        fmt = " ".join(f"{int(p[0])},{int(p[1])}" for p in corners)
        print(f"\nRun pipeline with auto corners:")
        print(f'  python pipeline.py "{path}" --court-corners {fmt}')
        print(f"\nOR just run (pipeline auto-detects if no corners given):")
        print(f'  python pipeline.py "{path}"')
    else:
        print("\nAuto-detection failed. Use pick_corners.py to click corners manually.")
