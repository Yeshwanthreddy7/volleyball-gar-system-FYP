"""
pick_corners.py — Click the 4 court corners on the first video frame.

Usage:
  python pick_corners.py "videoplayback (1).mp4"

Click in this order:
  1. Far-Left  (top-left corner of court in video — far end, left sideline)
  2. Far-Right (top-right corner of court — far end, right sideline)
  3. Near-Right (bottom-right corner — near end closest to camera, right)
  4. Near-Left  (bottom-left corner — near end closest to camera, left)

Press R to reset and re-click. Press Q or close window to quit early.
"""

import sys
import cv2
import numpy as np

LABELS = [
    "1. Far-Left  (TL)",
    "2. Far-Right (TR)",
    "3. Near-Right (BR)",
    "4. Near-Left  (BL)",
]
COLORS = [
    (0, 255, 255),   # yellow
    (0, 165, 255),   # orange
    (0, 0, 255),     # red
    (255, 0, 0),     # blue
]


def pick(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)
    ret, orig = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)

    corners = []
    frame = orig.copy()

    def draw_state():
        nonlocal frame
        frame = orig.copy()
        h, w = frame.shape[:2]
        # instructions
        instructions = [
            "Click the 4 court corners in order:",
            "  1 Far-Left (TL)   2 Far-Right (TR)",
            "  3 Near-Right (BR) 4 Near-Left (BL)",
            "R = reset   Q = quit",
        ]
        for i, txt in enumerate(instructions):
            cv2.putText(frame, txt, (10, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # draw already-clicked corners
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(frame, (cx, cy), 7, COLORS[i], -1)
            cv2.putText(frame, LABELS[i], (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[i], 2, cv2.LINE_AA)
        # draw polygon if >= 2 points
        if len(corners) >= 2:
            pts = np.array(corners, np.int32)
            cv2.polylines(frame, [pts], isClosed=(len(corners)==4), color=(0,255,0), thickness=2)
        cv2.imshow("Court Corner Picker", frame)

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 4:
                corners.append((x, y))
                draw_state()
                if len(corners) == 4:
                    _print_result()

    def _print_result():
        tl, tr, br, bl = corners
        cmd = (f"--court-corners "
               f"{tl[0]},{tl[1]} {tr[0]},{tr[1]} "
               f"{br[0]},{br[1]} {bl[0]},{bl[1]}")
        print("\n" + "="*60)
        print("  Court corners collected!")
        print("="*60)
        print(f"  Far-Left  (TL) : {tl}")
        print(f"  Far-Right (TR) : {tr}")
        print(f"  Near-Right(BR) : {br}")
        print(f"  Near-Left (BL) : {bl}")
        print("\n  Copy this into your pipeline command:")
        print(f"  {cmd}")
        print("\n  Full example command:")
        print(f'  python pipeline.py "videoplayback (1).mp4" {cmd}')
        print("="*60)

    cv2.namedWindow("Court Corner Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Court Corner Picker", 1280, 720)
    cv2.setMouseCallback("Court Corner Picker", on_click)
    draw_state()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            corners.clear()
            draw_state()
            print("[RESET] Click the corners again.")

    cv2.destroyAllWindows()

    if len(corners) == 4:
        _print_result()
    else:
        print(f"[INFO] Only {len(corners)}/4 corners selected. Run again.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pick_corners.py <video_file>")
        sys.exit(1)
    pick(sys.argv[1])
