# Volleyball Tactical Analysis System
### Final Year Project — Group 56
**YOLOv11 · ByteTrack · Homography · Mamba SSM**

An end-to-end computer vision pipeline that automatically detects, tracks, and classifies volleyball tactics from match footage filmed from a back-center end-line camera.

---

## What It Does

Takes a raw volleyball match video + a tactic timeline CSV and produces a fully annotated output video showing:

| Annotation | Detail |
|---|---|
| Player bounding boxes | Near-team only (P1–P6), stable IDs that never switch |
| Ball bounding box | Pink box, kinematic prediction between missed frames |
| Ball motion trail | Last 40 positions, fading from bright to dim |
| Tactic banner (top-right) | Colour-coded label from CSV, visible for 8 s after anchor |
| CSV event log (bottom) | Running history of all triggered tactic events |
| Frame counter (top-left) | Frame number + MM:SS.ss timestamp |

**4 Tactic Classes:**

| Class | Box Colour | Description |
|---|---|---|
| Coordinated Attack | RED | Organised offensive pattern |
| Coordinated Defense | BLUE | Structured defensive formation |
| Delayed Support | YELLOW | Delayed back-court support movement |
| Spacing Breakdown | GREEN | Breakdown of positional spacing |

---

## Project Structure

```
grp_56_FYP/
├── annotate_from_csv.py       ← MAIN SCRIPT — annotate video from a pre-labeled CSV
├── pipeline.py                ← Core library (YOLO, ByteTrack, BallTracker, drawing)
├── mamba_model.py             ← Mamba SSM classifier + shared constants
├── train_mamba.py             ← Train the Mamba model on feature CSVs
├── infer_mamba.py             ← Run a trained Mamba checkpoint on a video or CSV
├── prepare_training_data.py   ← Extract 14-feature CSVs from labeled clips
├── extract_clips.py           ← Cut rally clips from full match videos
├── pick_corners.py            ← Interactive tool to click court corners
├── auto_corners.py            ← Automatic court corner detection (HSV)
├── label_clips.py             ← Rule-based auto-labeler for extracted clips
├── labels_template.csv        ← Template for manual labeling
├── requirements.txt           ← Python dependencies
└── dataset/
    ├── videoplayback (1).mp4  ← Input match video (1280×720, 30 fps, 39 min)
    └── csv_video(1).txt       ← Pre-labeled tactic timeline (126 events)
```

---

## Quick Start — Generate Annotated Video

> The virtual environment (`venv/`) is already set up. Just run these two commands.

### Step 1 — Navigate into the project folder

```powershell
cd "C:\Users\yashwanth\Downloads\Yeshwanth_FYP\grp_56_FYP"
```

### Step 2 — Run the annotation pipeline

```powershell
.\venv\Scripts\python.exe -u annotate_from_csv.py --video "dataset/videoplayback (1).mp4" --csv "dataset/csv_video(1).txt" --output annotated_csv.mp4 --conf 0.20 --ball-conf 0.08 --stride 1 --max-players 6
```

**Processing time:** ~1.5 hours for a 39-minute video on CPU.  
Progress is printed every 500 frames so you can see it running.

### Step 3 — Open the output

```powershell
Start-Process "C:\Users\yashwanth\Downloads\Yeshwanth_FYP\grp_56_FYP\annotated_csv.mp4"
```

**Output files created:**
```
annotated_csv.mp4          ← fully annotated video
annotated_csv_report.csv   ← per-frame tactic label table
```

---

## Installation (First Time Only)

```powershell
cd "C:\Users\yashwanth\Downloads\Yeshwanth_FYP\grp_56_FYP"
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

YOLOv11 weights (`yolo11n.pt`) are downloaded automatically on the first run.

---

## All Flags — `annotate_from_csv.py`

| Flag | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to input match video |
| `--csv` | *(required)* | Path to tactic timeline CSV |
| `--output` | `annotated_csv.mp4` | Output video filename |
| `--conf` | `0.20` | YOLO confidence threshold for person detection |
| `--ball-conf` | `0.08` | YOLO confidence for ball detection (lower = more sensitive) |
| `--stride` | `1` | Run YOLO every N frames (1 = every frame, 2 = 2× faster) |
| `--max-players` | `6` | Max player slots (6 = standard volleyball team) |
| `--display-window` | `8.0` | Seconds to keep tactic banner on screen after anchor time |
| `--near-team-y` | `0.0` | Pixel-Y fraction fallback when homography fails (0 = disabled) |
| `--track-all` | off | If set, track both teams, referees, and bench staff |
| `--court-corners` | auto | 4 pixel corners TL TR BR BL for precise team filtering |
| `--yolo-model` | `yolo11n.pt` | Path to YOLO weights |

---

## System Architecture

```
Match Video (.mp4)
      │
      ▼
YOLOv11n Detection
  ├── Persons  (class 0)  — conf ≥ 0.20
  └── Ball     (class 32) — conf ≥ 0.08
      │
      ▼
Near-Team Filter (homography)
  └── Keeps players with court Y ≥ 900 cm (net line) or body centre ≥ 700 cm
      (jumping players included via centre-of-body check)
      │
      ▼
ByteTrack  (lost_track_buffer = 150 frames)
  └── Persistent tracker IDs across occlusions
      │
      ▼
PlayerIDMapper
  └── Maps ByteTrack IDs → stable display slots P1–P6 (never reassigned)
      │
      ▼
BallTracker  (kinematic bridge)
  └── Constant-velocity prediction for up to 30 missed frames (~1 second)
      Auto-resets if predicted position diverges off-screen
      │
      ▼
Homography  (cv2.findHomography)
  └── Pixel coordinates → real-world court centimetres
      Auto-detected from frame 0 if no manual corners given
      │
      ▼
14-Feature Vector per frame
  └── [ball_x, ball_y, p1_x, p1_y, p2_x, p2_y, … p6_x, p6_y]  (court cm)
      │
      ▼
Mamba SSM Classifier  (seq_len = 29, input_dim = 14)
  └── 4-class tactic output: Attack / Defense / Delayed / Spacing
      │
      ▼
Frame Annotation
  └── Rounded player boxes + badge · Ball box + trail · Tactic panel · Event log
      │
      ▼
annotated_csv.mp4  +  annotated_csv_report.csv
```

---

## Court Coordinate System

```
Camera (back-center, elevated 6–10 m behind Team A's end line)

Y=1800 ─── Team A end line  (near — camera side)
             │                │
             │   Team A       │  ← tracked (P1–P6)
             │   Y: 900–1800  │
Y= 900 ──── NET ─────────────────
             │   Team B       │  ← filtered out
             │   Y: 0–900     │
Y=   0 ─── Team B end line   (far)

X=0 (left sideline) ────────────── X=900 cm (right sideline)
```

---

## Training the Mamba Model (Optional)

Follow these steps only if you want to train the model on new match data.

### 1 — Create your labels file

Use `labels_template.csv` as a starting point. Fill one row per rally:

```
video_file,anchor_time,tactic_class
match1.mp4,00:02:14.5,coordinated_attack
match1.mp4,00:05:33.0,spacing_breakdown
```

`anchor_time` = moment of ball contact (serve / spike / dig).

### 2 — Extract rally clips

```powershell
.\venv\Scripts\python.exe extract_clips.py --labels labels.csv
```

Creates 2-second clips under `dataset/Coordinated_Attack/`, `dataset/Spacing_Breakdown/`, etc.

### 3 — Prepare training feature CSVs

```powershell
.\venv\Scripts\python.exe prepare_training_data.py dataset --output-dir training_csv --court-corners 42,18 1238,18 1238,702 42,702
```

### 4 — Train

```powershell
.\venv\Scripts\python.exe train_mamba.py training_csv --epochs 60 --batch_size 32 --lr 1e-3 --checkpoint mamba_checkpoint.pt
```

Outputs: checkpoint `.pt`, confusion matrix `.png`, `training_history.csv`.

### 5 — Infer with the trained model

```powershell
.\venv\Scripts\python.exe pipeline.py "dataset/videoplayback (1).mp4" mamba_checkpoint.pt --court-corners 42,18 1238,18 1238,702 42,702 --output-video annotated.mp4 --output-csv predictions.csv
```

---

## How to Set Precise Court Corners

Better court corners = better team filtering (fewer opponent players leaking through).

```powershell
.\venv\Scripts\python.exe pick_corners.py "dataset/videoplayback (1).mp4"
```

Click the 4 corners in this order on the video frame:
1. **Far-Left** — far end of court, left sideline
2. **Far-Right** — far end of court, right sideline
3. **Near-Right** — near end (camera side), right sideline
4. **Near-Left** — near end (camera side), left sideline

Copy the printed pixel coordinates into `--court-corners` when running annotation.

---

## Mamba Model Architecture

```
Input:  (batch, seq_len=29, input_dim=14)
             ↓
    Linear(14 → d_model=64)
             ↓
    MambaBlock × 4   [SelectiveSSM + pre-norm residual]
             ↓
    LayerNorm
             ↓
    Global Average Pool  →  (batch, 64)
             ↓
    Dropout → Linear(64→32) → GELU → Dropout → Linear(32→4)
             ↓
Output: logits (batch, 4)   ~45,000 trainable parameters
```

Reference: Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752

---

## Results

Full annotation run on `videoplayback (1).mp4` (39 min, 70,591 frames):

| Metric | Value |
|---|---|
| Total frames annotated | 70,591 / 70,591 (100%) |
| CSV events detected | 126 / 126 |
| Coordinated Attack | 40 events (31.7%) |
| Coordinated Defense | 40 events (31.7%) |
| Spacing Breakdown | 39 events (31.0%) |
| Delayed Support | 7 events (5.6%) |
| Processing time (CPU, stride=1) | ~1.5 hours |

---

## Requirements

| Package | Minimum Version |
|---|---|
| Python | 3.10+ |
| ultralytics | 8.3.0 |
| supervision | 0.21.0 |
| torch | 2.0.0 |
| torchvision | 0.15.0 |
| opencv-python | 4.7.0 |
| numpy | 1.24.0 |
| pandas | 2.0.0 |
| scikit-learn | 1.3.0 |
| matplotlib | 3.7.0 |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `.\venv\Scripts\python.exe not recognized` | Run `cd grp_56_FYP` first — you must be inside this folder |
| Ball not appearing in output video | Lower `--ball-conf` to `0.05` |
| Players missing boxes (jumping, blurred) | Lower `--conf` to `0.15` |
| Opponent team boxes showing | Run `pick_corners.py` and pass precise corners via `--court-corners` |
| Video takes too long to process | Use `--stride 2` to halve processing time |
| Ghost boxes in empty areas | Already fixed — boxes disappear after 5 frames of no detection |
| Out of memory | Use `yolo11n.pt` (nano) instead of larger variants |

---

## Authors

Group 56 — Final Year Project
