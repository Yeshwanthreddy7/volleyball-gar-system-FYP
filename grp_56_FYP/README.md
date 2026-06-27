# Volleyball Tactical Video Analysis System
### Back-Center End-Line Camera | YOLOv11 + ByteTrack + Homography + Mamba SSM

---

## Court Coordinate System

```
Camera (back-center, behind Team A end line, elevated 6–10 m)
       │
  Y=1800 ──── Team A end line (near, camera side) ────
  Y=1200 ──── 3m attack line (Team A) ────────────────
  Y= 900 ──── NET ─────────────────────────────────────
  Y= 600 ──── 3m attack line (Team B) ────────────────
  Y=   0 ──── Team B end line (far) ──────────────────
               X=0         X=900
            (left side)  (right side)
```

- **X-axis**: 0–900 cm (9 m court width, left → right from camera perspective)  
- **Y-axis**: 0–1800 cm (18 m court length, far end → camera end)  
- **Net**: Y = 900 cm  
- **Team A (tracked)**: Y = 900–1800 cm (near team, camera is behind them)  
- **Team B (not tracked)**: Y = 0–900 cm

---

## 4 Tactic Classes

| Index | Label | Description |
|-------|-------|-------------|
| 1 | Coordinated Attack | ≥2 attackers moving toward net together with high sync |
| 2 | Coordinated Defense | Team shifting laterally as a unit, formation maintained |
| 3 | Delayed Support | Player arrives late to support position after ball contact |
| 4 | Spacing Breakdown | Formation gap (>600 cm) or collision (<50 cm) |

---

## File Overview

```
fyp/
├── mamba_model.py          ← Mamba SSM architecture + shared constants
├── label_clips.py          ← Rule-based auto-labeling engine
├── extract_clips.py        ← Cuts rally clips from full match videos
├── prepare_training_data.py← Converts clips → training CSVs (sliding window)
├── train_mamba.py          ← Training loop with augmentation, metrics
├── pipeline.py             ← Full inference pipeline (video → annotated video)
├── infer_mamba.py          ← Standalone inference (CSV or video)
├── requirements.txt        ← Python dependencies
└── labels_template.csv     ← Annotation template for your 12 match videos
```

---

## Installation

```bash
pip install -r requirements.txt
```

For YOLOv11, ultralytics will download weights automatically on first run.  
Alternatively download manually:
```bash
# YOLOv11 nano (fastest, use for testing)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# YOLOv11 small (better accuracy)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

---

## Step-by-Step Workflow

### Step 1 — Annotate Your 12 Match Videos

Copy `labels_template.csv` → `labels.csv`.  
Watch each video and fill in one row per rally you want labeled:

```csv
video_file,anchor_time,tactic_class
match1.mp4,00:02:14.5,coordinated_attack
match1.mp4,00:05:33.0,spacing_breakdown
match1.mp4,00:08:47.2,coordinated_defense
match2.mp4,00:01:22.0,delayed_support
```

`anchor_time` = the moment the ball is contacted (serve/spike/dig).  
Format: `HH:MM:SS.s` or `MM:SS.s`

---

### Step 2 — Extract Rally Clips

```bash
python extract_clips.py --labels labels.csv
```

Creates:
```
dataset/
  Coordinated_Attack/   ← clip_0001.mp4, clip_0002.mp4, …
  Coordinated_Defense/
  Delayed_Support/
  Spacing_Breakdown/
```

Each clip is **2 seconds** (60 frames), starting 0.3 s before the anchor time.

---

### Step 3 — Prepare Training CSVs

**Important**: Measure your court corners first (see below), then run:

```bash
python prepare_training_data.py dataset \
    --output-dir training_csv \
    --yolo-model yolo11n.pt \
    --court-corners 42,18 1238,18 1238,702 42,702 \
    --stride 5
```

- `--court-corners TL TR BR BL`: pixel coordinates of the 4 court boundary corners  
  as seen by your back-center camera (TL=far-left, TR=far-right, BR=near-right, BL=near-left).  
- `--stride 5`: sliding window step → generates multiple CSVs per clip  
- Without `--court-corners`, linear pixel scaling is used (less accurate)

Output: `training_csv/*.csv` — one CSV per 29-frame window, labeled.

---

### Step 4 — Train the Mamba Model

```bash
python train_mamba.py training_csv \
    --epochs 60 \
    --batch_size 32 \
    --lr 1e-3 \
    --checkpoint mamba_checkpoint.pt
```

Training output includes:
- Per-epoch: train loss, train acc, val loss, val acc, val macro-F1
- Final test set: accuracy, macro-F1, balanced accuracy
- **Per-class F1** for all 4 tactic classes
- **Formatted confusion matrix** (printed + saved as `mamba_checkpoint_cm.png`)
- `training_history.csv` with all per-epoch metrics

---

### Step 5 — Run Inference on a New Match Video

```bash
python pipeline.py new_match.mp4 mamba_checkpoint.pt \
    --yolo-model yolo11n.pt \
    --court-corners 42,18 1238,18 1238,702 42,702 \
    --stride 15 \
    --output-video annotated_match.mp4 \
    --output-csv predictions.csv
```

The annotated video shows:
- **Rounded bounding boxes** around each tracked Team A player (color = tactic)
- **Player ID** badge above each box (P1, P2, …)
- **Ball marker**: white ring + orange fill (detected) or grey ring (predicted)
- **Ball trail**: fading orange dots showing recent ball path
- **Net reference line** at the approximate net position
- **Tactic banner** at the bottom:
  - Large text: ATTACKING / DEFENDING / DELAYED SUPPORT / SPACING BREAKDOWN
  - Small text: full label name + confidence score
- **Frame counter** and timestamp (top left)

---

### Step 6 — Test on a Single CSV (quick model check)

```bash
python infer_mamba.py --csv training_csv/some_clip.csv \
    --checkpoint mamba_checkpoint.pt
```

Prints the predicted tactic and per-class probability bar chart.

---

## How to Find Court Corners (Homography Setup)

1. Open your match video in any video player.
2. Pause at a clear frame where all 4 court boundary corners are visible.
3. Note the **pixel (x, y)** coordinates of:

```
TL = Top-Left corner     = Team B's far-left corner (far end of court)
TR = Top-Right corner    = Team B's far-right corner
BR = Bottom-Right corner = Team A's near-right corner (camera side)
BL = Bottom-Left corner  = Team A's near-left corner
```

Use a tool like GIMP, Paint, or VLC (Tools → Media Info shows pixel position) to read pixel coords.

Example: `--court-corners 42,18 1238,18 1238,702 42,702`

---

## Data Augmentation (applied during training)

| Augmentation | Description |
|---|---|
| Horizontal flip (50%) | Mirror X coordinates of all players/ball; simulates attack from opposite wing |
| Time jitter (±3 frames) | Shift window start randomly; varied temporal viewpoints |
| Gaussian noise (σ=0.01) | Small random perturbation on normalised features |

---

## Model Architecture

```
Input: (batch, 29, 14)
           ↓
  Linear(14 → d_model=64)
           ↓
  MambaBlock × 4  [SelectiveSSM + pre-norm residual]
           ↓
  LayerNorm
           ↓
  GlobalAveragePool (over 29 frames) → (batch, 64)
           ↓
  Dropout → Linear(64→32) → GELU → Dropout → Linear(32→4)
           ↓
Output: logits (batch, 4)
```

Trainable parameters: ~45,000

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No players detected | Lower `--conf-threshold` to 0.25 |
| Wrong team tracked | Check `--court-corners` pixel values (TL/TR/BR/BL order) |
| All classified as one tactic | Check class distribution in training_csv; collect more samples for minority classes |
| YOLOv11 not found | Run: `pip install ultralytics --upgrade` |
| ByteTrack IDs jumping | Normal for occlusion; ByteTrack handles reappearance automatically |
| Ball never detected | YOLOv11n uses COCO class 32 (sports ball); consider fine-tuning on volleyball images |
