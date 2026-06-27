"""
label_clips.py – Rule-Based Auto-Labeling Engine
              for Back-Center End-Line Camera View.

Court coordinate system (after homography):
  X : 0–900 cm   (9m width, left → right from camera perspective)
  Y : 0–1800 cm  (18m length, far end Y=0 → camera end Y=1800)
  Net at Y = 900 cm
  Team A (near, tracked): Y = 900–1800 cm
  Team B (far, not tracked): Y = 0–900 cm (not used here)

Each CSV has 41 frames from a labeled rally clip:
  - Evaluation window : frames 30–41  (used to determine the label)
  - Training window   : frames  1–29  (saved with target_label column)

4 Output Labels:
  1 → Coordinated Attack
  2 → Coordinated Defense
  3 → Delayed Support
  4 → Spacing Breakdown

Usage:
  python label_clips.py <directory_of_41frame_csvs>
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

from mamba_model import (
    COURT_WIDTH_CM, NET_Y_CM, COURT_LENGTH_CM,
    N_PLAYERS, SEQ_LEN, LABEL_TO_IDX,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

EVAL_START = 30    # inclusive  (frames used to DERIVE the label)
EVAL_END   = 41    # inclusive
TRAIN_END  = SEQ_LEN  # frames 1–29 saved for model training
FPS        = 30    # video frame rate

# Player column pairs
PLAYER_COLS = [
    ("p1_x", "p1_y"), ("p2_x", "p2_y"), ("p3_x", "p3_y"),
    ("p4_x", "p4_y"), ("p5_x", "p5_y"), ("p6_x", "p6_y"),
]

# Near-team region: Team A occupies Y ≥ NET_Y_CM
NEAR_TEAM_MIN_Y = NET_Y_CM          # 900 cm
NEAR_TEAM_MAX_Y = COURT_LENGTH_CM   # 1800 cm

# For coordinated attack: ball near the net (within 200 cm of net from near side)
BALL_ATTACK_ZONE_MAX_Y = NET_Y_CM + 200   # 1100 cm

# 1-based label mapping for CSV / report output
LABEL_INDEX = {
    1: "Coordinated Attack",
    2: "Coordinated Defense",
    3: "Delayed Support",
    4: "Spacing Breakdown",
}
LABEL_TO_INDEX = {v: k for k, v in LABEL_INDEX.items()}

# ANSI colours for terminal output
_ANSI = {
    "Coordinated Attack":  "\033[38;5;208m",
    "Coordinated Defense": "\033[38;5;12m",
    "Delayed Support":     "\033[38;5;14m",
    "Spacing Breakdown":   "\033[38;5;9m",
    "reset":               "\033[0m",
}


def _c(label: str, text: str, use_color: bool = True) -> str:
    if not use_color:
        return text
    return f"{_ANSI.get(label, '')}{text}{_ANSI['reset']}"


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _player_positions(df: pd.DataFrame) -> np.ndarray:
    """Return (n_frames, N_PLAYERS, 2) array of player positions."""
    return np.stack(
        [df[[xc, yc]].to_numpy(dtype=float) for xc, yc in PLAYER_COLS],
        axis=1,
    )


def _occupied(pos_frame: np.ndarray) -> np.ndarray:
    """Return only rows representing real Team A players (court Y ≥ 900 cm).
    Correctly handles both zero-padded and centroid-filled absent slots."""
    return pos_frame[pos_frame[:, 1] >= NEAR_TEAM_MIN_Y]


def _nn_distances(pos_frame: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distances (cm) for occupied players in one frame.
    Deduplicates near-identical positions caused by centroid-fill for absent slots."""
    occ = _occupied(pos_frame)
    if len(occ) < 2:
        return np.array([])
    # Remove positions within 5 cm of each other (centroid-filled absent slots
    # all land at the exact same point, creating false 0-cm "collision" signals).
    unique: list[np.ndarray] = [occ[0]]
    for pt in occ[1:]:
        if all(np.linalg.norm(pt - u) > 5.0 for u in unique):
            unique.append(pt)
    occ = np.array(unique)
    if len(occ) < 2:
        return np.array([])
    diff = occ[:, np.newaxis, :] - occ[np.newaxis, :, :]   # (k,k,2)
    d    = np.linalg.norm(diff, axis=-1)                    # (k,k)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=-1)                                   # (k,)


def compute_sync_score(positions: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity of player velocity vectors.
    Returns a float in [-1, 1]; 1 = all moving in same direction.
    """
    if len(positions) < 2:
        return 0.0
    velocities = np.diff(positions, axis=0)   # (L-1, N, 2)
    scores: list[float] = []
    for vel_frame in velocities:
        norms = np.linalg.norm(vel_frame, axis=1)   # (N,)
        active = norms >= 1e-6
        if active.sum() < 2:
            continue
        unit = vel_frame[active] / norms[active, np.newaxis]
        sim  = unit @ unit.T
        k    = active.sum()
        ri, ci = np.triu_indices(k, k=1)
        scores.extend(sim[ri, ci].tolist())
    return float(np.mean(scores)) if scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Rule implementations (evaluated on frames 30–41)
# ──────────────────────────────────────────────────────────────────────────────

def rule_coordinated_attack(eval_df: pd.DataFrame) -> bool:
    """
    Coordinated Attack – offensive structure.

    Triggers when ALL conditions hold:
      1. Ball detected AND near the net (Y in [NET_Y_CM, NET_Y_CM+200]).
      2. Top-2 fastest players speed > 300 cm/s  (attackers sprinting).
      3. At least 1 of those top-2 players is moving TOWARD the net
         (mean Y-velocity < -5 cm/frame — Y decreases as you approach net).
      4. Bottom-4 players average speed < 200 cm/s  (setter/support holding).
      5. Sync score > 0.4  (players moving in the same general direction).

    Court Y-axis: Y=1800 is camera end (near), Y=900 is net.
    Approach run = player moving from high Y (back court) toward low Y (net).
    """
    ball_y = eval_df["ball_y"].values.astype(float)
    ball_x = eval_df["ball_x"].values.astype(float)

    if np.all(ball_x == 0) and np.all(ball_y == 0):
        return False

    ball_near_net = (ball_y <= BALL_ATTACK_ZONE_MAX_Y) & (ball_y >= NEAR_TEAM_MIN_Y)
    if not ball_near_net.any():
        return False

    positions  = _player_positions(eval_df)           # (n, 6, 2)
    velocities = np.diff(positions, axis=0)           # (n-1, 6, 2)
    speeds_cmf = np.linalg.norm(velocities, axis=-1).mean(axis=0)  # (6,) cm/frame
    speeds_cms = speeds_cmf * FPS                     # cm/s

    mean_vy  = velocities[:, :, 1].mean(axis=0)      # (6,) mean Y-vel per player
    top2_idx = np.argsort(speeds_cms)[-2:]

    # At least 1 of the 2 fastest must be moving toward the net (Y decreasing)
    if (mean_vy[top2_idx] < -5.0).sum() < 1:
        return False

    part    = np.partition(speeds_cms, -2)
    top2    = part[-2:].mean()
    bottom4 = part[:-2].mean()

    sync = compute_sync_score(positions)

    return top2 > 300 and bottom4 < 200 and sync > 0.4


def rule_coordinated_defense(eval_df: pd.DataFrame) -> bool:
    """
    Coordinated Defense – defensive structure.

    Triggers when ALL conditions hold:
      1. Ball detected AND moving toward the near team's baseline
         (ball_y trend is positive: ball travels from Y=900 net toward Y=1800 camera)
         OR ball is already deep in near half (mean ball_y > NET_Y_CM + 300).
         Rationale: opponent has attacked; ball is flying into near team's court.
      2. Players are SPREAD in a defensive formation:
         mean std of their Y positions > 150 cm across frames.
         (front-row blockers near Y=900, back-row diggers near Y=1500–1800)
      3. Mean player speed < 150 cm/s  (players are SET in position, not charging).
      4. Formation shape is STABLE: spatial-variance std < 35% of mean.
         (defensive W-shape is held, not collapsing or spreading rapidly)
    """
    ball_x = eval_df["ball_x"].values.astype(float)
    ball_y = eval_df["ball_y"].values.astype(float)

    # Ball must be detected in the window
    detected = (ball_x != 0) | (ball_y != 0)
    if detected.sum() < 3:
        return False

    # Check ball is incoming: Y increasing (ball flying toward near baseline)
    # Use only frames where ball is actually seen
    ball_y_det = ball_y[detected]
    if len(ball_y_det) >= 2:
        ball_trend = float(np.polyfit(np.arange(len(ball_y_det)), ball_y_det, 1)[0])
    else:
        ball_trend = 0.0

    ball_deep = ball_y_det.mean() > NET_Y_CM + 300   # ball past mid-court on near side
    if ball_trend < 2.0 and not ball_deep:
        return False   # ball NOT incoming and NOT deep → not a defensive scenario

    positions = _player_positions(eval_df)   # (n, 6, 2)

    # Players spread: mean std of Y positions across frames must be > 150 cm
    y_stds = []
    for pos in positions:
        occ = _occupied(pos)
        if len(occ) >= 3:
            y_stds.append(float(occ[:, 1].std()))
    if not y_stds or float(np.mean(y_stds)) < 150.0:
        return False

    # Low mean speed: players are positioned, not actively attacking
    speeds_cmf = np.linalg.norm(np.diff(positions, axis=0), axis=-1)  # (n-1, 6)
    if speeds_cmf.mean() * FPS > 150.0:
        return False

    # Formation shape stable: spatial variance std < 35% of mean
    centroid = np.zeros((len(positions), 2), dtype=float)
    for f, pos in enumerate(positions):
        occ = _occupied(pos)
        centroid[f] = occ.mean(axis=0) if len(occ) > 0 else 0.0

    sp_var = np.zeros(len(positions))
    for f, pos in enumerate(positions):
        occ = _occupied(pos)
        if len(occ) == 0:
            continue
        d = occ - centroid[f]
        sp_var[f] = float((d ** 2).sum(axis=1).mean())

    mean_sv = sp_var.mean()
    if mean_sv == 0:
        return False

    return sp_var.std() < 0.35 * mean_sv


def rule_delayed_support(eval_df: pd.DataFrame) -> bool:
    """
    Delayed Support – temporal failure in support movement.

    A player who should react to a ball contact (dig, set, spike) on the near
    team's side peaks their running speed MORE than 5 frames after that event —
    meaning they were slow to start moving in support.

    Requires ball to be reliably detected throughout the window (≥ 80 % of frames).
    If ball is absent or intermittent the contact event cannot be trusted.

    Steps:
      1. Verify ball is detected in ≥ 80 % of frames.
      2. Find a contact event:
           Primary  – Y-velocity of ball reverses sign (ball direction changes).
           Fallback – sudden ≥ 60 % deceleration in 1 frame.
      3. Require the ball is detected continuously for at least 3 frames
         before and 5 frames after the detected impact.
      4. Find the nearest player to the ball at the impact frame.
      5. Check that this player's peak speed comes > 5 frames after impact.
    """
    ball_x = eval_df["ball_x"].values.astype(float)
    ball_y = eval_df["ball_y"].values.astype(float)

    # Require ball detected in at least 80 % of frames (not "any zero = skip")
    detected = (ball_x != 0) | (ball_y != 0)
    if detected.mean() < 0.80:
        return False

    by_vel = np.diff(ball_y)
    bx_vel = np.diff(ball_x)
    impact = None

    # Primary: Y-velocity sign reversal (ball bounces off player)
    for i in range(1, len(by_vel)):
        if by_vel[i - 1] * by_vel[i] <= 0:
            # Both surrounding frames must be detected
            win_start = max(0, i - 2)
            win_end   = min(len(detected), i + 6)
            if detected[win_start:win_end].all():
                impact = i
                break

    # Fallback: sudden ≥ 60 % XY-speed deceleration in 1 frame
    if impact is None:
        spd = np.sqrt(bx_vel ** 2 + by_vel ** 2)
        for i in range(1, len(spd)):
            if spd[i - 1] > 5.0 and spd[i] < spd[i - 1] * 0.4:
                win_start = max(0, i - 2)
                win_end   = min(len(detected), i + 6)
                if detected[win_start:win_end].all():
                    impact = i
                    break

    if impact is None:
        return False

    positions = _player_positions(eval_df)   # (n, 6, 2)
    ball_at   = np.array([ball_x[impact], ball_y[impact]], dtype=float)

    # Find the player nearest to ball at impact (the one who should react fastest)
    occ_mask = positions[impact, :, 1] >= NEAR_TEAM_MIN_Y
    if not occ_mask.any():
        return False
    dists   = np.linalg.norm(positions[impact] - ball_at, axis=-1)
    dists[~occ_mask] = np.inf
    closest = int(np.argmin(dists))

    traj   = positions[:, closest, :]
    speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1)

    if len(speeds) == 0:
        return False

    reaction = int(np.argmax(speeds))
    return (reaction - impact) > 5


def rule_spacing_breakdown(eval_df: pd.DataFrame) -> bool:
    """
    Spacing Breakdown – structural failure in team formation.

    Fires ONLY when a genuine positional problem exists in the majority of frames
    (≥ 50 % of frames with enough players), preventing this from being a catch-all.

    Two failure modes:
      • Gap too large   : max nearest-neighbour > 450 cm  (half the court uncovered)
      • Cluster overlap : min nearest-neighbour <  50 cm  (players colliding/stacked)

    Threshold reduced from 600 → 450 cm:
      A 600 cm gap = two-thirds of the 900 cm court width — that's unrealistically
      large before being called a breakdown.  450 cm is already a serious gap.

    Requires ≥ 50 % of evaluation frames to trigger so a single noisy detection
    frame doesn't label an otherwise organised sequence as a breakdown.
    """
    positions  = _player_positions(eval_df)
    fire_count = 0
    total      = 0

    for pos in positions:
        nn = _nn_distances(pos)
        if len(nn) == 0:
            continue
        total += 1
        if nn.max() > 450 or nn.min() < 50:
            fire_count += 1

    if total == 0:
        return False
    return (fire_count / total) >= 0.50


# ──────────────────────────────────────────────────────────────────────────────
# Label assignment (priority order)
# ──────────────────────────────────────────────────────────────────────────────

def label_clip(df: pd.DataFrame) -> str:
    """
    Apply rules to ALL rows in df and return label.
    Callers are responsible for passing the correct evaluation window.

    Priority (first match wins):
      1. Coordinated Attack  – ball near net, attackers sprinting toward net
      2. Coordinated Defense – ball incoming, players spread, low speed, stable shape
      3. Delayed Support     – player reaction peaks > 5 frames after ball contact
      4. Spacing Breakdown   – gap > 450 cm or overlap < 50 cm in ≥ 50 % of frames
      else → Unclassified    (no strong pattern found — NOT saved as training data)
    """
    if rule_coordinated_attack(df):   return "Coordinated Attack"
    if rule_coordinated_defense(df):  return "Coordinated Defense"
    if rule_delayed_support(df):      return "Delayed Support"
    if rule_spacing_breakdown(df):    return "Spacing Breakdown"
    return "Unclassified"


# ──────────────────────────────────────────────────────────────────────────────
# Report generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(label_counts: dict) -> str:
    total = sum(label_counts.values()) or 1
    use_color = sys.stdout.isatty()

    def bar(n):
        w = 20
        f = round(n / total * w)
        return "█" * f + "░" * (w - f)

    def pct(n):
        return f"{round(n/total*100):3d}%"

    n_ca = label_counts.get("Coordinated Attack", 0)
    n_cd = label_counts.get("Coordinated Defense", 0)
    n_ds = label_counts.get("Delayed Support", 0)
    n_sb = label_counts.get("Spacing Breakdown", 0)

    coordinated = n_ca + n_cd
    efficiency  = round(coordinated / total * 100)
    sep = "=" * 58

    rows = [
        "", sep, "         Team Coordination Analysis Report", sep,
        _c("Coordinated Attack",
           f"  [1] Coordinated Attack   : {n_ca:4d}  {pct(n_ca)}  {bar(n_ca)}", use_color),
        _c("Coordinated Defense",
           f"  [2] Coordinated Defense  : {n_cd:4d}  {pct(n_cd)}  {bar(n_cd)}", use_color),
        _c("Delayed Support",
           f"  [3] Delayed Support      : {n_ds:4d}  {pct(n_ds)}  {bar(n_ds)}", use_color),
        _c("Spacing Breakdown",
           f"  [4] Spacing Breakdown    : {n_sb:4d}  {pct(n_sb)}  {bar(n_sb)}", use_color),
        "",
        f"  Total clips analysed          : {total}",
        f"  Coordination efficiency score : {efficiency}%",
    ]

    if total > 1:
        dominant = max(label_counts, key=label_counts.get)
        insight_map = {
            "Spacing Breakdown":   "High Spacing Breakdown – work on formation & positioning.",
            "Delayed Support":     "High Delayed Support   – improve reaction time to ball contact.",
            "Coordinated Attack":  "High Coordinated Attack – strong offensive structure.",
            "Coordinated Defense": "High Coordinated Defense – excellent defensive cohesion.",
        }
        qual = (
            "GOOD (≥ 60% positive)"      if efficiency >= 60 else
            "MODERATE (40–59% positive)" if efficiency >= 40 else
            "NEEDS IMPROVEMENT (< 40%)"
        )
        rows += [
            "", "  Insights:",
            f"  • {insight_map.get(dominant, '')}",
            f"  • Overall coordination quality: {qual}.",
        ]

    rows.append(sep)
    return "\n".join(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Batch processing loop
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = (
    {"frame_id", "ball_x", "ball_y"}
    | {c for pair in PLAYER_COLS for c in pair}
)


def process_directory(input_dir: str) -> None:
    """
    For every CSV in input_dir:
      1. Load the CSV (must have 41 rows, columns as above).
      2. Apply rules on frames 30–41 to determine label.
      3. If Unclassified → delete the file.
      4. Otherwise, keep frames 1–29, add 'target_label' column, overwrite.
    """
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'.")
        return

    processed = deleted = errors = 0
    label_counts: dict[str, int] = {}

    for filename in sorted(csv_files):
        filepath = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as exc:
            print(f"[ERROR] {filename}: {exc}")
            errors += 1
            continue

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(f"[SKIP]  {filename} – missing columns: {missing}")
            errors += 1
            continue

        eval_df  = df[(df["frame_id"] >= EVAL_START) & (df["frame_id"] <= EVAL_END)].copy()
        train_df = df[(df["frame_id"] >= 1)          & (df["frame_id"] <= TRAIN_END)].copy()

        if eval_df.empty:
            print(f"[SKIP]  {filename} – evaluation window empty")
            errors += 1
            continue

        label = label_clip(eval_df)

        if label == "Unclassified":
            os.remove(filepath)
            print(f"[DELETE] {filename} – Unclassified")
            deleted += 1
            continue

        train_df = train_df.reset_index(drop=True)
        train_df["target_label"] = label
        train_df.to_csv(filepath, index=False)

        idx = LABEL_TO_INDEX.get(label, 0)
        print(f"[OK]    {filename} – [{idx}] {label}")
        label_counts[label] = label_counts.get(label, 0) + 1
        processed += 1

    print(f"\nDone.  Processed={processed}  Deleted={deleted}  Errors/Skipped={errors}")
    if label_counts:
        print(generate_report(label_counts))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python label_clips.py <directory_of_41frame_csvs>")
        sys.exit(1)
    d = sys.argv[1]
    if not os.path.isdir(d):
        print(f"Error: '{d}' is not a directory.")
        sys.exit(1)
    process_directory(d)
