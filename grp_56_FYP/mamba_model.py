"""
mamba_model.py – Mamba Selective State Space Model Classifier
             for Volleyball Tactical Analysis.

Camera Setup (Back-Center End-Line View):
  Camera is placed behind and centered on Team A's end line, elevated 6-10m.
  Court coordinate system (after homography):
    X : 0 – 900 cm   (9m court width, left sideline → right sideline)
    Y : 0 – 1800 cm  (18m court length, far end → camera end)
    Net at Y = 900 cm
    Team A (near, tracked): Y = 900 – 1800 cm
    Team B (far, not tracked): Y = 0 – 900 cm

Architecture:
  Input  : (batch, seq_len=29, input_dim=14)
           14 = ball_x, ball_y + p1_x,p1_y … p6_x,p6_y  (court cm)
  Encoder: Linear projection → N × MambaBlock → LayerNorm
  Head   : Mean-pool over time → Dropout → Linear → 4-class softmax

4 Tactic Classes:
  0 – Spacing Breakdown
  1 – Delayed Support
  2 – Coordinated Attack
  3 – Coordinated Defense

Reference: Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with
           Selective State Spaces. arXiv:2312.00752.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Shared constants (imported by all other modules)
# ──────────────────────────────────────────────────────────────────────────────

COURT_WIDTH_CM  = 900    # X axis: left sideline to right sideline
COURT_LENGTH_CM = 1800   # Y axis: far end (Y=0) to camera end (Y=1800)
NET_Y_CM        = 900    # Net bisects the 18m length at Y=900

NEAR_TEAM_MIN_Y = NET_Y_CM        # Team A (tracked): Y ≥ 900 cm
NEAR_TEAM_MAX_Y = COURT_LENGTH_CM # Team A (tracked): Y ≤ 1800 cm

SEQ_LEN    = 29   # frames per classification window
N_PLAYERS  = 6    # players tracked per team
INPUT_DIM  = 14   # 2 (ball) + 6×2 (players)

FEATURE_COLS = [
    "ball_x", "ball_y",
    "p1_x", "p1_y",
    "p2_x", "p2_y",
    "p3_x", "p3_y",
    "p4_x", "p4_y",
    "p5_x", "p5_y",
    "p6_x", "p6_y",
]

# Label index → name (0-based for PyTorch CrossEntropyLoss)
LABEL_NAMES = [
    "Spacing Breakdown",    # 0
    "Delayed Support",      # 1
    "Coordinated Attack",   # 2
    "Coordinated Defense",  # 3
]
NUM_CLASSES = len(LABEL_NAMES)

# 1-based numeric index used in reports / CSV output
LABEL_TO_IDX = {
    "Coordinated Attack":  1,
    "Coordinated Defense": 2,
    "Delayed Support":     3,
    "Spacing Breakdown":   4,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Selective State Space Model (S6) – core Mamba building block
# ──────────────────────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Single Mamba block implementing the selective scan (S6) mechanism.

    Parameters
    ----------
    d_model : inner model dimension
    d_state : SSM state dimension N  (paper default 16)
    d_conv  : causal depthwise-conv width (paper default 4)
    dt_rank : rank of the Δ projection  (default ceil(d_model/16))
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Input expansion (gate + inner)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)

        # Causal depthwise conv along the time axis
        self.conv1d = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_model, bias=True,
        )

        # SSM parameter projections: inner → (dt_rank + 2·N)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)

        # Δ (dt) projection with initialisation matching the paper
        dt_init_std = self.dt_rank ** -0.5
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_bias = torch.empty(d_model).uniform_(0.001, 0.1)
        self.dt_proj.bias = nn.Parameter(torch.log(torch.expm1(dt_bias)))

        # A: log-space diagonal state-transition matrix (D × N)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A.expand(d_model, -1)))

        # D: residual skip weight
        self.D = nn.Parameter(torch.ones(d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # ── Selective scan ────────────────────────────────────────────────────────

    def _selective_scan(
        self,
        x:     torch.Tensor,  # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A:     torch.Tensor,  # (D, N)
        B:     torch.Tensor,  # (B, L, N)
        C:     torch.Tensor,  # (B, L, N)
    ) -> torch.Tensor:        # (B, L, D)
        """Discretised ZOH recurrence (sequential; no CUDA kernel required)."""
        B_sz, L, D = x.shape
        N = A.shape[1]

        # Ā = exp(Δ ⊗ A)  shape: (B, L, D, N)
        delta_A  = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )
        # B̄x = Δ ⊗ B ⊗ x  shape: (B, L, D, N)
        delta_Bx = (
            delta.unsqueeze(-1)   # (B, L, D, 1)
            * B.unsqueeze(2)      # (B, L, 1, N)
            * x.unsqueeze(-1)     # (B, L, D, 1)
        )

        h = x.new_zeros(B_sz, D, N)
        ys: list[torch.Tensor] = []
        for t in range(L):
            h  = delta_A[:, t] * h + delta_Bx[:, t]      # (B, D, N)
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1)) # (B, D)

        return torch.stack(ys, dim=1)   # (B, L, D)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, _ = x.shape

        xz          = self.in_proj(x)                   # (B, L, 2D)
        x_in, z     = xz.chunk(2, dim=-1)               # each (B, L, D)

        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)                         # (B, L, D)

        bcdt      = self.x_proj(x_conv)                 # (B, L, dt_rank+2N)
        delta_raw = bcdt[..., :self.dt_rank]
        B_mat     = bcdt[..., self.dt_rank: self.dt_rank + self.d_state]
        C_mat     = bcdt[..., self.dt_rank + self.d_state:]

        delta = F.softplus(self.dt_proj(delta_raw))     # (B, L, D)
        A     = -torch.exp(self.A_log.float())           # (D, N)

        y = self._selective_scan(x_conv, delta, A, B_mat, C_mat)  # (B, L, D)
        y = y + x_conv * self.D                         # skip connection
        y = y * F.silu(z)                               # gating

        return self.out_proj(y)                         # (B, L, D)


# ──────────────────────────────────────────────────────────────────────────────
# Mamba Block  (pre-norm residual wrapper)
# ──────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Pre-LayerNorm residual Mamba block."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ──────────────────────────────────────────────────────────────────────────────
# Full Mamba Sequence Classifier
# ──────────────────────────────────────────────────────────────────────────────

class MambaClassifier(nn.Module):
    """
    Mamba-based 4-class sequence classifier for volleyball tactical analysis.

    Parameters
    ----------
    input_dim   : features per frame (default 14)
    d_model     : model dimension (default 64)
    n_layers    : stacked Mamba blocks (default 4)
    d_state     : SSM state dimension (default 16)
    d_conv      : conv kernel width (default 4)
    num_classes : output classes (default 4)
    dropout     : classifier head dropout (default 0.2)
    """

    def __init__(
        self,
        input_dim:   int   = INPUT_DIM,
        d_model:     int   = 64,
        n_layers:    int   = 4,
        d_state:     int   = 16,
        d_conv:      int   = 4,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()

        # Linear input embedding
        self.embed = nn.Linear(input_dim, d_model)

        # Mamba backbone
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state, d_conv=d_conv)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # Classifier head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, input_dim) → logits: (B, num_classes)"""
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        h = h.mean(dim=1)       # global average pool over time
        return self.head(h)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[str, torch.Tensor]:
        """
        Predict tactic for one clip.
        x: (seq_len, input_dim) or (1, seq_len, input_dim)
        Returns (label_string, probs_tensor)
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        probs = F.softmax(self(x), dim=-1)[0]
        return LABEL_NAMES[probs.argmax().item()], probs
