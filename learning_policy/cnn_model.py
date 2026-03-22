"""
CNN policy network model.

Output head variants (all consume backbone output (B, C, 8, 8) → (B, num_moves)):

  flatten          Linear(C*64, M)                   ~16M params @ C=128. Baseline.
  reduce           Conv1x1(C→Cs) → Linear(Cs*64, M)  ~4M @ Cs=32. Bottlenecked but full interaction.
  spatial_shared   Conv1x1(C→68) + gather            ~9K. logit(f→t) depends only on features at f.
  spatial_unshared einsum with (64, C, 68) + gather   ~560K. Per-square separate decoder.
  bilinear         Two Conv1x1(C→D), dot product      ~16K @ D=64. Uses features at both f and t.
  factored         Linear(C*64,64)×2, score=f[from]+g[to]  ~1M. No from/to interaction.

Spatial/bilinear/factored heads require a vocab tensor (M, 3) of (from_sq, to_sq, promo)
to build gather indices at init.
"""

import math
import torch
import torch.nn as nn
from einops import pack, repeat

W, H = 8, 8
N_SQ = W * H  # 64
# Spatial heads: 64 dest planes + 4 promo-type correction planes (N,B,R,Q → promo codes 1–4).
# logit(f→t, 0) = out[f, t]
# logit(f→t, p) = out[f, t] + out[f, 64 + p - 1]   for p in {1,2,3,4}
N_PLANES = N_SQ + 4  # 68


class ResCNNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.ln1 = nn.RMSNorm([channels, W, H]).to(dtype=torch.bfloat16)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding="same")
        self.gelu1 = nn.GELU()
        self.ln2 = nn.RMSNorm([channels, W, H]).to(dtype=torch.bfloat16)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding="same")
        self.gelu2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gelu1(self.conv1(self.ln1(x)))
        y = self.gelu2(self.conv2(self.ln2(y)))
        return x + y


# ---------------------------------------------------------------------------
# Output heads. All take (B, C, 8, 8) → (B, num_moves).
# ---------------------------------------------------------------------------


def _spatial_gather(vocab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build gather indices for the 68-plane spatial heads.

    vocab: (M, 3) long tensor of (from_sq, to_sq, promo).
    Returns (base_idx, promo_idx, promo_mask), each shape (M,).
      base_idx:   from_sq * 68 + to_sq                 — always gathered
      promo_idx:  from_sq * 68 + 64 + (promo-1)        — gathered only when promo>0
      promo_mask: 1.0 where promo>0 else 0.0           — zeros out promo term otherwise
    For promo=0 rows, promo_idx is set to base_idx (value irrelevant; masked to 0).
    """
    from_sq, to_sq, promo = vocab[:, 0], vocab[:, 1], vocab[:, 2]
    base_idx = from_sq * N_PLANES + to_sq
    promo_mask = (promo > 0).float()
    promo_plane = N_SQ + (promo - 1).clamp(min=0)
    promo_idx_raw = from_sq * N_PLANES + promo_plane
    promo_idx = torch.where(promo > 0, promo_idx_raw, base_idx)
    return base_idx, promo_idx, promo_mask


class FlattenHead(nn.Module):
    """Linear(C*64, M). The baseline giant head."""

    def __init__(self, C: int, num_moves: int):
        super().__init__()
        self.norm = nn.RMSNorm(C * N_SQ).to(dtype=torch.bfloat16)
        self.out = nn.Linear(C * N_SQ, num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(x.flatten(1)))


class ReduceHead(nn.Module):
    """Conv1x1(C→Cs) → Linear(Cs*64, M). Bottlenecked flatten."""

    def __init__(self, C: int, Cs: int, num_moves: int):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(C, Cs, 1), nn.GELU())
        self.norm = nn.RMSNorm(Cs * N_SQ).to(dtype=torch.bfloat16)
        self.out = nn.Linear(Cs * N_SQ, num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.reduce(x).flatten(1)))


class SpatialSharedHead(nn.Module):
    """Conv1x1(C→68) applied at every square, then gather per-move logits.

    Planes 0–63 encode destination square; planes 64–67 encode promo-type (N,B,R,Q)
    as an additive correction. Shared weights across source squares.
    """

    def __init__(self, C: int, vocab: torch.Tensor):
        super().__init__()
        self.norm = nn.RMSNorm([C, W, H]).to(dtype=torch.bfloat16)
        self.conv = nn.Conv2d(C, N_PLANES, 1)
        base, promo, mask = _spatial_gather(vocab)
        self.register_buffer("base_idx", base)
        self.register_buffer("promo_idx", promo)
        self.register_buffer("promo_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.norm(x))                 # (B, 68, 8, 8)
        flat = x.flatten(2).transpose(1, 2).flatten(1)  # (B, 64*68)
        return flat[:, self.base_idx] + flat[:, self.promo_idx] * self.promo_mask


class SpatialUnsharedHead(nn.Module):
    """Per-square separate C→68 projection. Each source square has its own decoder."""

    def __init__(self, C: int, vocab: torch.Tensor):
        super().__init__()
        self.norm = nn.RMSNorm([C, W, H]).to(dtype=torch.bfloat16)
        # weight[s, c, p] projects channel c at square s to plane p
        self.weight = nn.Parameter(torch.empty(N_SQ, C, N_PLANES))
        self.bias = nn.Parameter(torch.zeros(N_SQ, N_PLANES))
        bound = 1 / math.sqrt(C)
        nn.init.uniform_(self.weight, -bound, bound)
        base, promo, mask = _spatial_gather(vocab)
        self.register_buffer("base_idx", base)
        self.register_buffer("promo_idx", promo)
        self.register_buffer("promo_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).flatten(2).transpose(1, 2)           # (B, 64, C)
        out = torch.einsum("bsc,scp->bsp", x, self.weight)    # (B, 64, 68)
        flat = (out + self.bias).flatten(1)                   # (B, 64*68)
        return flat[:, self.base_idx] + flat[:, self.promo_idx] * self.promo_mask


class BilinearHead(nn.Module):
    """Separate from/to projections, score = <from_feat, to_feat>.

    logit(from→to) depends on features at BOTH endpoints (unlike spatial heads).
    Promotions distinguished by a learnable per-promo-type bias only (5 scalars).
    """

    def __init__(self, C: int, vocab: torch.Tensor, D: int = 64):
        super().__init__()
        self.norm = nn.RMSNorm([C, W, H]).to(dtype=torch.bfloat16)
        self.from_proj = nn.Conv2d(C, D, 1)
        self.to_proj = nn.Conv2d(C, D, 1)
        self.promo_bias = nn.Parameter(torch.zeros(5))
        self.register_buffer("from_idx", vocab[:, 0])
        self.register_buffer("to_idx", vocab[:, 1])
        self.register_buffer("promo_idx", vocab[:, 2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        f = self.from_proj(x).flatten(2)  # (B, D, 64)
        t = self.to_proj(x).flatten(2)    # (B, D, 64)
        # score[b, m] = sum_d f[b, d, from[m]] * t[b, d, to[m]]
        score = (f[:, :, self.from_idx] * t[:, :, self.to_idx]).sum(1)
        return score + self.promo_bias[self.promo_idx]


class FactoredHead(nn.Module):
    """Two global linear heads: score = origin_logit[from] + dest_logit[to].

    Cannot express interactions between from and to. Included as a
    lower-bound baseline.
    """

    def __init__(self, C: int, vocab: torch.Tensor):
        super().__init__()
        self.norm = nn.RMSNorm(C * N_SQ).to(dtype=torch.bfloat16)
        self.from_head = nn.Linear(C * N_SQ, N_SQ)
        self.to_head = nn.Linear(C * N_SQ, N_SQ)
        self.promo_bias = nn.Parameter(torch.zeros(5))
        self.register_buffer("from_idx", vocab[:, 0])
        self.register_buffer("to_idx", vocab[:, 1])
        self.register_buffer("promo_idx", vocab[:, 2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.flatten(1))
        f = self.from_head(x)  # (B, 64)
        t = self.to_head(x)    # (B, 64)
        return f[:, self.from_idx] + t[:, self.to_idx] + self.promo_bias[self.promo_idx]


class PolicyCNN(nn.Module):
    """CNN for move prediction with configurable depth and width.

    Takes Input:
    - planes uint8 (batch, 13, 8, 8),
    - meta: int8[5] - [turn, K, Q, k, q] (batch, 5)
    Output is (batch, num_moves) logits for vocab indices.
    """

    def combine_planes_and_meta(
        self, planes: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        """concat planes and broadcasted meta information into a single tensor.
        planes (batch, 13, 8, 8). meta (batch, 5). output (batch, 13 + 5, 8, 8)"""

        # unsqueeze w,h dims, then expand (Broadcast) those dims to be the right size
        meta_pl = repeat(meta, "b n -> b n h w", h=H, w=W)
        combined, _ = pack([planes, meta_pl], "b * h w")
        return combined.float()  # (batch, 13 + 5, 8, 8)

    def combine_planes_and_meta_flipped(
        self, planes: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        """Normalize board to always be from the current player's perspective.

        When it's black's turn (meta[:, 0] == -1):
        - Swap planes 0-5 (white) with 6-11 (black) → "my pieces" vs "their pieces"
        - Flip ranks so current player's pieces are at the bottom
        - Swap castling rights (mine vs theirs)

        planes (batch, 13, 8, 8). meta (batch, 5). output (batch, 13 + 4, 8, 8).
        Turn channel is dropped since it's always "my turn".
        """
        is_black = (meta[:, 0] == -1)  # (batch,)

        if is_black.any():
            # Swap white/black planes and flip ranks for black's perspective
            flipped_planes = torch.cat([planes[:, 6:12], planes[:, 0:6], planes[:, 12:13]], dim=1)
            flipped_planes = flipped_planes.flip(dims=[-2])

            # Swap castling: [turn, K, Q, k, q] → [turn, k, q, K, Q]
            flipped_meta = torch.stack([meta[:, 0], meta[:, 3], meta[:, 4], meta[:, 1], meta[:, 2]], dim=1)

            mask = is_black[:, None, None, None]  # (batch, 1, 1, 1) for planes
            planes = torch.where(mask, flipped_planes, planes)

            mask_meta = is_black[:, None]  # (batch, 1) for meta
            meta = torch.where(mask_meta, flipped_meta, meta)

        # Drop turn channel (index 0), keep castling rights (my_K, my_Q, their_k, their_q)
        castling = meta[:, 1:5]
        meta_pl = repeat(castling, "b n -> b n h w", h=H, w=W)
        combined, _ = pack([planes, meta_pl], "b * h w")
        return combined.float()  # (batch, 13 + 4, 8, 8)

    def __init__(
        self,
        num_moves: int,
        hidden_channels: int = 32,
        num_layers: int = 1,
        kernel_size: int = 15,
        flip_board: bool = False,
        head_type: str = "flatten",
        head_channels: int = 32,       # only used for head_type="reduce"
        bilinear_dim: int = 64,        # only used for head_type="bilinear"
        vocab: torch.Tensor = None,    # (M, 3) long, required for spatial/bilinear/factored
    ):
        super().__init__()
        self.flip_board = flip_board

        # 13 piece/ep planes + meta channels (4 if flipped, 5 if not)
        input_channels = 13 + (4 if flip_board else 5)
        layers = []

        # input -> hidden channels
        layers.append(
            nn.Conv2d(input_channels, hidden_channels, kernel_size, padding="same")
        )
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers):
            layers.append(ResCNNBlock(hidden_channels, kernel_size))

        self.layers = nn.Sequential(*layers)

        C = hidden_channels
        if head_type == "flatten":
            self.head = FlattenHead(C, num_moves)
        elif head_type == "reduce":
            self.head = ReduceHead(C, head_channels, num_moves)
        elif head_type == "spatial_shared":
            self.head = SpatialSharedHead(C, vocab)
        elif head_type == "spatial_unshared":
            self.head = SpatialUnsharedHead(C, vocab)
        elif head_type == "bilinear":
            self.head = BilinearHead(C, vocab, D=bilinear_dim)
        elif head_type == "factored":
            self.head = FactoredHead(C, vocab)

    def forward(self, planes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # planes (batch, 13, 8, 8)
        # meta (batch, 5)
        if self.flip_board:
            x = self.combine_planes_and_meta_flipped(planes, meta)  # (batch, 17, 8, 8)
        else:
            x = self.combine_planes_and_meta(planes, meta)  # (batch, 18, 8, 8)
        x = self.layers(x)  # (batch, C, 8, 8)
        return self.head(x)
