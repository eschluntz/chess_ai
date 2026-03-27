"""
Transformer policy network model.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.nn import functional as F

# Chess board constants (determined by data format, not tunable)
N_SQUARES = 64
N_PIECES = 14  # 6 mine + 6 theirs + empty + en-passant target
N_META = 5  # color + 4 castling rights
# Spatial head: 64 dest planes + 4 promo-type correction planes (N,B,R,Q)
N_PLANES = N_SQUARES + 4  # 68


@dataclass
class TransformerConfig:
    n_embed: int = 256
    num_heads: int = 8
    ff_ratio: int = 4
    num_layers: int = 8
    num_moves: int = 1968
    head_type: str = "flatten"  # "flatten" or "spatial_unshared"
    pos_encoding: str = "absolute"  # "absolute", "bias64", or "shaw2d"

    def default_run_name(self) -> str:
        return f"tf_L{self.num_layers}_D{self.n_embed}_H{self.num_heads}"

    def to_dict(self) -> dict:
        return asdict(self)

    def flops_per_sample(self) -> int:
        """Approximate training FLOPs per sample (forward + backward ≈ 3× forward).

        Counts dominant matmuls only; ±15% vs exact. Consistent across configs
        so ratios are accurate for pareto comparisons.
        """
        d, L, T = self.n_embed, self.num_layers, N_SQUARES
        per_block = 3 * T * d * d + 2 * T * T * d + T * d * d + 2 * T * d * 4 * d
        # qkv (3Td²) + sdpa (2T²d) + proj (Td²) + ffn up+down (2·T·4d·d)
        backbone = L * per_block
        if self.head_type == "flatten":
            head = T * d * self.num_moves
        else:
            head = T * d * N_PLANES
        return 3 * (backbone + head)  # ×3 for fwd+bwd


def _spatial_gather(vocab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build gather indices for the 68-plane spatial head.

    vocab: (M, 3) long tensor of (from_sq, to_sq, promo).
    Returns (base_idx, promo_idx, promo_mask), each shape (M,).
      base_idx:   from_sq * 68 + to_sq              — always gathered
      promo_idx:  from_sq * 68 + 64 + (promo-1)     — gathered only when promo>0
      promo_mask: 1.0 where promo>0 else 0.0
    """
    from_sq, to_sq, promo = vocab[:, 0], vocab[:, 1], vocab[:, 2]
    base_idx = from_sq * N_PLANES + to_sq
    promo_mask = (promo > 0).float()
    promo_plane = N_SQUARES + (promo - 1).clamp(min=0)
    promo_idx_raw = from_sq * N_PLANES + promo_plane
    promo_idx = torch.where(promo > 0, promo_idx_raw, base_idx)
    return base_idx, promo_idx, promo_mask


class SpatialUnsharedHead(nn.Module):
    """Per-square C→68 decoder. Each source square has its own weight matrix.

    logit(f→t, 0) = out[f, t]
    logit(f→t, p) = out[f, t] + out[f, 64+p-1]   for promo p in {1,2,3,4}

    Consumes (B, 64, C) — the transformer's natural per-token output.
    """

    def __init__(self, C: int, vocab: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(N_SQUARES, C, N_PLANES))
        self.bias = nn.Parameter(torch.zeros(N_SQUARES, N_PLANES))
        nn.init.uniform_(self.weight, -1 / math.sqrt(C), 1 / math.sqrt(C))
        base, promo, mask = _spatial_gather(vocab)
        self.register_buffer("base_idx", base)
        self.register_buffer("promo_idx", promo)
        self.register_buffer("promo_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, C) — already normed by the transformer's final norm
        out = torch.einsum("bsc,scp->bsp", x, self.weight) + self.bias  # (B, 64, 68)
        flat = out.flatten(1)  # (B, 64*68)
        return flat[:, self.base_idx] + flat[:, self.promo_idx] * self.promo_mask


def _displacement_buckets() -> torch.Tensor:
    """Precompute (64, 64) lookup mapping each (i, j) square pair to its
    displacement bucket in [0, 225). bucket = (Δrank+7)*15 + (Δfile+7)."""
    sq = torch.arange(N_SQUARES)
    rank, file = sq // 8, sq % 8
    d_rank = rank[None, :] - rank[:, None]  # (64, 64), range [-7, 7]
    d_file = file[None, :] - file[:, None]
    return (d_rank + 7) * 15 + (d_file + 7)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional chess-aware position encoding.

    pos_encoding modes:
      "absolute" — no attention bias (relies on input-layer position embedding)
      "bias64"   — per-head learned (64, 64) additive bias on attention logits
      "shaw2d"   — Shaw-style relative vectors indexed by (Δrank, Δfile) bucket,
                   225 buckets shared across heads. Hand-rolls attention.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_attn = config.n_embed
        self.num_heads = config.num_heads
        self.head_size = self.d_attn // config.num_heads
        self.pos_encoding = config.pos_encoding
        assert self.d_attn % config.num_heads == 0, (
            "d_attn must be divisible by number of heads"
        )
        assert self.pos_encoding in ("absolute", "bias64", "shaw2d"), (
            f"unknown pos_encoding: {self.pos_encoding!r}"
        )

        self.qkv = nn.Linear(config.n_embed, 3 * self.d_attn, bias=False)
        self.proj = nn.Linear(self.d_attn, config.n_embed)

        if self.pos_encoding == "bias64":
            self.pos_bias = nn.Parameter(
                torch.zeros(config.num_heads, N_SQUARES, N_SQUARES)
            )
        elif self.pos_encoding == "shaw2d":
            n_buckets = 15 * 15
            self.rel_k = nn.Parameter(torch.zeros(n_buckets, self.head_size))
            self.rel_v = nn.Parameter(torch.zeros(n_buckets, self.head_size))
            nn.init.normal_(self.rel_k, std=0.02)
            nn.init.normal_(self.rel_v, std=0.02)
            self.register_buffer("bucket", _displacement_buckets())

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "b t (three nh h) -> three b nh t h", three=3, nh=self.num_heads
        )

        if self.pos_encoding == "absolute":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        elif self.pos_encoding == "bias64":
            # pos_bias is (H, T, T); [None] adds a batch dim -> (1, H, T, T)
            # so it broadcasts against SDPA's (B, H, T, T) attention logits.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=self.pos_bias[None], is_causal=False
            )
        elif self.pos_encoding == "shaw2d":
            out = self._shaw_attention(q, k, v)

        out = rearrange(out, "b nh t h -> b t (nh h)")
        return self.proj(out)

    def _shaw_attention(self, q, k, v):
        """Shaw et al. 2018 relative attention, adapted for 2D chess displacement.

        Standard attention computes logit[i,j] = q_i · k_j. Shaw adds a second
        term that depends only on the *relative position* of j from i:

          logit[i,j] = q_i · k_j  +  q_i · rel_k[bucket[i,j]]
          out[i]     = Σ_j attn[i,j] · (v_j + rel_v[bucket[i,j]])

        bucket[i,j] maps each (from-square, to-square) pair to one of 225
        displacement buckets, so e.g. "one rank up, two files left" always
        hits the same learned vector regardless of absolute board position.

        Einsum index convention throughout:
          b=batch, h=head, i=query square, j=key/value square, d=head_size
        """
        scale = 1.0 / math.sqrt(self.head_size)

        # Expand the 225-entry tables into full (T, T, d) tensors by indexing
        # with the bucket lookup. self.bucket is (T, T) with values in [0, 225);
        # rel_k[self.bucket] does a gather -> rk[i,j,:] = rel_k[bucket[i,j],:].
        rk = self.rel_k[self.bucket]  # (T, T, d)
        rv = self.rel_v[self.bucket]  # (T, T, d)

        # --- attention logits ---
        # content term: for each (i,j), dot q_i with k_j over the d axis.
        #   "bhid,bhjd->bhij" = sum_d q[b,h,i,d] * k[b,h,j,d]
        content_logits = torch.einsum("bhid,bhjd->bhij", q, k)

        # position term: for each (i,j), dot q_i with the relative-key vector
        # for that displacement. rk has no batch/head dims so they broadcast.
        #   "bhid,ijd->bhij" = sum_d q[b,h,i,d] * rk[i,j,d]
        position_logits = torch.einsum("bhid,ijd->bhij", q, rk)

        attn = F.softmax((content_logits + position_logits) * scale, dim=-1)

        # --- weighted values ---
        # content term: standard attn·v. Sum over j (the keys).
        #   "bhij,bhjd->bhid" = sum_j attn[b,h,i,j] * v[b,h,j,d]
        content_out = torch.einsum("bhij,bhjd->bhid", attn, v)

        # position term: same weights, but applied to the relative-value table.
        #   "bhij,ijd->bhid" = sum_j attn[b,h,i,j] * rv[i,j,d]
        position_out = torch.einsum("bhij,ijd->bhid", attn, rv)

        return content_out + position_out


class FeedForward(nn.Module):
    """MLP block with expansion and projection"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_dim = config.n_embed * config.ff_ratio
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embed),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention then feedforward with residual connections"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.norm1 = nn.RMSNorm(config.n_embed).to(dtype=torch.bfloat16)
        self.norm2 = nn.RMSNorm(config.n_embed).to(dtype=torch.bfloat16)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Transformer for chess move prediction.

    Tokenization:
    - Each of the 64 squares is a token. Empty squares get an "empty" embedding.
    - Board is normalized to current player's perspective (flip when black to move).

    Input embedding:
    - token_embed = piece_embedding[piece_at_square] + position_embedding[square]
    - Meta (color + castling) projected to n_embed and broadcast-added to all tokens.

    Input:  planes uint8 (batch, 13, 8, 8), meta int8 (batch, 5) = [turn, K, Q, k, q]
    Output: (batch, num_moves) logits
    """

    def __init__(self, config: TransformerConfig, vocab: torch.Tensor | None = None):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(N_PIECES, config.n_embed)
        self.position_embedding = nn.Embedding(N_SQUARES, config.n_embed)
        self.meta_linear = nn.Linear(N_META, config.n_embed)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.RMSNorm(config.n_embed).to(dtype=torch.bfloat16)

        if config.head_type == "flatten":
            self.lm_head = nn.Linear(config.n_embed * N_SQUARES, config.num_moves)
        elif config.head_type == "spatial_unshared":
            assert vocab is not None, "spatial_unshared head requires vocab"
            self.lm_head = SpatialUnsharedHead(config.n_embed, vocab)

    def planes_to_toks_flipped(
        self, planes: torch.Tensor, meta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize to current player's perspective and convert planes → token IDs.

        When it's black's turn (meta[:, 0] == -1):
        - Swap planes 0-5 (white) with 6-11 (black) → "my pieces" vs "their pieces"
        - Flip ranks so current player's pieces are at the bottom
        - Swap castling rights (mine vs theirs)

        Returns:
            tokens: (batch, 64) long — 0=empty, 1-12=pieces, 13=EP target
            flipped_meta: (batch, 5) — color bit kept, castling normalized to mine/theirs
        """
        device = planes.device
        is_black = meta[:, 0] == -1  # (batch,)

        if is_black.any():
            flipped_planes = torch.cat(
                [planes[:, 6:12], planes[:, 0:6], planes[:, 12:13]], dim=1
            )
            flipped_planes = flipped_planes.flip(dims=[-2])

            # [turn, K, Q, k, q] → [turn, k, q, K, Q]
            flipped_meta = torch.stack(
                [meta[:, 0], meta[:, 3], meta[:, 4], meta[:, 1], meta[:, 2]], dim=1
            )

            mask = is_black[:, None, None, None]
            planes = torch.where(mask, flipped_planes, planes)
            meta = torch.where(is_black[:, None], flipped_meta, meta)

        # Planes are one-hot per square → multiply by 1..13 and sum to get token ID
        piece_idx = torch.arange(1, 14, device=device).view(1, 13, 1, 1)
        idx_boards = piece_idx * planes
        tokens = reduce(idx_boards, "b idx h w -> b h w", reduction="sum")
        tokens = rearrange(tokens, "b h w -> b (h w)")

        return tokens, meta

    def forward(self, planes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        toks, flipped_meta = self.planes_to_toks_flipped(planes, meta)
        _, T = toks.shape
        device = toks.device

        tok_emb = self.token_embedding(toks)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        meta_emb = self.meta_linear(flipped_meta.float())
        meta_emb = rearrange(meta_emb, "b d -> b 1 d")
        x = tok_emb + pos_emb + meta_emb
        # Embedding output is fp32 (autocast skips lookups); cast so RMSNorm's bf16
        # weights match the input and the fused kernel fires. CUDA-only because
        # on CPU there's no autocast to handle the downstream fp32-weight Linears.
        if x.is_cuda:
            x = x.to(torch.bfloat16)

        x = self.blocks(x)
        x = self.norm(x)
        if self.config.head_type == "flatten":
            x = rearrange(x, "b t n -> b (t n)")
        return self.lm_head(x)
