"""
Transformer policy network model.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.nn import functional as F

# Chess board constants (determined by data format, not tunable)
N_SQUARES = 64
N_PIECES = 14  # 6 mine + 6 theirs + empty + en-passant target
N_META = 5  # color + 4 castling rights


@dataclass
class TransformerConfig:
    n_embed: int = 256
    num_heads: int = 8
    ff_ratio: int = 4
    num_layers: int = 8
    num_moves: int = 1968

    def default_run_name(self) -> str:
        return f"tf_L{self.num_layers}_D{self.n_embed}_H{self.num_heads}"

    def to_dict(self) -> dict:
        return asdict(self)


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention using fused scaled_dot_product_attention"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_attn = config.n_embed  # convention, not required
        self.num_heads = config.num_heads
        self.head_size = self.d_attn // config.num_heads
        assert self.d_attn % config.num_heads == 0, (
            "d_attn must be divisible by number of heads"
        )

        self.qkv = nn.Linear(config.n_embed, 3 * self.d_attn, bias=False)
        self.proj = nn.Linear(self.d_attn, config.n_embed)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "b t (three nh h) -> three b nh t h", three=3, nh=self.num_heads
        )

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
        out = rearrange(out, "b nh t h -> b t (nh h)")
        return self.proj(out)


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

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(N_PIECES, config.n_embed)
        self.position_embedding = nn.Embedding(N_SQUARES, config.n_embed)
        self.meta_linear = nn.Linear(N_META, config.n_embed)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.RMSNorm(config.n_embed).to(dtype=torch.bfloat16)
        self.lm_head = nn.Linear(config.n_embed * N_SQUARES, config.num_moves)

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
        x = rearrange(x, "b t n -> b (t n)")
        return self.lm_head(x)
