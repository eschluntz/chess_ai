"""
Tests for transformer_model.py, focused on planes_to_toks_flipped where
off-by-one errors are easy.

Board representation reference (board_repr.py):
  planes[plane, rank, file], rank 0 = chess rank 1, file 0 = a-file
  Plane order: white P,N,B,R,Q,K (0-5), black p,n,b,r,q,k (6-11), en_passant (12)
  meta: [turn, K, Q, k, q]  (turn: +1 white, -1 black)
  Square index convention: sq = rank * 8 + file
"""

import pytest
import torch

from learning_policy.transformer_model import (
    Transformer,
    TransformerConfig,
    _displacement_buckets,
)

# Token IDs after tokenization: 0=empty, 1=wP, 2=wN, 3=wB, 4=wR, 5=wQ, 6=wK,
# 7=bP, 8=bN, 9=bB, 10=bR, 11=bQ, 12=bK, 13=EP
EMPTY, wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK, EP = range(14)

PIECE_NAMES = ["wP", "wN", "wB", "wR", "wQ", "wK",
               "bP", "bN", "bB", "bR", "bQ", "bK", "EP"]


@pytest.fixture
def model():
    cfg = TransformerConfig(n_embed=32, num_heads=4, num_layers=1, num_moves=1968)
    return Transformer(cfg)


def empty_planes(batch=1):
    return torch.zeros(batch, 13, 8, 8, dtype=torch.uint8)


def white_meta(batch=1):
    return torch.tensor([[1, 1, 1, 1, 1]] * batch, dtype=torch.int8)


def black_meta(batch=1):
    return torch.tensor([[-1, 1, 1, 1, 1]] * batch, dtype=torch.int8)


def test_empty_board(model):
    """Empty board → all-zero tokens."""
    toks, _ = model.planes_to_toks_flipped(empty_planes(), white_meta())
    assert toks.shape == (1, 64)
    assert (toks == 0).all()


@pytest.mark.parametrize("plane", range(13))
def test_piece_plane_to_token_id(model, plane):
    """Each piece plane → correct token ID (plane i → token i+1)."""
    planes = empty_planes()
    planes[0, plane, 0, 0] = 1  # piece at a1
    toks, _ = model.planes_to_toks_flipped(planes, white_meta())
    assert toks[0, 0] == plane + 1, f"plane {plane} ({PIECE_NAMES[plane]})"
    assert (toks[0, 1:] == 0).all()


@pytest.mark.parametrize("rank,file", [(0, 0), (0, 7), (7, 0), (7, 7), (3, 4)])
def test_square_numbering(model, rank, file):
    """Token index == rank*8 + file (vocab convention)."""
    planes = empty_planes()
    planes[0, 0, rank, file] = 1
    toks, _ = model.planes_to_toks_flipped(planes, white_meta())
    expected_sq = rank * 8 + file
    hit = (toks[0] == wP).nonzero().item()
    assert hit == expected_sq


def test_white_to_move_is_identity(model):
    """White-to-move: no flip happens."""
    planes = empty_planes()
    planes[0, 0, 1, 4] = 1  # white pawn on e2
    planes[0, 6, 6, 4] = 1  # black pawn on e7
    meta = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.int8)
    toks, out_meta = model.planes_to_toks_flipped(planes, meta)
    assert toks[0, 1 * 8 + 4] == wP
    assert toks[0, 6 * 8 + 4] == bP
    assert out_meta.tolist() == [[1, 1, 0, 0, 1]]


def test_black_to_move_flips(model):
    """Black-to-move: color swap + rank flip + castling swap."""
    planes = empty_planes()
    planes[0, 0, 1, 4] = 1  # white pawn on e2
    planes[0, 6, 6, 4] = 1  # black pawn on e7
    meta = torch.tensor([[-1, 1, 0, 0, 1]], dtype=torch.int8)
    toks, out_meta = model.planes_to_toks_flipped(planes, meta)

    # black pawn (plane 6) → "my pawn" (token wP), rank 6 → rank 1
    assert toks[0, 1 * 8 + 4] == wP
    # white pawn (plane 0) → "their pawn" (token bP), rank 1 → rank 6
    assert toks[0, 6 * 8 + 4] == bP
    # everything else empty
    mask = torch.ones(64, dtype=torch.bool)
    mask[12] = mask[52] = False
    assert (toks[0][mask] == 0).all()
    # castling: [turn, K, Q, k, q]=[-1,1,0,0,1] → [turn, k, q, K, Q]=[-1,0,1,1,0]
    assert out_meta.tolist() == [[-1, 0, 1, 1, 0]]


def test_ep_target_flips_with_rank(model):
    """En-passant target flips rank when black to move."""
    planes = empty_planes()
    planes[0, 12, 5, 3] = 1  # EP target at d6
    toks_w, _ = model.planes_to_toks_flipped(planes, white_meta())
    toks_b, _ = model.planes_to_toks_flipped(planes, black_meta())
    assert toks_w[0, 5 * 8 + 3] == EP
    assert toks_b[0, 2 * 8 + 3] == EP  # rank 5 → rank 2


def test_mixed_batch(model):
    """Mixed white/black batch: each sample flipped independently."""
    planes = empty_planes(batch=2)
    planes[0, 5, 0, 4] = 1  # batch 0: white king on e1
    planes[1, 11, 7, 4] = 1  # batch 1: black king on e8
    meta = torch.tensor([[1, 1, 1, 1, 1], [-1, 1, 1, 1, 1]], dtype=torch.int8)
    toks, _ = model.planes_to_toks_flipped(planes, meta)
    # batch 0 (white): king stays at e1
    assert toks[0, 0 * 8 + 4] == wK
    # batch 1 (black): black king → "my king" at flipped rank 0
    assert toks[1, 0 * 8 + 4] == wK


def test_token_dtype(model):
    """Tokens must be long for nn.Embedding."""
    planes = empty_planes()
    planes[0, 0, 0, 0] = 1
    toks, _ = model.planes_to_toks_flipped(planes, white_meta())
    assert toks.dtype in (torch.long, torch.int64)


def _random_input(batch=4):
    planes = torch.randint(0, 2, (batch, 13, 8, 8), dtype=torch.uint8)
    # At most one piece per square (otherwise sum trick gives out-of-range IDs)
    mask = torch.zeros_like(planes)
    choice = torch.randint(0, 13, (batch, 8, 8))
    mask.scatter_(1, choice.unsqueeze(1), 1)
    planes = planes * mask
    meta = torch.randint(-1, 2, (batch, 5), dtype=torch.int8)
    meta[:, 0] = torch.tensor([1, -1] * (batch // 2))
    return planes, meta


def test_forward_smoke(model):
    """End-to-end forward pass produces correct shape and finite values."""
    planes, meta = _random_input()
    out = model(planes, meta)
    assert out.shape == (4, 1968)
    assert out.isfinite().all()


# ---------------------------------------------------------------------------
# Position encoding tests
# ---------------------------------------------------------------------------


def test_displacement_buckets_shape_and_range():
    """Bucket lookup is (64, 64) with values in [0, 225)."""
    b = _displacement_buckets()
    assert b.shape == (64, 64)
    assert b.min() >= 0 and b.max() < 225


def test_displacement_buckets_diagonal_is_center():
    """Δ=(0,0) → center bucket 7*15+7=112."""
    b = _displacement_buckets()
    assert (b.diagonal() == 112).all()


def test_displacement_buckets_translation_invariant():
    """Same displacement → same bucket regardless of absolute position.

    Knight on c3 (sq=18) → e4 (sq=28): Δ=(+1, +2)
    Knight on f6 (sq=45) → h7 (sq=55): Δ=(+1, +2)
    These must map to the same bucket.
    """
    b = _displacement_buckets()
    assert b[18, 28] == b[45, 55]
    # And the bucket is (1+7)*15 + (2+7) = 129
    assert b[18, 28] == 129


def test_displacement_buckets_antisymmetric():
    """Swapping i and j negates the displacement → different bucket (unless Δ=0)."""
    b = _displacement_buckets()
    # a1→h8: Δ=(7,7), bucket (14)*15+14 = 224
    # h8→a1: Δ=(-7,-7), bucket 0*15+0 = 0
    assert b[0, 63] == 224
    assert b[63, 0] == 0


@pytest.mark.parametrize("pos_encoding", ["absolute", "bias64", "shaw2d"])
def test_pos_encoding_forward(pos_encoding):
    """All three encoding modes produce correct output shape and are finite."""
    cfg = TransformerConfig(
        n_embed=32, num_heads=4, num_layers=2, num_moves=1968, pos_encoding=pos_encoding
    )
    m = Transformer(cfg)
    planes, meta = _random_input()
    out = m(planes, meta)
    assert out.shape == (4, 1968)
    assert out.isfinite().all()


@pytest.mark.parametrize("pos_encoding", ["absolute", "bias64", "shaw2d"])
def test_pos_encoding_backward(pos_encoding):
    """Gradients flow through all three encoding modes."""
    cfg = TransformerConfig(
        n_embed=32, num_heads=4, num_layers=2, num_moves=1968, pos_encoding=pos_encoding
    )
    m = Transformer(cfg)
    planes, meta = _random_input()
    out = m(planes, meta)
    out.sum().backward()
    # Check the encoding-specific params received gradients
    if pos_encoding == "bias64":
        assert m.blocks[0].attn.pos_bias.grad is not None
        assert m.blocks[0].attn.pos_bias.grad.abs().sum() > 0
    elif pos_encoding == "shaw2d":
        assert m.blocks[0].attn.rel_k.grad is not None
        assert m.blocks[0].attn.rel_k.grad.abs().sum() > 0


def test_bias64_param_count():
    """bias64 adds H×64×64 params per layer."""
    cfg = TransformerConfig(n_embed=256, num_heads=8, num_layers=8, pos_encoding="bias64")
    m = Transformer(cfg)
    bias_params = sum(p.numel() for n, p in m.named_parameters() if "pos_bias" in n)
    assert bias_params == 8 * 8 * 64 * 64  # 262,144


def test_shaw2d_param_count():
    """shaw2d adds 225×head_size×2 params per layer."""
    cfg = TransformerConfig(n_embed=256, num_heads=8, num_layers=8, pos_encoding="shaw2d")
    m = Transformer(cfg)
    rel_params = sum(p.numel() for n, p in m.named_parameters() if "rel_" in n)
    head_size = 256 // 8
    assert rel_params == 8 * 225 * head_size * 2  # 115,200
