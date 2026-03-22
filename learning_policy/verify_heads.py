"""
Correctness verification for the 6 output heads before launching the sweep.

Tests:
  1. Gather correctness: synthetic input with one-hot square → only moves FROM that
     square get nonzero logits (spatial heads). Bilinear: only moves touching that
     square get nonzero logits.
  2. RMSNorm equivalence: RMSNorm([C,8,8]) vs RMSNorm(C*64) produce identical values.
  3. Logit magnitude: all heads produce roughly comparable logit std at init.
  4. No duplicate gather indices (each vocab slot maps to a unique position).
"""

import numpy as np
import torch
import torch.nn as nn

from cnn_model import (
    FlattenHead, ReduceHead, SpatialSharedHead, SpatialUnsharedHead,
    BilinearHead, FactoredHead, _spatial_gather, N_PLANES, N_SQ,
)

torch.manual_seed(0)

C = 128
vocab_np = np.load("cache/planes/vocab.npy")
vocab = torch.as_tensor(vocab_np, dtype=torch.long)
M = len(vocab)
print(f"vocab: {M} moves, from range {vocab[:,0].min()}-{vocab[:,0].max()}, "
      f"to range {vocab[:,1].min()}-{vocab[:,1].max()}, promo range {vocab[:,2].min()}-{vocab[:,2].max()}")

# =============================================================================
# Test 1: gather correctness via one-hot square injection
# =============================================================================
print("\n=== Test 1: gather correctness ===")

def onehot_square(sq: int) -> torch.Tensor:
    """(1, C, 8, 8) with ones only at (rank, file) = (sq//8, sq%8)."""
    x = torch.zeros(1, C, 8, 8)
    x[0, :, sq // 8, sq % 8] = 1.0
    return x

# --- SpatialSharedHead: only moves FROM the hot square should be nonzero ---
head = SpatialSharedHead(C, vocab)
# Zero the bias so conv(0) = 0 exactly
head.conv.bias.data.zero_()
head.norm.weight.data.fill_(1.0)

for test_sq in [0, 12, 27, 63]:  # a1, e2, d4, h8
    x = onehot_square(test_sq)
    with torch.no_grad():
        logits = head(x)[0]  # (M,)
    nonzero = (logits.abs() > 1e-6)
    nz_from = set(vocab[nonzero, 0].tolist())
    expected_count = (vocab[:, 0] == test_sq).sum().item()
    assert nz_from == {test_sq}, f"spatial_shared sq={test_sq}: nonzero from-squares = {nz_from}"
    assert nonzero.sum().item() == expected_count, \
        f"spatial_shared sq={test_sq}: {nonzero.sum()} nonzero, expected {expected_count}"
print(f"  spatial_shared: one-hot at sq → only moves FROM sq nonzero ✓")

# --- SpatialUnsharedHead: same semantics ---
head = SpatialUnsharedHead(C, vocab)
head.bias.data.zero_()
head.norm.weight.data.fill_(1.0)

for test_sq in [0, 27, 63]:
    x = onehot_square(test_sq)
    with torch.no_grad():
        logits = head(x)[0]
    nonzero = (logits.abs() > 1e-6)
    nz_from = set(vocab[nonzero, 0].tolist())
    expected_count = (vocab[:, 0] == test_sq).sum().item()
    assert nz_from == {test_sq}, f"spatial_unshared sq={test_sq}: nonzero from-squares = {nz_from}"
    assert nonzero.sum().item() == expected_count
print(f"  spatial_unshared: one-hot at sq → only moves FROM sq nonzero ✓")

# --- BilinearHead: only moves touching the hot square (as from OR to) should be nonzero ---
head = BilinearHead(C, vocab, D=64)
head.from_proj.bias.data.zero_()
head.to_proj.bias.data.zero_()
head.norm.weight.data.fill_(1.0)

for test_sq in [0, 27, 63]:
    x = onehot_square(test_sq)
    with torch.no_grad():
        logits = head(x)[0]
    nonzero = (logits.abs() > 1e-6)
    # bilinear: f[from]·t[to]. If only test_sq is nonzero, score != 0 iff from==test_sq AND to==test_sq.
    # But no legal move has from==to, so expect NO nonzero logits.
    nz_count = nonzero.sum().item()
    assert nz_count == 0, f"bilinear sq={test_sq}: expected 0 nonzero (no from==to moves), got {nz_count}"
print(f"  bilinear: one-hot at single sq → 0 nonzero (correct, f[sq]·t[sq'] needs both) ✓")

# Bilinear: two-hot (from_sq, to_sq) → that specific move should be nonzero
# Pick a known move from vocab
move_idx = 100
f_sq, t_sq, promo = vocab[move_idx].tolist()
x = torch.zeros(1, C, 8, 8)
x[0, :, f_sq // 8, f_sq % 8] = 1.0
x[0, :, t_sq // 8, t_sq % 8] = 1.0
with torch.no_grad():
    logits = head(x)[0]
assert logits[move_idx].abs() > 1e-6, f"bilinear: two-hot ({f_sq},{t_sq}) → move {move_idx} should be nonzero"
# All moves with (from, to) ∈ {f_sq, t_sq} × {f_sq, t_sq} should be nonzero
touching = ((vocab[:, 0] == f_sq) | (vocab[:, 0] == t_sq)) & \
           ((vocab[:, 1] == f_sq) | (vocab[:, 1] == t_sq))
nonzero = (logits.abs() > 1e-6)
assert nonzero.equal(touching), f"bilinear two-hot: nonzero pattern mismatch"
print(f"  bilinear: two-hot (f,t) → exactly moves with endpoints in {{f,t}} nonzero ✓")

# --- Promotion additive-decomposition: verify promo moves = base + correction ---
# For a promo move m with (f, t, p>0), logit should be out[f, t] + out[f, 64+p-1].
# For the non-promo move with the same (f, t), logit should be just out[f, t].
# So: logit(promo) - logit(nonpromo) should equal the correction plane value.
# We test by finding vocab pairs (f, t, 0) and (f, t, Q) and checking the difference.
head = SpatialSharedHead(C, vocab)
head.conv.bias.data.zero_()
head.norm.weight.data.fill_(1.0)
x = torch.randn(1, C, 8, 8)
with torch.no_grad():
    logits = head(x)[0]
    # Manually compute planes output
    raw = head.conv(head.norm(x))  # (1, 68, 8, 8)
    raw = raw.flatten(2).transpose(1, 2)[0]  # (64, 68)

# Find a (f, t) pair that exists both with promo=0 and promo=4 (Q)
for i, (f, t, p) in enumerate(vocab.tolist()):
    if p != 4:
        continue
    # look for (f, t, 0)
    match = ((vocab[:, 0] == f) & (vocab[:, 1] == t) & (vocab[:, 2] == 0)).nonzero()
    if len(match) > 0:
        j = match[0].item()
        # logit[i] - logit[j] should equal raw[f, 64 + 4 - 1] = raw[f, 67]
        diff = (logits[i] - logits[j]).item()
        expected = raw[f, 67].item()
        assert abs(diff - expected) < 1e-4, f"promo decomp: diff={diff}, expected={expected}"
        # also verify base term
        assert abs(logits[j].item() - raw[f, t].item()) < 1e-4
        print(f"  promo additive decomposition verified (move {j}↔{i}: f={f}, t={t}) ✓")
        break

# --- FactoredHead: uses global flatten, so one-hot leaks everywhere. Skip spatial test. ---

# =============================================================================
# Test 2: RMSNorm equivalence (spatial [C,8,8] vs flat C*64)
# =============================================================================
print("\n=== Test 2: RMSNorm([C,8,8]) == RMSNorm(C*64) after flatten ===")
norm_spatial = nn.RMSNorm([C, 8, 8])
norm_flat = nn.RMSNorm(C * 64)
x = torch.randn(4, C, 8, 8)
out_spatial = norm_spatial(x).flatten(1)
out_flat = norm_flat(x.flatten(1))
max_diff = (out_spatial - out_flat).abs().max().item()
assert max_diff < 1e-5, f"RMSNorm mismatch: max diff = {max_diff}"
print(f"  max elementwise diff: {max_diff:.2e} ✓")

# =============================================================================
# Test 3: logit magnitude at init
# =============================================================================
print("\n=== Test 3: logit std at init (random input, post-norm std ≈ 1) ===")
x = torch.randn(32, C, 8, 8)
heads = {
    "flatten": FlattenHead(C, M),
    "reduce": ReduceHead(C, 32, M),
    "spatial_shared": SpatialSharedHead(C, vocab),
    "spatial_unshared": SpatialUnsharedHead(C, vocab),
    "bilinear": BilinearHead(C, vocab, D=64),
    "factored": FactoredHead(C, vocab),
}
for name, h in heads.items():
    with torch.no_grad():
        logits = h(x)
    std = logits.std().item()
    mean = logits.mean().item()
    print(f"  {name:18s}  std={std:7.3f}  mean={mean:7.3f}")

# =============================================================================
# Test 4: gather index sanity (68-plane additive scheme)
# =============================================================================
print("\n=== Test 4: gather index sanity ===")
base_idx, promo_idx, promo_mask = _spatial_gather(vocab)
# base_idx must be unique across non-promo moves (those are distinguished only by base)
nonpromo = (vocab[:, 2] == 0)
n_unique_base = len(torch.unique(base_idx[nonpromo]))
n_nonpromo = nonpromo.sum().item()
assert n_unique_base == n_nonpromo, f"base_idx collision among non-promo: {n_unique_base}/{n_nonpromo}"
# ranges
assert base_idx.min() >= 0 and base_idx.max() < N_SQ * N_PLANES
assert promo_idx.min() >= 0 and promo_idx.max() < N_SQ * N_PLANES
# promo_mask should be 1 for exactly 176 moves (44 each of N,B,R,Q)
n_promo = int(promo_mask.sum().item())
assert n_promo == 176, f"expected 176 promo moves, got {n_promo}"
# promo_idx for promo moves should land in planes 64–67
promo_plane = promo_idx[promo_mask > 0] % N_PLANES
assert (promo_plane >= N_SQ).all() and (promo_plane < N_PLANES).all(), \
    f"promo_idx lands in wrong plane range: {promo_plane.min()}-{promo_plane.max()}"
print(f"  {n_nonpromo} non-promo base_idx unique, {n_promo} promo moves masked, "
      f"promo planes in [{promo_plane.min()},{promo_plane.max()}] ⊂ [64,68) ✓")

# =============================================================================
# Test 5: flip_board consistency
# =============================================================================
# A black-to-move position with a piece on sq S gets rank-flipped so the piece
# appears at flip_sq(S). The target move (S→T) is remapped via flip_table to the
# vocab index of (flip_sq(S)→flip_sq(T)). The spatial head, gathering from the
# ORIGINAL vocab coords of that remapped index, reads features at flip_sq(S) —
# exactly where the piece now sits after the plane flip. Verify this alignment.
print("\n=== Test 5: flip_board consistency ===")
from data import MoveVocab
from pathlib import Path

mv = MoveVocab(Path("cache/planes/vocab.npy"))
flip_table = mv.flip_table  # (M,) int32

def flip_sq(sq): return (7 - sq // 8) * 8 + sq % 8

# For every move j, flip_table[j] points to the move with flipped coords.
# So vocab[flip_table[j]] == (flip_sq(from_j), flip_sq(to_j), promo_j).
for j in range(M):
    f, t, p = vocab[j].tolist()
    i = flip_table[j]
    fi, ti, pi = vocab[i].tolist()
    assert fi == flip_sq(f) and ti == flip_sq(t) and pi == p, \
        f"flip_table mismatch at j={j}: ({f},{t},{p}) → idx {i} = ({fi},{ti},{pi})"
print(f"  flip_table correctly remaps all {M} moves to flipped coords ✓")

# Now verify that the spatial head's gather of the remapped index reads from
# the flipped source square. Use a head with zeroed bias.
head = SpatialSharedHead(C, vocab)
head.conv.bias.data.zero_()
head.norm.weight.data.fill_(1.0)

# Pick a black move: e.g. a knight jump from b8 (sq=57) to c6 (sq=42).
# After flip: from b1 (sq=1) to c3 (sq=18). Find the vocab index of (57, 42, 0).
orig_from, orig_to = 57, 42
orig_j = ((vocab[:, 0] == orig_from) & (vocab[:, 1] == orig_to) & (vocab[:, 2] == 0)).nonzero()[0].item()
remapped_i = int(flip_table[orig_j])
fi, ti, _ = vocab[remapped_i].tolist()
assert fi == flip_sq(orig_from) and ti == flip_sq(orig_to)

# Simulate the plane flip: put features at the ORIGINAL square (57), then flip ranks.
x = torch.zeros(1, C, 8, 8)
x[0, :, orig_from // 8, orig_from % 8] = 1.0  # piece at b8
x_flipped = x.flip(dims=[-2])                  # now at b1 (sq=1)
with torch.no_grad():
    logits = head(x_flipped)[0]
# The remapped target logit should be nonzero (it reads from sq=1, where the piece is)
assert logits[remapped_i].abs() > 1e-6, "remapped target logit is zero after flip"
# And ALL nonzero logits should have from_sq == fi (== flip_sq(57) == 1)
nonzero = (logits.abs() > 1e-6)
nz_from = set(vocab[nonzero, 0].tolist())
assert nz_from == {fi}, f"expected nonzero only from sq={fi}, got {nz_from}"
print(f"  flip alignment: piece@{orig_from} flipped→{fi}, remapped target reads from {fi} ✓")

print("\nAll verification tests passed.")
