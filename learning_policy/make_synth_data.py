"""
Generate a tiny synthetic dataset for local training validation.
Shapes/dtypes match the real precomputed format; content is random.
"""

import numpy as np
from pathlib import Path

N = 10_000
OUT = Path(__file__).parent / "cache" / "planes" / "synth"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed=42)

# Planes: at most one piece per square (pick a random plane 0-12, or empty)
planes = np.zeros((N, 13, 8, 8), dtype=np.uint8)
piece = rng.integers(0, 14, size=(N, 8, 8))  # 0-12 piece, 13 empty
for p in range(13):
    planes[:, p, :, :] = (piece == p).astype(np.uint8)

# Meta: [turn, K, Q, k, q]
meta = np.zeros((N, 5), dtype=np.int8)
meta[:, 0] = rng.choice([-1, 1], size=N)  # turn
meta[:, 1:5] = rng.integers(0, 2, size=(N, 4))  # castling

# Analyses: one per position. [move_idx, depth, knodes, cp, is_mate]
vocab = np.load(OUT.parent / "vocab.npy")
num_moves = len(vocab)
analysis_data = np.zeros((N, 5), dtype=np.int32)
analysis_data[:, 0] = rng.integers(0, num_moves, size=N)  # move_idx
analysis_data[:, 1] = rng.integers(20, 60, size=N)  # depth
analysis_data[:, 2] = rng.integers(100, 10000, size=N)  # knodes
analysis_data[:, 3] = rng.integers(-300, 300, size=N)  # cp
analysis_data[:, 4] = 0  # is_mate

# Offsets: position i's analyses are at [offsets[i], offsets[i+1])
analysis_offsets = np.arange(N + 1, dtype=np.uint32)

np.save(OUT / "train_planes.npy", planes)
np.save(OUT / "train_meta.npy", meta)
np.save(OUT / "train_analysis_data.npy", analysis_data)
np.save(OUT / "train_analysis_offsets.npy", analysis_offsets)

print(f"Wrote {N:,} synthetic positions to {OUT}")
print(f"  planes: {planes.shape} {planes.dtype}")
print(f"  meta: {meta.shape} {meta.dtype}")
print(f"  analysis_data: {analysis_data.shape} {analysis_data.dtype}")
print(f"  analysis_offsets: {analysis_offsets.shape} {analysis_offsets.dtype}")
print(f"  vocab: {num_moves} moves")
