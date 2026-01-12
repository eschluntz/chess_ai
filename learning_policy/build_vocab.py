"""
Build move vocabulary from precomputed data.

The vocab maps unique (from_sq, to_sq, promotion) tuples to indices.
This is an optimization layer - the canonical representation remains
(from_sq, to_sq, promotion), and vocab just reduces the output space
from 20,480 (64*64*5) to ~1,968 actually-occurring moves.

Usage:
    python build_vocab.py                           # uses cache/compact/50M
    python build_vocab.py cache/compact/1M          # specify data dir
    python build_vocab.py dir1 dir2                 # combine multiple dirs
"""

from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent / "cache" / "compact"


def build_vocab(data_dirs: list[str] = None):
    """Build vocab from one or more data directories. Saves to cache/compact/vocab.npy."""
    if not data_dirs:
        data_dirs = [str(CACHE_DIR / "50M")]

    all_moves = []

    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        print(f"Loading moves from {data_dir}...")

        for prefix in ["train", "eval"]:
            from_sq = np.load(data_dir / f"{prefix}_from_sq.npy", mmap_mode='r')
            to_sq = np.load(data_dir / f"{prefix}_to_sq.npy", mmap_mode='r')
            promotion = np.load(data_dir / f"{prefix}_promotion.npy", mmap_mode='r')

            moves = np.stack([from_sq, to_sq, promotion], axis=1)
            all_moves.append(moves)
            print(f"  {prefix}: {len(moves):,} samples")

    # Combine and dedupe
    all_moves = np.concatenate(all_moves, axis=0)
    unique_moves = np.unique(all_moves, axis=0)  # Sorted by default

    print(f"\nTotal samples: {len(all_moves):,}")
    print(f"Unique moves: {len(unique_moves):,}")

    # Save to shared location
    vocab_path = CACHE_DIR / "vocab.npy"
    np.save(vocab_path, unique_moves)
    print(f"Saved {vocab_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(build_vocab)
