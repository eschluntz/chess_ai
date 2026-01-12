"""
Data loading from precomputed planes numpy files.

Uses consolidated 13-plane format for fast training:
    planes.npy: (N, 13, 8, 8) uint8 - 12 piece planes + en_passant
    meta.npy:   (N, 5) int8         - [turn, K, Q, k, q]
    moves.npy:  (N, 3) uint8        - [from_sq, to_sq, promotion]

For the legacy compact (int8 board) format, see data_compact.py.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import torch

# Suppress warning about non-writable tensors from mmap
warnings.filterwarnings('ignore', message='.*not writable.*')

CACHE_DIR = Path(__file__).parent / "cache" / "planes"


class MoveVocab:
    """Bidirectional mapping between (from_sq, to_sq, promotion) and vocab indices."""

    def __init__(self, vocab_path: Path):
        self.moves = np.load(vocab_path)  # (num_moves, 3)
        self.num_moves = len(self.moves)

        # Build tensor lookup table: packed_key -> index
        self._lookup = torch.zeros(64 * 320, dtype=torch.long)
        for idx, (f, t, p) in enumerate(self.moves):
            key = int(f) * 320 + int(t) * 5 + int(p)
            self._lookup[key] = idx

    def to_indices(self, moves: torch.Tensor) -> torch.Tensor:
        """Convert (batch, 3) moves to vocab indices. Vectorized tensor operation."""
        keys = moves[:, 0].long() * 320 + moves[:, 1].long() * 5 + moves[:, 2].long()
        return self._lookup[keys]

    def to_uci(self, idx: int) -> str:
        """Convert vocab index to UCI string."""
        f, t, p = self.moves[idx]
        from_file, from_rank = f % 8, f // 8
        to_file, to_rank = t % 8, t // 8
        uci = chr(ord('a') + from_file) + str(from_rank + 1)
        uci += chr(ord('a') + to_file) + str(to_rank + 1)
        if p > 0:
            uci += ['', 'n', 'b', 'r', 'q'][p]
        return uci


class PlanesDataset:
    """Iterator that yields (planes, meta, target) batches from memory-mapped numpy arrays."""

    def __init__(self, data_dir: Path, prefix: str, batch_size: int, vocab: MoveVocab):
        self.batch_size = batch_size
        self.vocab = vocab

        # Memory-map consolidated arrays
        self.planes = np.load(data_dir / f"{prefix}_planes.npy", mmap_mode='r')
        self.meta = np.load(data_dir / f"{prefix}_meta.npy", mmap_mode='r')
        self.moves = np.load(data_dir / f"{prefix}_moves.npy", mmap_mode='r')

        self.num_samples = len(self.planes)
        self.num_batches = self.num_samples // batch_size

    def __iter__(self):
        self.time_mmap = 0.0
        self.time_tensor = 0.0
        self.time_vocab = 0.0

        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size

            t0 = time.perf_counter()
            planes = self.planes[start:end]
            meta = self.meta[start:end]
            moves = self.moves[start:end]
            self.time_mmap += time.perf_counter() - t0

            t0 = time.perf_counter()
            planes_t = torch.as_tensor(planes)
            meta_t = torch.as_tensor(meta)
            moves_t = torch.as_tensor(moves)
            self.time_tensor += time.perf_counter() - t0

            t0 = time.perf_counter()
            target = self.vocab.to_indices(moves_t)
            self.time_vocab += time.perf_counter() - t0

            yield planes_t, meta_t, target

    def __len__(self):
        return self.num_batches


def get_dataloaders(batch_size: int, num_samples: str = "50M"):
    """Load precomputed planes features from numpy files.

    Returns:
        train_loader: Iterator yielding (planes, meta, target) batches
        eval_loader: Iterator yielding (planes, meta, target) batches
        vocab: MoveVocab for decoding predictions
    """
    data_dir = CACHE_DIR / num_samples

    vocab_path = CACHE_DIR / "vocab.npy"
    if not vocab_path.exists():
        vocab_path = Path(__file__).parent / "cache" / "compact" / "vocab.npy"
    vocab = MoveVocab(vocab_path)

    train_loader = PlanesDataset(data_dir, "train", batch_size, vocab)
    eval_loader = PlanesDataset(data_dir, "eval", batch_size, vocab)

    print(f"Train: {train_loader.num_samples:,} samples ({train_loader.num_batches:,} batches)")
    print(f"Eval: {eval_loader.num_samples:,} samples ({eval_loader.num_batches:,} batches)")
    print(f"Vocab: {vocab.num_moves:,} unique moves")

    return train_loader, eval_loader, vocab
