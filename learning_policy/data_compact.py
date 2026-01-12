"""
Data loading for compact (int8 board) format - LEGACY.

This format stores boards as (N, 8, 8) int8 arrays where each value encodes the
piece type. It requires GPU-side expansion to 12 binary planes during training,
which adds ~40% overhead to the forward pass.

The planes format (data.py) pre-expands to 12 planes during precompute, trading
disk space for faster training. Use that format for new experiments.

This file is kept for backwards compatibility with existing compact datasets.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import torch

from board_repr_compact import BoardState
from data import MoveVocab

# Suppress warning about non-writable tensors from mmap
warnings.filterwarnings('ignore', message='.*not writable.*')

CACHE_DIR = Path(__file__).parent / "cache" / "compact"


class CompactDataset:
    """Iterator that yields (BoardState, target_indices) batches from memory-mapped numpy arrays.

    Reads contiguous chunks for efficiency. No shuffling (data is pre-shuffled).
    """

    def __init__(self, data_dir: Path, prefix: str, batch_size: int, vocab: MoveVocab):
        self.batch_size = batch_size
        self.vocab = vocab

        # Memory-map all arrays
        self.boards = np.load(data_dir / f"{prefix}_boards.npy", mmap_mode='r')
        self.turn = np.load(data_dir / f"{prefix}_turn.npy", mmap_mode='r')
        self.castling = np.load(data_dir / f"{prefix}_castling.npy", mmap_mode='r')
        self.en_passant = np.load(data_dir / f"{prefix}_en_passant.npy", mmap_mode='r')
        self.from_sq = np.load(data_dir / f"{prefix}_from_sq.npy", mmap_mode='r')
        self.to_sq = np.load(data_dir / f"{prefix}_to_sq.npy", mmap_mode='r')
        self.promotion = np.load(data_dir / f"{prefix}_promotion.npy", mmap_mode='r')

        self.num_samples = len(self.boards)
        self.num_batches = self.num_samples // batch_size

    def __iter__(self):
        # Timing accumulators (reset each epoch)
        self.time_mmap = 0.0
        self.time_tensor = 0.0
        self.time_vocab = 0.0

        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size

            # Time mmap reads
            t0 = time.perf_counter()
            boards = self.boards[start:end]
            turn = self.turn[start:end]
            castling = self.castling[start:end]
            en_passant = self.en_passant[start:end]
            from_sq = self.from_sq[start:end]
            to_sq = self.to_sq[start:end]
            promotion = self.promotion[start:end]
            self.time_mmap += time.perf_counter() - t0

            # Time tensor creation
            t0 = time.perf_counter()
            state = BoardState(
                boards=torch.as_tensor(boards),
                turn=torch.as_tensor(turn),
                castling=torch.as_tensor(castling),
                en_passant=torch.as_tensor(en_passant),
            )
            from_t = torch.as_tensor(from_sq)
            to_t = torch.as_tensor(to_sq)
            promo_t = torch.as_tensor(promotion)
            self.time_tensor += time.perf_counter() - t0

            # Time vocab lookup
            t0 = time.perf_counter()
            target = self.vocab.to_indices(from_t, to_t, promo_t)
            self.time_vocab += time.perf_counter() - t0

            yield state, target

    def __len__(self):
        return self.num_batches


def get_compact_dataloaders(batch_size: int, num_samples: str = "50M"):
    """Load precomputed compact features from numpy files.

    Args:
        batch_size: Number of samples per batch
        num_samples: Dataset size folder name (e.g., "1M", "50M", "full")

    Returns:
        train_loader: Iterator yielding (BoardState, target_indices) batches
        eval_loader: Iterator yielding (BoardState, target_indices) batches
        vocab: MoveVocab for decoding predictions
    """
    data_dir = CACHE_DIR / num_samples
    vocab = MoveVocab(CACHE_DIR / "vocab.npy")

    train_loader = CompactDataset(data_dir, "train", batch_size, vocab)
    eval_loader = CompactDataset(data_dir, "eval", batch_size, vocab)

    print(f"Train: {train_loader.num_samples:,} samples ({train_loader.num_batches:,} batches)")
    print(f"Eval: {eval_loader.num_samples:,} samples ({eval_loader.num_batches:,} batches)")
    print(f"Vocab: {vocab.num_moves:,} unique moves")

    return train_loader, eval_loader, vocab
