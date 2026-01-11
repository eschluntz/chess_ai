"""
Data loading from precomputed compact numpy files.
"""

import warnings
from pathlib import Path

import numpy as np
import torch

from board_repr import BoardState, MoveLabel

# Suppress warning about non-writable tensors from mmap
# Safe because we only read, then copy to GPU
warnings.filterwarnings('ignore', message='.*not writable.*')

CACHE_DIR = Path(__file__).parent / "cache" / "compact"


class CompactDataset:
    """Iterator that yields (BoardState, MoveLabel) batches from memory-mapped numpy arrays.

    Reads contiguous chunks for efficiency. No shuffling (data is pre-shuffled).
    """

    def __init__(self, data_dir: Path, prefix: str, batch_size: int):
        self.batch_size = batch_size

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
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size

            state = BoardState(
                boards=torch.as_tensor(self.boards[start:end]),
                turn=torch.as_tensor(self.turn[start:end]),
                castling=torch.as_tensor(self.castling[start:end]),
                en_passant=torch.as_tensor(self.en_passant[start:end]),
            )
            label = MoveLabel(
                from_sq=torch.as_tensor(self.from_sq[start:end]),
                to_sq=torch.as_tensor(self.to_sq[start:end]),
                promotion=torch.as_tensor(self.promotion[start:end]),
            )
            yield state, label

    def __len__(self):
        return self.num_batches


def get_dataloaders(batch_size: int, num_samples: str = "50M"):
    """Load precomputed compact features from numpy files.

    Args:
        batch_size: Number of samples per batch
        num_samples: Dataset size folder name (e.g., "1M", "50M", "784M")

    Returns:
        train_loader: Iterator yielding (BoardState, MoveLabel) batches
        eval_loader: Iterator yielding (BoardState, MoveLabel) batches
    """
    data_dir = CACHE_DIR / num_samples

    train_loader = CompactDataset(data_dir, "train", batch_size)
    eval_loader = CompactDataset(data_dir, "eval", batch_size)

    print(f"Train: {train_loader.num_samples:,} samples ({train_loader.num_batches:,} batches)")
    print(f"Eval: {eval_loader.num_samples:,} samples ({eval_loader.num_batches:,} batches)")

    return train_loader, eval_loader
