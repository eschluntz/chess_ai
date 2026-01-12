"""
Data loading from precomputed compact numpy files.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import torch

from board_repr import BoardState

# Suppress warning about non-writable tensors from mmap
# Safe because we only read, then copy to GPU
warnings.filterwarnings('ignore', message='.*not writable.*')

CACHE_DIR = Path(__file__).parent / "cache" / "compact"


class MoveVocab:
    """Bidirectional mapping between (from_sq, to_sq, promotion) and vocab indices."""

    def __init__(self, vocab_path: Path):
        # vocab.npy is (num_moves, 3) array of [from_sq, to_sq, promotion]
        self.moves = np.load(vocab_path)  # (num_moves, 3)
        self.num_moves = len(self.moves)

        # Build tensor lookup table: packed_key -> index
        # Pack: from*320 + to*5 + promo (max = 63*320 + 63*5 + 4 = 20479)
        self._lookup = torch.zeros(64 * 320, dtype=torch.long)
        for idx, (f, t, p) in enumerate(self.moves):
            key = int(f) * 320 + int(t) * 5 + int(p)
            self._lookup[key] = idx

    def to_indices(self, from_sq: torch.Tensor, to_sq: torch.Tensor, promotion: torch.Tensor) -> torch.Tensor:
        """Convert move components to vocab indices. Vectorized tensor operation."""
        keys = from_sq.long() * 320 + to_sq.long() * 5 + promotion.long()
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


def get_dataloaders(batch_size: int, num_samples: str = "50M"):
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
