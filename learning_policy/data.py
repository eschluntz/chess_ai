"""
Data loading for precomputed planes format with soft labels.

Storage format:
    planes.npy:        (N, 13, 8, 8) uint8 - 12 piece planes + en_passant
    meta.npy:          (N, 5) int8 - [turn, K, Q, k, q]
    label_indices.npy: (total_nonzero,) uint16 - move vocab indices
    label_probs.npy:   (total_nonzero,) float16 - probabilities
    label_offsets.npy: (N + 1,) uint32 - position boundaries
"""

import time
import warnings
from pathlib import Path

import numpy as np
import torch

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


class SoftLabelDataset:
    """Iterator that yields (planes, meta, soft_target) batches with sparse->dense conversion."""

    def __init__(self, data_dir: Path, prefix: str, batch_size: int, num_classes: int):
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Memory-map position arrays
        self.planes = np.load(data_dir / f"{prefix}_planes.npy", mmap_mode='r')
        self.meta = np.load(data_dir / f"{prefix}_meta.npy", mmap_mode='r')

        # Load sparse labels
        self.label_indices = torch.from_numpy(
            np.load(data_dir / f"{prefix}_label_indices.npy").astype(np.int64)
        )
        self.label_probs = torch.from_numpy(
            np.load(data_dir / f"{prefix}_label_probs.npy").astype(np.float32)
        )
        self.label_offsets = np.load(data_dir / f"{prefix}_label_offsets.npy")

        self.num_samples = len(self.planes)
        self.num_batches = self.num_samples // batch_size

    def __iter__(self):
        self.time_mmap = 0.0
        self.time_tensor = 0.0
        self.time_labels = 0.0

        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size

            t0 = time.perf_counter()
            planes = self.planes[start:end]
            meta = self.meta[start:end]
            self.time_mmap += time.perf_counter() - t0

            t0 = time.perf_counter()
            planes_t = torch.as_tensor(planes)
            meta_t = torch.as_tensor(meta)
            self.time_tensor += time.perf_counter() - t0

            # Vectorized soft target construction
            t0 = time.perf_counter()

            label_start = self.label_offsets[start]
            label_end = self.label_offsets[end]

            batch_indices = self.label_indices[label_start:label_end]
            batch_probs = self.label_probs[label_start:label_end]

            # Compute row indices for scatter
            local_offsets = self.label_offsets[start:end+1] - label_start
            row_indices = torch.zeros(len(batch_indices), dtype=torch.long)
            for j in range(self.batch_size):
                row_indices[local_offsets[j]:local_offsets[j+1]] = j

            # Scatter into dense tensor
            soft_target = torch.zeros(self.batch_size, self.num_classes)
            soft_target[row_indices, batch_indices] = batch_probs

            self.time_labels += time.perf_counter() - t0

            yield planes_t, meta_t, soft_target

    def __len__(self):
        return self.num_batches


def get_dataloaders(batch_size: int, num_samples: str = "50M"):
    """Load precomputed data with soft labels.

    Returns:
        train_loader: Iterator yielding (planes, meta, soft_target) batches
        eval_loader: Iterator yielding (planes, meta, soft_target) batches
        num_classes: Number of move classes (vocab size)
    """
    data_dir = CACHE_DIR / num_samples

    # Get num_classes from vocab
    vocab_path = CACHE_DIR / "vocab.npy"
    vocab = np.load(vocab_path)
    num_classes = len(vocab)

    # Load stats if available
    stats_path = data_dir / "stats.npy"
    if stats_path.exists():
        stats = np.load(stats_path, allow_pickle=True).item()
        print(f"Loaded: {stats['unique_positions']:,} unique positions")
        print(f"  (from {stats['raw_samples']:,} raw samples, {stats['dedup_ratio']:.2f}x dedup)")
        print(f"  Avg moves per position: {stats['avg_moves_per_position']:.2f}")

    train_loader = SoftLabelDataset(data_dir, "train", batch_size, num_classes)
    eval_loader = SoftLabelDataset(data_dir, "eval", batch_size, num_classes)

    print(f"Train: {train_loader.num_samples:,} positions ({train_loader.num_batches:,} batches)")
    print(f"Eval: {eval_loader.num_samples:,} positions ({eval_loader.num_batches:,} batches)")
    print(f"Vocab: {num_classes:,} unique moves")

    return train_loader, eval_loader, num_classes
