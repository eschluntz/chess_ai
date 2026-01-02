"""
Data loading from precomputed numpy files.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import torch

# Suppress warning about non-writable tensors from mmap
# Safe because we only read, then copy to GPU
warnings.filterwarnings('ignore', message='.*not writable.*')

CACHE_DIR = Path(__file__).parent / "cache"


class BatchedDataset:
    """Iterator that yields batches from memory-mapped numpy arrays.

    Reads contiguous chunks for efficiency. No shuffling (data is pre-shuffled).
    """

    def __init__(self, features_path: Path, labels_path: Path, batch_size: int):
        self.features = np.load(features_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        self.batch_size = batch_size
        self.num_samples = len(self.labels)
        self.num_batches = self.num_samples // batch_size

    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            # Zero-copy view of mmap - safe since we only read, then copy to GPU
            X = torch.as_tensor(self.features[start:end])
            y = torch.as_tensor(self.labels[start:end])
            yield X, y

    def __len__(self):
        return self.num_batches


def get_dataloaders(batch_size: int, cache_dir: str = None, num_workers: int = 0):
    """Load precomputed features from numpy files.

    Returns batch iterators (not DataLoaders) for efficiency.
    num_workers is ignored (mmap doesn't benefit from multiprocessing).
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    precomputed_dir = Path(cache_dir) / "precomputed"

    train_loader = BatchedDataset(
        precomputed_dir / "train_features.npy",
        precomputed_dir / "train_labels.npy",
        batch_size,
    )
    eval_loader = BatchedDataset(
        precomputed_dir / "eval_features.npy",
        precomputed_dir / "eval_labels.npy",
        batch_size,
    )

    with open(precomputed_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    print(f"Train: {train_loader.num_samples:,} samples ({train_loader.num_batches:,} batches)")
    print(f"Eval: {eval_loader.num_samples:,} samples ({eval_loader.num_batches:,} batches)")

    return train_loader, eval_loader, vocab
