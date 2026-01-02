"""
Data loading from precomputed numpy files.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CACHE_DIR = Path(__file__).parent / "cache"


class PrecomputedDataset(Dataset):
    """Dataset that reads from memory-mapped numpy arrays."""

    def __init__(self, features_path: Path, labels_path: Path):
        self.features = np.load(features_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def collate_fn(batch):
    """Convert batch to tensors."""
    features, labels = zip(*batch)
    X = torch.tensor(np.array(features), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)
    return X, y


def get_dataloaders(batch_size: int, cache_dir: str = None, num_workers: int = 0):
    """Load precomputed features from numpy files."""
    if cache_dir is None:
        cache_dir = CACHE_DIR
    precomputed_dir = Path(cache_dir) / "precomputed"

    train_ds = PrecomputedDataset(
        precomputed_dir / "train_features.npy",
        precomputed_dir / "train_labels.npy",
    )
    eval_ds = PrecomputedDataset(
        precomputed_dir / "eval_features.npy",
        precomputed_dir / "eval_labels.npy",
    )

    with open(precomputed_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    print(f"Train: {len(train_ds):,}, Eval: {len(eval_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, eval_loader, vocab
