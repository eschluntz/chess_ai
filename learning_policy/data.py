"""
Data loading using HuggingFace datasets with memory-mapped Arrow storage.
No manual sharding needed - Arrow handles it automatically.
"""

from functools import partial
from pathlib import Path

import chess
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from features import extract_features_piece_square

CACHE_DIR = Path(__file__).parent / "cache"
EVAL_SIZE = 10_000


def get_all_promotions() -> set[str]:
    """Generate all possible promotion moves in UCI format."""
    promotions = set()
    files = "abcdefgh"
    promo_pieces = "qrbn"

    for from_rank, to_rank in [("7", "8"), ("2", "1")]:  # White and black promotions
        for f_idx, from_file in enumerate(files):
            from_sq = f"{from_file}{from_rank}"
            # Forward and diagonal captures
            for to_f_idx in [f_idx - 1, f_idx, f_idx + 1]:
                if 0 <= to_f_idx <= 7:
                    to_sq = f"{files[to_f_idx]}{to_rank}"
                    promotions.update(f"{from_sq}{to_sq}{p}" for p in promo_pieces)

    return promotions


def build_move_vocabulary(ds) -> dict[str, int]:
    """Build vocabulary from dataset + all promotions."""
    print("Building vocabulary...")
    moves = set()

    # Sample in batches to avoid loading all at once
    sample_size = min(1_000_000, len(ds))
    for i in range(0, sample_size, 10_000):
        batch = ds.select(range(i, min(i + 10_000, sample_size)))
        moves.update(line.split()[0] for line in batch["line"])

    moves.update(get_all_promotions())

    vocab = {move: idx for idx, move in enumerate(sorted(moves))}
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def load_full_dataset():
    """Load full dataset with Arrow memory-mapping (doesn't load into RAM)."""
    print("Loading dataset (memory-mapped)...")
    ds = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    print(f"Dataset size: {len(ds):,} samples")
    return ds


def collate_fn(batch, vocab, device):
    """Convert batch of examples to feature tensors."""
    features = []
    labels = []
    for example in batch:
        board = chess.Board(example["fen"])
        features.append(extract_features_piece_square(board))
        target_move = example["line"].split()[0]
        labels.append(vocab[target_move])

    X = torch.tensor(np.array(features), dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return X, y


def get_dataloaders(batch_size: int, device: torch.device, on_checkpoint=None):
    """Load dataset, build vocab, return dataloaders and vocab."""
    ds = load_full_dataset()
    if on_checkpoint:
        print("Committing after dataset load...")
        on_checkpoint()

    vocab = build_move_vocabulary(ds)
    if on_checkpoint:
        print("Committing after vocab build...")
        on_checkpoint()

    # Shuffle first (avoids bias if original data is ordered), then split
    # Note: just creates index mapping, no disk write. Reads will be random access.
    ds = ds.shuffle(seed=42)

    eval_ds = ds.select(range(EVAL_SIZE))
    train_ds = ds.select(range(EVAL_SIZE, len(ds)))

    print(f"Train: {len(train_ds):,}, Eval: {len(eval_ds):,}")

    collate = partial(collate_fn, vocab=vocab, device=device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # Already shuffled above
        collate_fn=collate,
        num_workers=0,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    return train_loader, eval_loader, vocab
