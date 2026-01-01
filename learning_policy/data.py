"""
Data loading utilities for policy network training.

Uses the Lichess/chess-position-evaluations dataset.
Extracts the first move from Stockfish's recommended line as the target.
"""

import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Move vocabulary cache
VOCAB_CACHE: dict[str, int] | None = None

# Cache paths
CACHE_DIR = Path(__file__).parent / "cache"
SHUFFLED_CACHE = CACHE_DIR / "shuffled_10m_dataset.parquet"

# Reserve first N positions for evaluation
EVAL_SET_SIZE = 10_000


def get_raw_shuffled_data() -> pd.DataFrame:
    """
    Load the shuffled 10M position dataset.
    Uses cached parquet if available, otherwise streams from HuggingFace.
    """
    if SHUFFLED_CACHE.exists():
        print(f"Loading from cache: {SHUFFLED_CACHE}")
        return pd.read_parquet(SHUFFLED_CACHE)

    print("No cache found. Streaming from HuggingFace...")
    pool_size = 10_000_000

    ds = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    train_data = list(ds["train"].take(pool_size))

    df = pd.DataFrame(train_data)
    df = df.dropna(subset=["fen"])
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(SHUFFLED_CACHE, index=False)
    print(f"Saved cache to: {SHUFFLED_CACHE}")

    return df


def extract_first_move(line: str) -> str:
    """Extract the first move from a space-separated line of UCI moves."""
    return line.split()[0]


def get_policy_data(
    num_samples: int,
    offset: int = 0,
    skip_eval_set: bool = True,
) -> pd.DataFrame:
    """
    Get policy training data with FEN positions and target moves.

    Args:
        num_samples: Number of samples to return
        offset: Additional offset from start (added to eval set skip if applicable)
        skip_eval_set: If True, skip first EVAL_SET_SIZE samples (reserved for eval)

    Returns:
        DataFrame with columns: fen, target_move, cp, mate
    """
    raw_df = get_raw_shuffled_data()

    start_idx = offset
    if skip_eval_set:
        start_idx += EVAL_SET_SIZE

    end_idx = start_idx + num_samples
    subset = raw_df.iloc[start_idx:end_idx].copy()

    # Extract first move as target
    subset["target_move"] = subset["line"].apply(extract_first_move)

    return subset[["fen", "target_move", "cp", "mate"]]


def get_eval_data() -> pd.DataFrame:
    """Get evaluation data (first EVAL_SET_SIZE samples from shuffled dataset)."""
    raw_df = get_raw_shuffled_data()
    subset = raw_df.head(EVAL_SET_SIZE).copy()
    subset["target_move"] = subset["line"].apply(extract_first_move)
    return subset[["fen", "target_move", "cp", "mate"]]


def get_train_data(num_samples: int) -> pd.DataFrame:
    """
    Get training data (samples after eval set).
    """
    return get_policy_data(num_samples, offset=0, skip_eval_set=True)


def get_all_promotions() -> set[str]:
    """Generate all possible promotion moves in UCI format."""
    promotions = set()
    files = "abcdefgh"
    promo_pieces = "qrbn"

    # White promotions: rank 7 -> rank 8
    for f_idx, from_file in enumerate(files):
        from_sq = f"{from_file}7"
        # Forward
        promotions.update(f"{from_sq}{from_file}8{p}" for p in promo_pieces)
        # Capture left
        if f_idx > 0:
            promotions.update(f"{from_sq}{files[f_idx - 1]}8{p}" for p in promo_pieces)
        # Capture right
        if f_idx < 7:
            promotions.update(f"{from_sq}{files[f_idx + 1]}8{p}" for p in promo_pieces)

    # Black promotions: rank 2 -> rank 1
    for f_idx, from_file in enumerate(files):
        from_sq = f"{from_file}2"
        # Forward
        promotions.update(f"{from_sq}{from_file}1{p}" for p in promo_pieces)
        # Capture left
        if f_idx > 0:
            promotions.update(f"{from_sq}{files[f_idx - 1]}1{p}" for p in promo_pieces)
        # Capture right
        if f_idx < 7:
            promotions.update(f"{from_sq}{files[f_idx + 1]}1{p}" for p in promo_pieces)

    return promotions


def build_move_vocabulary() -> dict[str, int]:
    """
    Build vocabulary of all UCI moves in the dataset plus all possible promotions.
    Returns dict mapping move string -> index. Cached after first call.
    """
    global VOCAB_CACHE
    if VOCAB_CACHE is not None:
        return VOCAB_CACHE

    print("Building move vocabulary from dataset...")
    df = get_raw_shuffled_data()
    all_moves = set(df["line"].apply(extract_first_move))

    # Add all possible promotions to ensure coverage
    all_promotions = get_all_promotions()
    new_promos = all_promotions - all_moves
    all_moves = all_moves | all_promotions

    unique_moves = sorted(all_moves)
    VOCAB_CACHE = {move: idx for idx, move in enumerate(unique_moves)}
    print(f"Vocabulary size: {len(VOCAB_CACHE)} ({len(new_promos)} promotions added)")
    return VOCAB_CACHE


def get_index_to_move() -> dict[int, str]:
    """Get reverse mapping from index to move string."""
    vocab = build_move_vocabulary()
    return {idx: move for move, idx in vocab.items()}
