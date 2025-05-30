"""
Helper functions for evaluation and training scripts.
Extracted from 2-eval-accuracy.py to avoid code duplication.
"""

from typing import Any

import chess
import pandas as pd
from datasets import load_dataset

# Cache for the raw shuffled dataset (before expensive processing)
_raw_shuffled_cache = None

# Global constant for evaluation set size
EVAL_SET_SIZE = 10000


def get_raw_shuffled_data() -> pd.DataFrame:
    """
    Get the raw shuffled dataset with minimal processing.
    Uses a cached parquet file if available, otherwise creates it from streaming data.
    """
    global _raw_shuffled_cache
    import os

    # Check if we already have the raw data cached in memory
    if _raw_shuffled_cache is not None:
        print("Using cached raw shuffled dataset from memory")
        return _raw_shuffled_cache

    # Check if we have a saved parquet file
    cache_file = "learning/cache/shuffled_1m_dataset.parquet"

    if os.path.exists(cache_file):
        print(f"Loading shuffled dataset from cache file: {cache_file}")
        train_df = pd.read_parquet(cache_file)

        # Cache in memory too
        _raw_shuffled_cache = train_df
        return train_df

    # Need to create the cached file
    print("No cached file found. Creating shuffled dataset from streaming data...")

    # Always load 1M positions from the beginning using streaming
    pool_size = 1_000_000

    print(f"Streaming first {pool_size:,} positions...")
    ds = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    train_data = list(ds["train"].take(pool_size))

    # Convert to pandas and basic cleanup
    print("Converting to pandas and basic cleanup...")
    train_df = pd.DataFrame(train_data)
    train_df = train_df.dropna(subset=["fen"])

    # Shuffle the raw dataframe to remove temporal bias
    print("Shuffling raw data to remove temporal bias...")
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save to parquet for future use
    print(f"Saving shuffled dataset to cache file: {cache_file}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    train_df.to_parquet(cache_file, index=False)

    print("Dataset preparation complete. Future runs will load from cache much faster.")

    # Cache the result in memory
    _raw_shuffled_cache = train_df

    return _raw_shuffled_cache


def process_subset(
    raw_df: pd.DataFrame,
    include_mates: bool = True,
    max_mate_distance: int = 15,
    max_cp: int = 750,
) -> pd.DataFrame:
    """
    Process a subset of raw data with filtering, score extraction, and board creation.
    """
    # Filter positions
    raw_df["should_skip"] = raw_df.apply(
        lambda row: should_skip_position(
            row,
            include_mates=include_mates,
            max_mate_distance=max_mate_distance,
            max_cp=max_cp,
        ),
        axis=1,
    )
    filtered_df = raw_df[~raw_df["should_skip"]].copy()

    # Extract scores and create boards
    filtered_df["true_score"] = filtered_df.apply(extract_centipawn_score, axis=1)
    filtered_df["board"] = filtered_df["fen"].apply(create_board_from_fen)

    # Add is_mate column for compatibility with eval_common.py
    filtered_df["is_mate"] = filtered_df["mate"].notna()

    print(
        f"Processed {len(raw_df)} raw positions -> {len(filtered_df)} valid positions"
    )

    return filtered_df[["fen", "true_score", "board", "is_mate"]]


def get_eval_df(num_samples: int = EVAL_SET_SIZE, **kwargs) -> pd.DataFrame:
    """
    Get evaluation data - first N samples from the shuffled dataset.
    Only processes the subset we actually need.

    Args:
        num_samples: Number of samples for evaluation (default: EVAL_SET_SIZE)
        **kwargs: Additional arguments passed to process_subset

    Returns:
        DataFrame with evaluation data
    """
    # Get raw shuffled data
    raw_df = get_raw_shuffled_data()

    if num_samples > len(raw_df):
        print(
            f"Warning: Requested {num_samples:,} samples but only {len(raw_df):,} available"
        )
        raw_subset = raw_df
    else:
        raw_subset = raw_df.head(num_samples)

    print(f"Processing first {len(raw_subset):,} positions for evaluation...")

    # Process only the subset we need
    eval_df = process_subset(raw_subset, **kwargs)

    return eval_df


def get_train_df(num_train_samples: int, **kwargs) -> pd.DataFrame:
    """
    Get training data - samples after the first EVAL_SET_SIZE from the shuffled dataset.
    Only processes the subset we actually need.

    Args:
        num_train_samples: Number of training samples to return
        **kwargs: Additional arguments passed to process_subset

    Returns:
        DataFrame with training data
    """
    # Get raw shuffled data
    raw_df = get_raw_shuffled_data()

    # Skip first EVAL_SET_SIZE (reserved for evaluation)
    available_for_training = len(raw_df) - EVAL_SET_SIZE

    if num_train_samples > available_for_training:
        print(
            f"Warning: Requested {num_train_samples:,} training samples but only {available_for_training:,} available after reserving {EVAL_SET_SIZE:,} for evaluation"
        )
        raw_subset = raw_df[EVAL_SET_SIZE:]
    else:
        raw_subset = raw_df[EVAL_SET_SIZE : EVAL_SET_SIZE + num_train_samples]

    print(
        f"Processing {len(raw_subset):,} positions for training (after skipping {EVAL_SET_SIZE:,} for evaluation)..."
    )

    # Process only the subset we need
    train_df = process_subset(raw_subset, **kwargs)

    return train_df


def create_board_from_fen(fen: str) -> chess.Board:
    """Create a chess.Board object from FEN string."""
    return chess.Board(fen)


def should_skip_position(
    example: dict[str, Any],
    include_mates: bool = True,
    max_mate_distance: int = 15,
    max_cp: int = 750,
) -> bool:
    """
    Determine if a position should be skipped based on filtering criteria.

    Args:
        example: The position data from the dataset
        include_mates: Whether to include mate positions (default: True)
        max_mate_distance: Maximum mate distance to include (default: 15)
        max_cp: Maximum centipawn value to include (default: 750)

    Returns:
        Boolean indicating if position should be skipped
    """
    # Skip if neither mate nor cp value exists
    if pd.isna(example.get("mate")) and pd.isna(example.get("cp")):
        return True

    if pd.notna(example.get("mate")):
        if not include_mates:
            return True

        # Filter out very long mates
        if abs(example["mate"]) > max_mate_distance:
            return True

        return False
    else:
        # Skip if cp is missing
        if pd.isna(example.get("cp")):
            return True

        # Filter out extreme positions
        if abs(example["cp"]) > max_cp:
            return True

        return False


def extract_centipawn_score(example: dict[str, Any]) -> int:
    """
    Extract the centipawn score from the example, converting mate scores if needed.

    For mate positions, uses the formula: 20000 - (300 * mate_in_moves)
    This gives mate-in-1 = 19700 cp, mate-in-10 = 17000 cp, etc.

    This function should only be called after should_skip_position returns False,
    so we can safely assume any mate score needs conversion.

    Returns:
        centipawn_score: The centipawn score (either direct or converted from mate)
    """
    if pd.notna(example.get("mate")):
        mate_in_moves = example["mate"]
        if mate_in_moves > 0:
            # Positive means white has mate
            return 20000 - (300 * mate_in_moves)
        else:
            # Negative means black has mate
            return -20000 + (300 * abs(mate_in_moves))
    else:
        # Return cp value, or 0 if it's missing (should be rare after filtering)
        return example.get("cp", 0)
