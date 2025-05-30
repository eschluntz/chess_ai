"""
Helper functions for evaluation and training scripts.
Extracted from 2-eval-accuracy.py to avoid code duplication.
"""

from typing import Any

import chess
import pandas as pd
from datasets import load_dataset


def get_training_dataset_stream(skip_first_n=10000):
    """
    Get the Lichess dataset stream for training, skipping the first N positions.
    
    The first 10k positions are reserved for evaluation to avoid data leakage.
    
    Args:
        skip_first_n: Number of positions to skip (default: 10000)
        
    Returns:
        Dataset stream starting after the skipped positions
    """
    ds = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    return ds["train"].skip(skip_first_n)


def get_train_df(
    num_train_samples: int,
    skip_first_n: int = 10000,
    include_mates: bool = True,
    max_mate_distance: int = 15,
    max_cp: int = 750,
) -> pd.DataFrame:
    """
    Get preprocessed training data as a DataFrame with chess boards and scores.
    
    To avoid temporal bias, this function loads 1M positions, shuffles them,
    and then samples the requested number.
    
    Args:
        num_train_samples: Number of training samples to return
        skip_first_n: Number of positions to skip (default: 10000)
        include_mates: Whether to include mate positions (default: True)
        max_mate_distance: Maximum mate distance to include (default: 15)
        max_cp: Maximum centipawn value to include (default: 750)
        
    Returns:
        DataFrame with columns: ['fen', 'true_score', 'board']
    """
    import time
    
    start_time = time.time()
    
    # Always load 1M positions to create a large pool
    pool_size = 1_000_000
    print(f"Loading {pool_size:,} positions to create shuffled pool...")
    load_start = time.time()
    train_stream = get_training_dataset_stream(skip_first_n)
    train_data = list(train_stream.take(pool_size))
    load_time = time.time() - load_start
    print(f"  Loading took {load_time:.2f} seconds")

    # Process training data
    print("Processing training data...")
    process_start = time.time()
    train_df = pd.DataFrame(train_data)
    train_df = train_df.dropna(subset=["fen"])
    dataframe_time = time.time() - process_start
    print(f"  DataFrame creation took {dataframe_time:.2f} seconds")

    # Filter positions
    filter_start = time.time()
    train_df["should_skip"] = train_df.apply(
        lambda row: should_skip_position(
            row, 
            include_mates=include_mates,
            max_mate_distance=max_mate_distance,
            max_cp=max_cp
        ), 
        axis=1
    )
    train_df = train_df[~train_df["should_skip"]].copy()
    filter_time = time.time() - filter_start
    print(f"  Filtering took {filter_time:.2f} seconds")

    # Extract scores and create boards
    score_start = time.time()
    train_df["true_score"] = train_df.apply(extract_centipawn_score, axis=1)
    score_time = time.time() - score_start
    print(f"  Score extraction took {score_time:.2f} seconds")
    
    board_start = time.time()
    train_df["board"] = train_df["fen"].apply(create_board_from_fen)
    board_time = time.time() - board_start
    print(f"  Board creation took {board_time:.2f} seconds")

    print(f"Filtered to {len(train_df)} valid positions")
    
    # Shuffle the dataframe to remove temporal bias
    print("Shuffling data to remove temporal bias...")
    shuffle_start = time.time()
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuffle_time = time.time() - shuffle_start
    print(f"  Shuffling took {shuffle_time:.2f} seconds")
    
    # Sample the requested number of positions
    sample_start = time.time()
    if num_train_samples > len(train_df):
        print(f"Warning: Requested {num_train_samples:,} samples but only {len(train_df):,} available")
        sampled_df = train_df
    else:
        sampled_df = train_df.head(num_train_samples)
        print(f"Sampled {num_train_samples:,} positions from shuffled pool")
    sample_time = time.time() - sample_start
    print(f"  Sampling took {sample_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"Total get_train_df time: {total_time:.2f} seconds")
    
    # Return only the columns needed for training
    return sampled_df[["fen", "true_score", "board"]]


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


def process_dataset_batch(
    dataset_stream: Any,
    num_samples: int,
    max_mate_distance: int = 15,
    max_cp: int = 750,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Process dataset and return filtered examples as a DataFrame.
    Uses functional approach with dataset.take() and DataFrame operations.

    Args:
        dataset_stream: The streaming dataset
        num_samples: Number of samples to process
        max_mate_distance: Maximum mate distance to include (default: 15)
        max_cp: Maximum centipawn value to include (default: 750)

    Returns:
        tuple: (processed_df, stats_dict)
    """
    # Load data using take()
    print(f"Loading {num_samples} positions...")
    data = list(dataset_stream.take(num_samples))

    # Convert to DataFrame
    df = pd.DataFrame(data)
    initial_count = len(df)

    # First, filter out any rows with NaN values in critical columns
    df = df.dropna(subset=["fen"])

    # Apply filtering using the helper function (always include mates for processing)
    df["should_skip"] = df.apply(
        lambda row: should_skip_position(
            row, include_mates=True, max_mate_distance=max_mate_distance, max_cp=max_cp
        ),
        axis=1,
    )

    # Filter out skipped positions
    df_filtered = df[~df["should_skip"]].copy()

    # Add true score and metadata for remaining positions
    df_filtered["true_score"] = df_filtered.apply(extract_centipawn_score, axis=1)
    df_filtered["is_mate"] = df_filtered["mate"].notna()

    # Calculate simple stats
    stats = {
        "total_loaded": initial_count,
        "total_filtered": initial_count - len(df_filtered),
        "mate_positions": df_filtered["is_mate"].sum(),
        "total_processed": len(df_filtered),
    }

    # Return only needed columns
    processed_df = df_filtered[["fen", "true_score", "is_mate"]]

    return processed_df, stats
