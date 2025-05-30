"""
Helper functions for evaluation and training scripts.
Extracted from 2-eval-accuracy.py to avoid code duplication.
"""

from typing import Any

import chess
import pandas as pd


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
