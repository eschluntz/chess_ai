"""
Tests for feature extraction utilities.
"""

import chess
import numpy as np

from learning_board_value.feature_extraction import (
    NUM_CASTLING_FEATURES,
    NUM_COLORS,
    NUM_MATERIAL_FEATURES,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    NUM_TURN_FEATURES,
    PIECE_SQUARE_FEATURES,
    TOTAL_FEATURES,
    extract_features_piece_square,
)


def test_extract_features_piece_square_starting_position():
    """Test feature extraction on the starting chess position with exact expected patterns."""
    board = chess.Board()  # Starting position

    features = extract_features_piece_square(board)

    # Check basic properties
    assert len(features) == TOTAL_FEATURES, (
        f"Expected {TOTAL_FEATURES} features, got {len(features)}"
    )
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"

    # Helper function to extract piece features as 8x8 grids
    def get_piece_grid(piece_idx: int, color: int) -> np.ndarray:
        """Extract features for a specific piece type and color as 8x8 grid."""
        start_idx = piece_idx * (NUM_SQUARES * NUM_COLORS)
        grid = np.zeros((8, 8), dtype=np.float32)
        for square in range(64):
            rank, file = divmod(square, 8)
            feature_idx = start_idx + square * NUM_COLORS + color
            grid[rank, file] = features[feature_idx]
        return grid

    # Test white rooks (piece_idx=3, color=0)
    white_rook_grid = get_piece_grid(3, 0)  # Rooks are index 3 in PIECE_TYPES
    # fmt: off
    expected_white_rooks = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1],  # rank 1: rooks on a1, h1
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 2
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 3
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 4
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 6
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 7
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 8
    ], dtype=np.float32)
    # fmt: on
    np.testing.assert_array_equal(
        white_rook_grid, expected_white_rooks, "White rook positions don't match"
    )

    # Test black rooks (piece_idx=3, color=1)
    black_rook_grid = get_piece_grid(3, 1)
    # fmt: off
    expected_black_rooks = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 1
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 2
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 3
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 4
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 6
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 7
        [1, 0, 0, 0, 0, 0, 0, 1],  # rank 8: rooks on a8, h8
    ], dtype=np.float32)
    # fmt: on
    np.testing.assert_array_equal(
        black_rook_grid, expected_black_rooks, "Black rook positions don't match"
    )

    # Test white pawns (piece_idx=0, color=0)
    white_pawn_grid = get_piece_grid(0, 0)  # Pawns are index 0 in PIECE_TYPES
    # fmt: off
    expected_white_pawns = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 1
        [1, 1, 1, 1, 1, 1, 1, 1],  # rank 2: all white pawns
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 3
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 4
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 6
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 7
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 8
    ], dtype=np.float32)
    # fmt: on
    np.testing.assert_array_equal(
        white_pawn_grid, expected_white_pawns, "White pawn positions don't match"
    )

    # Test white king (piece_idx=5, color=0)
    white_king_grid = get_piece_grid(5, 0)  # King is index 5 in PIECE_TYPES
    # fmt: off
    expected_white_king = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0],  # rank 1: king on e1
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 2
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 3
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 4
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 6
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 7
        [0, 0, 0, 0, 0, 0, 0, 0],  # rank 8
    ], dtype=np.float32)
    # fmt: on
    np.testing.assert_array_equal(
        white_king_grid, expected_white_king, "White king position doesn't match"
    )

    # Check additional features (starting after piece-square features)
    additional_features = features[PIECE_SQUARE_FEATURES:]

    # Turn feature: should be 1 for white to move (starting position)
    assert additional_features[0] == 1, (
        f"Expected turn feature 1 (white), got {additional_features[0]}"
    )

    # Castling rights: all should be 1 in starting position
    for i in range(1, 5):
        assert additional_features[i] == 1, (
            f"Expected castling feature {i} to be 1, got {additional_features[i]}"
        )

    # Material balance features: should all be 0 in starting position (equal material)
    for i in range(5, 5 + NUM_MATERIAL_FEATURES):
        assert additional_features[i] == 0, (
            f"Expected material balance feature {i} to be 0, got {additional_features[i]}"
        )


def test_extract_features_piece_square_feature_count():
    """Test that feature extraction returns the expected number of features."""
    board = chess.Board()
    features = extract_features_piece_square(board)

    expected_total = (
        NUM_PIECE_TYPES * NUM_SQUARES * NUM_COLORS
        + NUM_TURN_FEATURES
        + NUM_CASTLING_FEATURES
        + NUM_MATERIAL_FEATURES
    )

    assert len(features) == expected_total == TOTAL_FEATURES
