"""
Tests for feature extraction utilities.
"""

import chess
import numpy as np

from learning.feature_extraction import (
    extract_features_piece_square,
    PIECE_TYPES,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    NUM_COLORS,
    NUM_CASTLING_FEATURES,
    NUM_TURN_FEATURES,
    NUM_MATERIAL_FEATURES,
    TOTAL_FEATURES,
)


def test_extract_features_piece_square_starting_position():
    """Test feature extraction on the starting chess position."""
    board = chess.Board()  # Starting position
    
    features = extract_features_piece_square(board)
    
    # Check basic properties
    assert len(features) == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES} features, got {len(features)}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    
    # Check that we have the right number of pieces
    # Starting position: 8 pawns, 2 rooks, 2 knights, 2 bishops, 1 queen, 1 king per side
    expected_piece_counts = {
        chess.PAWN: 8,
        chess.ROOK: 2, 
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.QUEEN: 1,
        chess.KING: 1
    }
    
    # Piece-square features are in first 768 positions (6 pieces * 64 squares * 2 colors)
    piece_square_features = features[:768]
    
    # Count active piece-square features for each piece type
    for piece_idx, piece_type in enumerate(PIECE_TYPES):
        # Features for this piece type start at piece_idx * FEATURES_PER_PIECE_TYPE
        features_per_piece_type = NUM_SQUARES * NUM_COLORS
        start_idx = piece_idx * features_per_piece_type
        end_idx = start_idx + features_per_piece_type
        piece_features = piece_square_features[start_idx:end_idx]
        
        # Count white pieces (even indices) and black pieces (odd indices)
        white_count = np.sum(piece_features[::2])  # Even indices
        black_count = np.sum(piece_features[1::2])  # Odd indices
        
        expected_count = expected_piece_counts[piece_type]
        assert white_count == expected_count, f"White {piece_type}: expected {expected_count}, got {white_count}"
        assert black_count == expected_count, f"Black {piece_type}: expected {expected_count}, got {black_count}"
    
    # Check additional features (starting at index 768)
    additional_features = features[768:]
    
    # Turn feature (index 0 of additional): should be 1 for white to move (starting position)
    assert additional_features[0] == 1, f"Expected turn feature 1 (white), got {additional_features[0]}"
    
    # Castling rights (indices 1-4): all should be 1 in starting position
    for i in range(1, 5):
        assert additional_features[i] == 1, f"Expected castling feature {i} to be 1, got {additional_features[i]}"
    
    # Material balance features (indices 5-10): should all be 0 in starting position
    for i in range(5, 11):
        assert additional_features[i] == 0, f"Expected material balance feature {i} to be 0, got {additional_features[i]}"


def test_extract_features_piece_square_feature_count():
    """Test that feature extraction returns the expected number of features."""
    board = chess.Board()
    features = extract_features_piece_square(board)
    
    expected_total = (NUM_PIECE_TYPES * NUM_SQUARES * NUM_COLORS + 
                     NUM_TURN_FEATURES + NUM_CASTLING_FEATURES + NUM_MATERIAL_FEATURES)
    
    assert len(features) == expected_total == TOTAL_FEATURES