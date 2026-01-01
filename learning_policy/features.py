"""
Feature extraction for chess positions.
"""

import chess
import numpy as np

# Constants for piece-square feature extraction
PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

NUM_PIECE_TYPES = len(PIECE_TYPES)
NUM_SQUARES = 64
NUM_COLORS = 2
NUM_CASTLING_FEATURES = 4
NUM_TURN_FEATURES = 1
NUM_MATERIAL_FEATURES = NUM_PIECE_TYPES

PIECE_SQUARE_FEATURES = NUM_PIECE_TYPES * NUM_SQUARES * NUM_COLORS  # 768
FEATURES_PER_PIECE_TYPE = NUM_SQUARES * NUM_COLORS  # 128
ADDITIONAL_FEATURES = NUM_TURN_FEATURES + NUM_CASTLING_FEATURES + NUM_MATERIAL_FEATURES  # 11
TOTAL_FEATURES = PIECE_SQUARE_FEATURES + ADDITIONAL_FEATURES  # 779


def extract_features_piece_square(board: chess.Board) -> np.ndarray:
    """
    Extract piece-square features from a chess board.

    Creates 768 binary features (6 piece types * 64 squares * 2 colors)
    plus 11 additional features (turn, castling, material balance).

    Returns: 779-dimensional float32 array
    """
    features = np.zeros(TOTAL_FEATURES, dtype=np.float32)

    for piece_idx, piece_type in enumerate(PIECE_TYPES):
        white_squares = board.pieces(piece_type, chess.WHITE)
        for square in white_squares:
            features[piece_idx * FEATURES_PER_PIECE_TYPE + square * NUM_COLORS] = 1

        black_squares = board.pieces(piece_type, chess.BLACK)
        for square in black_squares:
            features[piece_idx * FEATURES_PER_PIECE_TYPE + square * NUM_COLORS + 1] = 1

    base_idx = PIECE_SQUARE_FEATURES

    features[base_idx] = 1 if board.turn == chess.WHITE else -1
    features[base_idx + 1] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    features[base_idx + 2] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    features[base_idx + 3] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    features[base_idx + 4] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

    for i, piece_type in enumerate(PIECE_TYPES):
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        features[base_idx + 5 + i] = white_count - black_count

    return features
