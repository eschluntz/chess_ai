"""
Feature extraction utilities for chess position evaluation.
"""

import chess
import numpy as np


def extract_features_basic(board: chess.Board) -> np.ndarray:
    """
    Extract features from a chess board for the random forest model.
    Simple features based on material and piece positions.
    """
    features = []

    # Material count for each piece type
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in piece_types:
            count = len(board.pieces(piece_type, color))
            features.append(count if color == chess.WHITE else -count)

    # Piece-square features (simplified)
    # Count pieces in center squares (d4, d5, e4, e5)
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    white_center = sum(
        1
        for sq in center_squares
        if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
    )
    black_center = sum(
        1
        for sq in center_squares
        if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK
    )
    features.append(white_center - black_center)

    # Pawn structure features
    # Count doubled pawns
    white_doubled = 0
    black_doubled = 0
    for file in range(8):
        white_pawns_on_file = sum(
            1
            for rank in range(8)
            if board.piece_at(chess.square(file, rank))
            == chess.Piece(chess.PAWN, chess.WHITE)
        )
        black_pawns_on_file = sum(
            1
            for rank in range(8)
            if board.piece_at(chess.square(file, rank))
            == chess.Piece(chess.PAWN, chess.BLACK)
        )
        if white_pawns_on_file > 1:
            white_doubled += white_pawns_on_file - 1
        if black_pawns_on_file > 1:
            black_doubled += black_pawns_on_file - 1
    features.append(white_doubled - black_doubled)

    # King safety (distance from center)
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    if white_king_square is not None:
        white_king_center_dist = abs(chess.square_file(white_king_square) - 3.5) + abs(
            chess.square_rank(white_king_square) - 3.5
        )
    else:
        white_king_center_dist = 0
    if black_king_square is not None:
        black_king_center_dist = abs(chess.square_file(black_king_square) - 3.5) + abs(
            chess.square_rank(black_king_square) - 3.5
        )
    else:
        black_king_center_dist = 0
    features.append(white_king_center_dist - black_king_center_dist)

    # Turn to move (1 if white, -1 if black)
    features.append(1 if board.turn == chess.WHITE else -1)

    # Castling rights
    features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    features.append(-1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    features.append(-1 if board.has_queenside_castling_rights(chess.BLACK) else 0)

    return np.array(features)


def extract_features_piece_square(board: chess.Board) -> np.ndarray:
    """
    Extract features that allow learning piece-square tables.
    Creates a feature for every possible board position for every piece type.

    This creates 6 piece types * 64 squares * 2 colors = 768 binary features
    plus the same additional features as the basic extractor.
    """
    # Pre-allocate the feature array
    # 768 piece-square features + 11 additional features
    features = np.zeros(779, dtype=np.float32)

    # Piece-square features: one-hot encoding for each piece type on each square
    # Order: pawn, knight, bishop, rook, queen, king
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    # Get piece locations for each type and color
    for piece_idx, piece_type in enumerate(piece_types):
        # White pieces
        white_squares = board.pieces(piece_type, chess.WHITE)
        for square in white_squares:
            # Feature index: piece_type_index * 128 + square * 2 + 0 (white)
            features[piece_idx * 128 + square * 2] = 1

        # Black pieces
        black_squares = board.pieces(piece_type, chess.BLACK)
        for square in black_squares:
            # Feature index: piece_type_index * 128 + square * 2 + 1 (black)
            features[piece_idx * 128 + square * 2 + 1] = 1

    # Add additional features starting at index 768
    base_idx = 768

    # Turn to move (1 if white, -1 if black)
    features[base_idx] = 1 if board.turn == chess.WHITE else -1

    # Castling rights
    features[base_idx + 1] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    features[base_idx + 2] = (
        1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    )
    features[base_idx + 3] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    features[base_idx + 4] = (
        1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    )

    # Total material count (helps model understand material balance)
    for i, piece_type in enumerate(piece_types):
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        features[base_idx + 5 + i] = white_count - black_count

    return features


# For backward compatibility
extract_features = extract_features_basic
