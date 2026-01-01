"""
Visualization utilities for piece-square tables and model features.
"""

import chess
import numpy as np

from core.eval import PIECE_VALUES
from learning.feature_extraction import PIECE_TYPES


def build_feature_names():
    """Build human-readable feature names."""
    piece_names = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King",
    }

    feature_names = []
    # Piece-square features
    for piece_type, piece_name in piece_names.items():
        for square in range(64):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            square_name = f"{chr(ord('a') + file)}{rank + 1}"
            feature_names.append(f"White_{piece_name}_{square_name}")
            feature_names.append(f"Black_{piece_name}_{square_name}")

    # Additional features
    feature_names.extend(
        [
            "Turn_White",
            "Castling_White_Kingside",
            "Castling_White_Queenside",
            "Castling_Black_Kingside",
            "Castling_Black_Queenside",
        ]
    )
    return feature_names


def get_piece_square_tables(model):
    """
    Extract learned piece-square tables from the model coefficients.
    Returns a dictionary mapping piece type to 8x8 array of values.
    """
    coefficients = model.model.coef_
    piece_square_tables = {}

    piece_types = PIECE_TYPES
    piece_names = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King",
    }

    for piece_idx, piece_type in enumerate(piece_types):
        # Initialize tables for white and black
        white_table = np.zeros((8, 8))
        black_table = np.zeros((8, 8))

        for square in range(64):
            rank = square // 8
            file = square % 8

            # Get coefficient for white piece on this square
            white_coef_idx = piece_idx * 128 + square * 2
            white_value = PIECE_VALUES[piece_type] + coefficients[white_coef_idx]
            white_table[rank, file] = white_value

            # Get coefficient for black piece on this square
            black_coef_idx = piece_idx * 128 + square * 2 + 1
            # Note: Black coefficients are negative in the model
            black_value = -PIECE_VALUES[piece_type] + coefficients[black_coef_idx]
            black_table[rank, file] = black_value

        piece_square_tables[piece_names[piece_type]] = {
            "white": white_table,
            "black": black_table,
            "deltas_white": white_table - PIECE_VALUES[piece_type],
            "deltas_black": black_table + PIECE_VALUES[piece_type],
        }

    # Add other feature coefficients
    base_idx = 768
    other_features = {
        "turn_bonus": coefficients[base_idx],
        "white_kingside_castling": coefficients[base_idx + 1],
        "white_queenside_castling": coefficients[base_idx + 2],
        "black_kingside_castling": coefficients[base_idx + 3],
        "black_queenside_castling": coefficients[base_idx + 4],
        "intercept": model.model.intercept_,
    }

    return piece_square_tables, other_features


def print_other_learned_features(model):
    """Print other learned features like castling and turn bonus."""
    _, other_features = get_piece_square_tables(model)
    print("\n" + "=" * 60)
    print("OTHER LEARNED FEATURES")
    print("=" * 60)
    print(f"Turn bonus (white to move): {other_features['turn_bonus']:.1f} cp")
    print(
        f"White kingside castling: {other_features['white_kingside_castling']:.1f} cp"
    )
    print(
        f"White queenside castling: {other_features['white_queenside_castling']:.1f} cp"
    )
    print(
        f"Black kingside castling: {other_features['black_kingside_castling']:.1f} cp"
    )
    print(
        f"Black queenside castling: {other_features['black_queenside_castling']:.1f} cp"
    )
    print(f"Model intercept: {other_features['intercept']:.1f} cp")


def visualize_piece_square_table(model, piece_name, color="white", show_deltas=True):
    """
    Print ASCII visualization of a piece-square table.
    """
    tables, other_features = get_piece_square_tables(model)

    if show_deltas:
        table = tables[piece_name][f"deltas_{color}"]
        print(f"\n{piece_name} Position Value Deltas ({color}):")
    else:
        table = tables[piece_name][color]
        print(f"\n{piece_name} Total Position Values ({color}):")

    print("    a      b      c      d      e      f      g      h")
    print("  +------+------+------+------+------+------+------+------+")

    for rank in range(7, -1, -1):  # Start from rank 8 down to rank 1
        print(f"{rank + 1} |", end="")
        for file in range(8):
            value = table[rank, file]
            # Format with sign for deltas, handle up to 6 characters
            if show_deltas and value >= 0:
                print(f" {value:+5.0f}|", end="")
            else:
                print(f" {value:5.0f}|", end="")
        print(f" {rank + 1}")
        if rank > 0:
            print("  +------+------+------+------+------+------+------+------+")

    print("  +------+------+------+------+------+------+------+------+")
    print("    a      b      c      d      e      f      g      h")

    # Print statistics about the values
    if show_deltas:
        abs_values = np.abs(table.flatten())
        print(f"  Average magnitude: {np.mean(abs_values):.1f} cp")
        print(f"  Max magnitude: {np.max(abs_values):.1f} cp")
        print(f"  Std deviation: {np.std(table.flatten()):.1f} cp")


def print_all_piece_square_tables(model):
    """Print all piece-square tables and other learned features."""
    print("\n" + "=" * 60)
    print("LEARNED PIECE-SQUARE TABLES (Deltas from base values)")
    print("=" * 60)

    piece_names = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]
    for piece_name in piece_names:
        visualize_piece_square_table(model, piece_name, color="white", show_deltas=True)

    # Show other learned features
    print_other_learned_features(model)
