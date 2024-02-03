#!/usr/bin/env python

"""
Contains classical eval functions for chess
"""
import chess


CHECKMATE_VALUE = 10_000


def _game_over_eval(board):
    """Handles evaluating boards that are over.
    Returns (score, done)"""

    if board.is_checkmate():
        score = CHECKMATE_VALUE if not board.turn else -CHECKMATE_VALUE
        return score, True
    if board.is_insufficient_material():
        return 0, True
    if not any(board.generate_legal_moves()):
        return 0, True

    # Automatic draws.
    if board.is_seventyfive_moves():
        return 0, True
    if board.is_fivefold_repetition():
        return 0, True

    return None, False


# Piece values

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King's value is not considered in this simple eval
}


def piece_value_eval(board):
    """Evaluates the board and returns a score and a done flag.

    board: chess.Board object

    Returns:
    score (int): The score of the board. Positive for white's advantage,
    negative for black's.
    done (bool): True if the game is over, False otherwise.
    """

    # Check if the game is done
    score, done = _game_over_eval(board)
    if done:
        return (score, done)

    # Calculate the score based on piece values
    score = 0
    for piece_type in PIECE_VALUES:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        score += (white_pieces - black_pieces) * PIECE_VALUES[piece_type]

    return score, False


# Piece-Square Tables
# These values are placeholders. Actual values should be filled in for each piece type.
# fmt: off
PIECE_SQUARE_TABLES = {
    chess.PAWN: [
        0,   0,   0,   0,   0,   0,   0,   0,
        50,  50,  50,  50,  50,  50,  50,  50,
        10,  10,  20,  30,  30,  20,  10,  10,
        5,   5,   10,  25,  25,  10,  5,   5,
        0,   0,   0,   20,  20,  0,   0,   0,
        5,  -5,  -10,  0,   0,  -10, -5,   5,
        5,   10,  10, -20, -20,  10,  10,  5,
        0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    chess.ROOK: [
         0,   0,   0,   0,   0,   0,   0,   0,
         5,  10,  10,  10,  10,  10,  10,   5,
        -5,   0,   0,   0,   0,   0,   0,  -5,  # noqa: E131
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
         0,   0,   0,   5,   5,   0,   0,   0,
    ],
    chess.QUEEN: [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,  # noqa: E131
          0,   0,   5,   5,   5,   5,   0,  -5,  # noqa: E131
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ],
    chess.KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
         20,  20,   0,   0,   0,   0,  20,  20,  # noqa: E131
         20,  30,  10,   0,   0,  10,  30,  20,  # noqa: E131
    ],
}
# fmt: on


def piece_position_eval(board):
    """Evaluates the board based on the position of each piece using piece-square tables.

    board: chess.Board object

    Returns:
    score (int): The score of the board. Positive for white's advantage,
    negative for black's.
    """
    # Check if the game is done
    score, done = _game_over_eval(board)
    if done:
        return (score, done)

    score = 0
    for piece_type in PIECE_SQUARE_TABLES:
        for square in board.pieces(piece_type, chess.WHITE):
            score += PIECE_SQUARE_TABLES[piece_type][square] + PIECE_VALUES[piece_type]
        for square in board.pieces(piece_type, chess.BLACK):
            square = chess.square_mirror(square)  # look at the table from black's perspective
            score -= PIECE_SQUARE_TABLES[piece_type][square] + PIECE_VALUES[piece_type]
    return score, False
