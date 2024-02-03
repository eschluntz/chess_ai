#!/usr/bin/env python

"""
Contains classical eval functions for chess
"""
import numpy as np
import chess


def _game_over_eval(board):
    """Handles evaluating boards that are over.
    Returns (score, done)"""

    if board.is_checkmate():
        return 100 if not board.turn else -100, True
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
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King's value is not considered in this simple eval
}


def piece_value_eval(board):
    """Evaluates the board and returns a score and a done flag.
    
    board: chess.Board object
    
    Returns:
    score (int): The score of the board. Positive for white's advantage, negative for black's.
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