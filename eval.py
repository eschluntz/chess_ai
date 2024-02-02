#!/usr/bin/env python

"""
Contains classical eval functions for chess
"""
import numpy as np
import chess


def _game_over_eval(board):
    """Handles evaluating boards that are over.
    Returns (score, done)"""
    outcome = board.outcome()
    if outcome is not None:
        if outcome.winner is None:
            return 0, True  # Draw
        elif outcome.winner:
            return 100, True  # White wins
        else:
            return -100, True  # Black wins
    return None, False


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
    
    # Piece values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King's value is not considered in this simple eval
    }
    
    # Calculate the score based on piece values
    score = 0
    for piece_type in piece_values:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        print(f"{piece_type} {white_pieces} {black_pieces}")
        score += (white_pieces - black_pieces) * piece_values[piece_type]
    
    return score, False
