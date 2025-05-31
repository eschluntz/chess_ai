#!/usr/bin/env python

"""
Contains classical eval functions for chess
"""

import chess

CHECKMATE_VALUE = 20_000  # Compatible with mate-to-cp conversion


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
            square = chess.square_mirror(
                square
            )  # look at the table from black's perspective
            score -= PIECE_SQUARE_TABLES[piece_type][square] + PIECE_VALUES[piece_type]
    return score, False


# Machine Learning Models


def load_linear_piece_square_model(model_path):
    """Load a trained linear piece-square model from disk."""
    import importlib.util
    import os
    import pickle
    import sys

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Import LinearPieceSquareModel class for pickle loading
    spec_31 = importlib.util.spec_from_file_location(
        "train_piece_value_table",
        os.path.join(
            os.path.dirname(__file__), "..", "learning", "31-train-piece-value-table.py"
        ),
    )
    train_piece_value_table = importlib.util.module_from_spec(spec_31)
    
    # The model was saved when running as __main__, so we need to make the class available there
    old_main = sys.modules.get("__main__")
    sys.modules["__main__"] = train_piece_value_table
    
    try:
        spec_31.loader.exec_module(train_piece_value_table)

        with open(model_path, "rb") as f:
            model = pickle.load(f)
    finally:
        # Restore the original __main__
        if old_main is not None:
            sys.modules["__main__"] = old_main

    return model


def load_random_forest_model(model_path):
    """Load a trained random forest model from disk."""
    import os
    import pickle

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old and new model formats
    if isinstance(data, dict):
        model = data["model"]
        feature_type = data["feature_type"]
    else:
        # Old format - assume basic features
        model = data
        feature_type = "piece_square"

    return model, feature_type


def linear_piece_square_eval(board, model=None, model_path=None):
    """Evaluates the board using a trained linear piece-square model.

    board: chess.Board object
    model: Pre-loaded LinearPieceSquareModel instance (optional)
    model_path: Path to saved model file (optional, used if model is None)

    Returns:
    score (int): The score of the board. Positive for white's advantage,
    negative for black's.
    done (bool): True if the game is over, False otherwise.
    """
    # Check if the game is done
    score, done = _game_over_eval(board)
    if done:
        return (score, done)

    # Load model if not provided
    if model is None:
        if model_path is None:
            # Try default path in saved_models
            import glob
            import os

            saved_models_dir = os.path.join(
                os.path.dirname(__file__), "..", "learning", "saved_models"
            )
            linear_models = glob.glob(
                os.path.join(saved_models_dir, "linear_piece_square_model_*.pkl")
            )
            if linear_models:
                # Use the most recent model
                model_path = max(linear_models, key=os.path.getmtime)
            else:
                raise ValueError(
                    "No model provided and no default model found. "
                    "Train a model using learning/31-train-piece-value-table.py"
                )
        model = load_linear_piece_square_model(model_path)

    # Extract features and predict
    from learning.feature_extraction import extract_features_piece_square

    features = extract_features_piece_square(board)
    score = int(model.predict([features])[0])

    return score, False


def random_forest_eval(board, model=None, model_path=None):
    """Evaluates the board using a trained random forest model.

    board: chess.Board object
    model: Pre-loaded RandomForestRegressor instance (optional)
    model_path: Path to saved model file (optional, used if model is None)

    Returns:
    score (int): The score of the board. Positive for white's advantage,
    negative for black's.
    done (bool): True if the game is over, False otherwise.
    """
    # Check if the game is done
    score, done = _game_over_eval(board)
    if done:
        return (score, done)

    # Load model if not provided
    if model is None:
        if model_path is None:
            # Try default path in saved_models
            import glob
            import os

            saved_models_dir = os.path.join(
                os.path.dirname(__file__), "..", "learning", "saved_models"
            )
            rf_models = glob.glob(
                os.path.join(saved_models_dir, "random_forest_chess_model_*.pkl")
            )
            if rf_models:
                # Use the most recent model
                model_path = max(rf_models, key=os.path.getmtime)
            else:
                raise ValueError(
                    "No model provided and no default model found. "
                    "Train a model using learning/30-train-simple-model.py"
                )
        model, _ = load_random_forest_model(model_path)

    # Extract features and predict
    from learning.feature_extraction import extract_features_piece_square

    features = extract_features_piece_square(board)
    score = int(model.predict([features])[0])

    return score, False


# Explicit evaluation functions for available models


def linear_1m_alpha10_no_mates_eval(board):
    """Evaluates using linear piece-square model: 1M samples, alpha=10, no mates."""
    import os

    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "learning",
        "saved_models",
        "linear_piece_square_model_1000000_alpha10_no_mates_v2.pkl",
    )
    return linear_piece_square_eval(board, model_path=model_path)


def linear_1m_alpha10_with_mates_eval(board):
    """Evaluates using linear piece-square model: 1M samples, alpha=10, with mates."""
    import os

    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "learning",
        "saved_models",
        "linear_piece_square_model_1000000_alpha10_with_mates_v2.pkl",
    )
    return linear_piece_square_eval(board, model_path=model_path)


def rf_1m_estimators500_depth25_no_mates_eval(board):
    """Evaluates using random forest model: 1M samples, 500 estimators, depth=25, no mates."""
    import os

    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "learning",
        "saved_models",
        "random_forest_chess_model_1000000_estimators500_depth25_no_mates_v2.pkl",
    )
    return random_forest_eval(board, model_path=model_path)


def rf_330k_estimators1000_unlimited_no_mates_eval(board):
    """Evaluates using random forest model: 330k samples, 1000 estimators, unlimited depth, no mates."""
    import os

    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "learning",
        "saved_models",
        "random_forest_chess_model_330000_estimators1000_depthNone_no_mates_v2.pkl",
    )
    return random_forest_eval(board, model_path=model_path)
