#!/usr/bin/env python

"""
Evaluate the accuracy of chess evaluation functions using the Lichess position evaluations dataset.

Usage:
    20-eval-accuracy.py [--num-samples=<n>] [--graphs]
    20-eval-accuracy.py -h | --help

Options:
    --num-samples=<n>   Number of positions to evaluate [default: 10000]
    --graphs            Show scatter plots
    -h --help           Show this screen
"""

# Import neural network loader
import importlib.util
import os
import pickle
from collections.abc import Callable

import chess
from docopt import docopt

from core.eval import piece_position_eval, piece_value_eval

# Import helper functions
from learning.eval_accuracy_helpers import get_eval_df
from learning.eval_common import (
    create_combined_scatter_plot,
    evaluate_all_functions,
    print_results_summary,
)
from learning.feature_extraction import (
    extract_features_basic,
    extract_features_piece_square,
)

# Import LinearPieceSquareModel class for pickle loading
spec_31 = importlib.util.spec_from_file_location(
    "train_piece_value_table",
    os.path.join(os.path.dirname(__file__), "31-train-piece-value-table.py")
)
train_piece_value_table = importlib.util.module_from_spec(spec_31)
spec_31.loader.exec_module(train_piece_value_table)
LinearPieceSquareModel = train_piece_value_table.LinearPieceSquareModel

spec = importlib.util.spec_from_file_location(
    "train_neural_network",
    os.path.join(os.path.dirname(__file__), "40-train-neural-network.py"),
)
train_neural_network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_neural_network)
load_neural_network_eval = train_neural_network.load_neural_network_eval


def load_random_forest_eval(
    model_path: str,
) -> Callable[[chess.Board], tuple[int, bool]]:
    """Load the trained random forest model and return an evaluation function."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old and new model formats
    if isinstance(data, dict):
        model = data["model"]
        feature_type = data["feature_type"]
    else:
        # Old format - assume basic features
        model = data
        feature_type = "basic"

    # Choose the appropriate feature extractor
    if feature_type == "piece_square":
        extract_features = extract_features_piece_square
    else:
        extract_features = extract_features_basic

    def random_forest_eval(board: chess.Board) -> tuple[int, bool]:
        """Evaluate a position using the trained random forest model."""
        features = extract_features(board)
        score = int(model.predict([features])[0])
        return score, False

    return random_forest_eval


def load_random_forest_models(model_dir: str) -> list[tuple[Callable, str]]:
    """Load all Random Forest models from the model directory."""
    models = []
    rf_model_files = sorted(
        [
            f
            for f in os.listdir(model_dir)
            if f.startswith("random_forest_chess_model_") and f.endswith(".pkl")
        ]
    )

    if rf_model_files:
        print(f"\nFound {len(rf_model_files)} Random Forest models:")
        for model_file in rf_model_files:
            # Extract info from filename
            # Format: random_forest_chess_model_<size>_<feature_type>.pkl
            parts = (
                model_file.replace("random_forest_chess_model_", "")
                .replace(".pkl", "")
                .split("_")
            )
            if len(parts) >= 2:
                # New format with feature type
                model_name = f"rf_{parts[0]}_{parts[1]}"
            else:
                # Old format
                model_name = f"rf_model_{parts[0]}"

            model_path = os.path.join(model_dir, model_file)
            print(f"  Loading {model_file}...")
            rf_eval = load_random_forest_eval(model_path)
            models.append((rf_eval, model_name))
    else:
        print("No Random Forest models found.")

    return models


def load_linear_piece_square_eval(
    model_path: str,
) -> Callable[[chess.Board], tuple[int, bool]]:
    """Load the trained linear piece-square model and return an evaluation function."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    def linear_piece_square_eval(board: chess.Board) -> tuple[int, bool]:
        """Evaluate a position using the trained linear piece-square model."""
        features = extract_features_piece_square(board)
        score = int(model.predict([features])[0])
        return score, False

    return linear_piece_square_eval


def load_linear_models(model_dir: str) -> list[tuple[Callable, str]]:
    """Load all Linear Piece-Square models from the model directory."""
    models = []
    linear_model_files = sorted(
        [
            f
            for f in os.listdir(model_dir)
            if f.startswith("linear_piece_square_model_") and f.endswith(".pkl")
        ]
    )

    if linear_model_files:
        print(f"\nFound {len(linear_model_files)} Linear Piece-Square models:")
        for model_file in linear_model_files:
            # Extract info from filename
            # Format: linear_piece_square_model_<size>_alpha<alpha>_<mates_suffix>_v2.pkl
            parts = (
                model_file.replace("linear_piece_square_model_", "")
                .replace(".pkl", "")
                .split("_")
            )
            if len(parts) >= 2:
                # Include more parameters in the name to make it unique
                model_name = f"linear_{'_'.join(parts[:3])}"  # size_alpha_mates
            else:
                model_name = f"linear_{parts[0]}"

            model_path = os.path.join(model_dir, model_file)
            print(f"  Loading {model_file}...")
            linear_eval = load_linear_piece_square_eval(model_path)
            models.append((linear_eval, model_name))
    else:
        print("No Linear Piece-Square models found.")

    return models


def load_neural_network_models(model_dir: str) -> list[tuple[Callable, str]]:
    """Load all Neural Network models from the model directory."""
    models = []
    nn_model_files = sorted(
        [
            f
            for f in os.listdir(model_dir)
            if f.startswith("neural_network_chess_model_") and f.endswith(".pkl")
        ]
    )

    if nn_model_files:
        print(f"\nFound {len(nn_model_files)} Neural Network models:")
        for model_file in nn_model_files:
            # Extract info from filename
            # Format: neural_network_chess_model_<size>_<suffix>.pkl
            parts = (
                model_file.replace("neural_network_chess_model_", "")
                .replace(".pkl", "")
                .split("_")
            )
            if len(parts) >= 2:
                model_name = f"nn_{parts[0]}_{parts[1]}"
            else:
                model_name = f"nn_{parts[0]}"

            model_path = os.path.join(model_dir, model_file)
            print(f"  Loading {model_file}...")
            nn_eval = load_neural_network_eval(model_path)
            models.append((nn_eval, model_name))
    else:
        print("No Neural Network models found.")

    return models


def main(num_samples: int = 5000, graphs: bool = False) -> None:
    """
    Evaluate chess evaluation functions.
    Uses the first positions from the dataset for evaluation.

    Args:
        num_samples: Number of positions to evaluate
        graphs: Whether to show scatter plots
    """
    # Define evaluation functions
    baseline_name = "piece_value_eval"

    # Start with hardcoded functions
    all_functions = [
        (piece_value_eval, "piece_value_eval"),
        (piece_position_eval, "piece_position_eval"),
    ]

    # Load all trained models
    model_dir = os.path.join(os.path.dirname(__file__), "saved_models")

    # Load Random Forest models
    all_functions.extend(load_random_forest_models(model_dir))

    # Load Linear Piece-Square models
    all_functions.extend(load_linear_models(model_dir))

    # Load Neural Network models
    all_functions.extend(load_neural_network_models(model_dir))

    if len(all_functions) == 2:  # Only baseline functions
        print("\nNo trained models found. Evaluating only hardcoded functions.")

    # Get evaluation data from the canonical shuffled dataset
    print(f"\nGetting {num_samples} evaluation positions from shuffled dataset...")
    processed_df = get_eval_df(num_samples)

    # Evaluate all functions on the same data
    print("\n" + "=" * 50)
    print("Evaluating all functions...")
    all_results = evaluate_all_functions(all_functions, processed_df)

    # Print results
    print_results_summary(all_results, baseline_name)

    # Create combined scatter plot
    if graphs:
        print("\nCreating scatter plots...")
        create_combined_scatter_plot(all_results, baseline_name)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    num_samples = int(arguments["--num-samples"])
    graphs = arguments["--graphs"]

    main(num_samples, graphs)
