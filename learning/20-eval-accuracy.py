#!/usr/bin/env python

"""
Evaluate the accuracy of chess evaluation functions using the Lichess position evaluations dataset.
Always uses the first 10,000 positions for evaluation.
"""

from datasets import load_dataset
import os
import pickle
from collections.abc import Callable
import chess

from core.eval import piece_value_eval, piece_position_eval

# Import helper functions
from learning.eval_accuracy_helpers import process_dataset_batch
from learning.eval_common import (
    evaluate_all_functions,
    create_combined_scatter_plot,
    print_results_summary,
)
from learning.feature_extraction import (
    extract_features_basic,
    extract_features_piece_square,
)

# Import neural network loader
import importlib.util

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
    # Import the module containing LinearPieceSquareModel
    import sys
    import importlib.util

    # Load the module
    module_path = os.path.join(
        os.path.dirname(__file__), "31-train-piece-value-table.py"
    )
    spec = importlib.util.spec_from_file_location("__main__", module_path)
    module = importlib.util.module_from_spec(spec)

    # The model was saved when running as __main__, so we need to make the class available there
    old_main = sys.modules.get("__main__")
    sys.modules["__main__"] = module

    try:
        # Execute the module to define the class
        spec.loader.exec_module(module)

        # Now load the pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    finally:
        # Restore the original __main__
        if old_main is not None:
            sys.modules["__main__"] = old_main

    def linear_piece_square_eval(board: chess.Board) -> tuple[int, bool]:
        """Evaluate a position using the trained linear piece-square model."""
        features = model.extract_features(board)
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
            # Format: linear_piece_square_model_<size>.pkl
            parts = (
                model_file.replace("linear_piece_square_model_", "")
                .replace(".pkl", "")
                .split("_")
            )
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


def main(num_samples: int = 5000) -> None:
    """
    Evaluate chess evaluation functions.
    Uses the first positions from the dataset for evaluation.

    Args:
        num_samples: Number of positions to evaluate
    """
    # Define evaluation functions
    baseline_name = "piece_value_eval"

    # Start with hardcoded functions
    all_functions = [
        (piece_value_eval, "piece_value_eval"),
        (piece_position_eval, "piece_position_eval"),
    ]

    # Load all trained models
    model_dir = os.path.join(os.path.dirname(__file__), "models")

    # Load Random Forest models
    all_functions.extend(load_random_forest_models(model_dir))

    # Load Linear Piece-Square models
    # all_functions.extend(load_linear_models(model_dir))

    # Load Neural Network models
    all_functions.extend(load_neural_network_models(model_dir))

    if len(all_functions) == 2:  # Only baseline functions
        print("\nNo trained models found. Evaluating only hardcoded functions.")

    # Load the streaming dataset
    print("Loading Lichess chess position evaluations dataset (streaming)...")
    ds_full = load_dataset("Lichess/chess-position-evaluations", streaming=True)

    # Process the first N positions (reserved for evaluation)
    print(f"\nProcessing first {num_samples} positions (evaluation set)...")
    dataset_stream = ds_full["train"]

    processed_df, stats = process_dataset_batch(
        dataset_stream.take(num_samples), num_samples
    )

    # Print data stats
    print("\nData statistics:")
    print(f"  Total positions loaded: {stats['total_loaded']}")
    print(f"  Total positions after filtering: {len(processed_df)}")
    print(f"  Mate positions: {stats['mate_positions']}")
    print(f"  Non-mate positions: {len(processed_df) - stats['mate_positions']}")
    print(f"  Total filtered out: {stats['total_filtered']}")

    # Evaluate all functions on the same data
    print("\n" + "=" * 50)
    print("Evaluating all functions...")
    all_results = evaluate_all_functions(all_functions, processed_df)

    # Print results
    print_results_summary(all_results, baseline_name)

    # Create combined scatter plot
    print("\nCreating scatter plots...")
    create_combined_scatter_plot(all_results, baseline_name)


if __name__ == "__main__":
    main(10_000)
