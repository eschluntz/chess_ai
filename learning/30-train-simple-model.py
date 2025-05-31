#!/usr/bin/env python

"""
Train a random forest model to evaluate chess positions using piece-square features.

Usage:
    30-train-simple-model.py
    30-train-simple-model.py single
    30-train-simple-model.py --help

Commands:
    (no command)    Run parameter sweep
    single          Train single model

Options:
    -h --help       Show this screen
"""

import csv
import os
import pickle
from datetime import datetime

import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for better display support
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.ensemble import RandomForestRegressor

# Import helper functions
from learning.eval_accuracy_helpers import get_eval_df, get_train_df
from learning.feature_extraction import extract_features_piece_square


def train_random_forest_model(
    num_train_samples: int = 50000,
    include_mates: bool = True,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
):
    """
    Train a random forest model on chess positions using piece-square features.

    Args:
        num_train_samples: Number of training samples to use
        include_mates: Whether to include mate positions
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node

    Returns:
        tuple: (model, metrics_dict) where metrics_dict contains train and validation errors
    """
    # Get preprocessed training data
    train_df = get_train_df(num_train_samples, include_mates=include_mates)

    # Extract features using piece-square extraction
    print("Extracting piece-square features...")
    X_train = np.array(
        [extract_features_piece_square(board) for board in train_df["board"]]
    )
    y_train = train_df["true_score"].values
    print(f"Feature shape: {X_train.shape}")

    # Train the model
    print("\nTraining Random Forest model with params:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: {min_samples_split}")
    print(f"  min_samples_leaf: {min_samples_leaf}")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Get validation data using get_eval_df
    print("Getting validation data...")
    val_df = get_eval_df(include_mates=include_mates)
    X_val = np.array(
        [extract_features_piece_square(board) for board in val_df["board"]]
    )
    y_val = val_df["true_score"].values

    # Evaluate
    train_predictions = model.predict(X_train)
    train_mae = np.mean(np.abs(train_predictions - y_train))
    train_rmse = np.sqrt(np.mean((train_predictions - y_train) ** 2))

    val_predictions = model.predict(X_val)
    val_mae = np.mean(np.abs(val_predictions - y_val))
    val_rmse = np.sqrt(np.mean((val_predictions - y_val) ** 2))

    print("\nTraining set performance:")
    print(f"  MAE: {train_mae:.2f} centipawns")
    print(f"  RMSE: {train_rmse:.2f} centipawns")
    print("\nValidation set performance:")
    print(f"  MAE: {val_mae:.2f} centipawns")
    print(f"  RMSE: {val_rmse:.2f} centipawns")

    # Save the model
    mates_suffix = "with_mates" if include_mates else "no_mates"
    model_filename = f"random_forest_chess_model_{num_train_samples}_estimators{n_estimators}_depth{max_depth}_{mates_suffix}_v2.pkl"
    model_path = os.path.join(os.path.dirname(__file__), "saved_models", model_filename)
    print(f"\nSaving model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_type": "piece_square"}, f)

    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")

    # Return model and metrics
    return model, {
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
    }


# Global figure for persistent plotting
_sweep_fig = None
_sweep_ax = None


def create_sweep_visualizations(results, timestamp):
    """Create line plots for the parameter sweep results."""
    global _sweep_fig, _sweep_ax

    if not results:
        return

    df = pd.DataFrame(results)

    # Create figure only once
    if _sweep_fig is None:
        plt.ion()  # Enable interactive mode
        plt.style.use("default")
        _sweep_fig, _sweep_ax = plt.subplots(1, 1, figsize=(12, 8))
        _sweep_fig.suptitle("Random Forest Parameter Sweep", fontsize=16)
        plt.show(block=False)

    # Clear the axis for redrawing
    _sweep_ax.clear()

    # Combined train/val MAE plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(df["name"].unique())))
    for i, name in enumerate(df["name"].unique()):
        name_data = df[df["name"] == name]
        color = colors[i]
        _sweep_ax.plot(
            name_data["num_samples"],
            name_data["train_mae"],
            marker="o",
            linestyle="-",
            color=color,
            label=f"Train {name}",
        )
        _sweep_ax.plot(
            name_data["num_samples"],
            name_data["val_mae"],
            marker="s",
            linestyle="--",
            color=color,
            label=f"Val {name}",
        )
    _sweep_ax.set_xlabel("Number of Samples")
    _sweep_ax.set_ylabel("MAE")
    _sweep_ax.set_xscale("log")
    _sweep_ax.legend()
    _sweep_ax.grid(True, alpha=0.3)
    _sweep_ax.set_title("Train vs Validation MAE")

    # Update the display
    _sweep_fig.canvas.draw()
    _sweep_fig.canvas.flush_events()

    # Save the plot
    plot_file = f"learning/results/random_forest_sweep_{timestamp}.png"
    _sweep_fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {plot_file}")


def run_parameter_sweep():
    """Run 2D parameter sweep over training sizes and random forest hyperparameters."""
    include_mates = False  # Set this to control whether to include mate positions

    # Training sizes to test (same range as linear model)
    training_sizes = [
        10_000,
        33_000,
        100_000,
        330_000,
        1_000_000,
        3_300_000,
        10_000_000,
    ]

    # Parameter combinations to test
    experiments = [
        # # Small model (fast, less accurate)
        # {
        #     "name": "small",
        #     "n_estimators": 10,
        #     "max_depth": 5,
        #     "min_samples_split": 50,
        #     "min_samples_leaf": 25,
        # },
        # # Medium-small model
        # {
        #     "name": "medium_small",
        #     "n_estimators": 30,
        #     "max_depth": 10,
        #     "min_samples_split": 20,
        #     "min_samples_leaf": 10,
        # },
        # Medium model (baseline)
        {
            "name": "medium",
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
        },
        # Large model
        {
            "name": "large",
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        # Very large model
        {
            "name": "very_large",
            "n_estimators": 500,
            "max_depth": 25,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        # Extra large model (maximum complexity)
        {
            "name": "extra_large",
            "n_estimators": 1000,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
    ]

    # Initialize CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mates_suffix = "with_mates" if include_mates else "no_mates"
    csv_file = f"learning/results/random_forest_sweep_{mates_suffix}_v2.csv"

    # Create results directory if it doesn't exist
    os.makedirs("learning/results", exist_ok=True)

    fieldnames = [
        "name",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "num_samples",
        "train_mae",
        "train_rmse",
        "val_mae",
        "val_rmse",
        "timestamp",
    ]

    # Load existing results if CSV exists
    results = []
    completed_combinations = set()

    if os.path.exists(csv_file):
        print(f"Loading existing results from {csv_file}")
        try:
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string values back to appropriate types
                    result = {
                        "name": row["name"],
                        "n_estimators": int(row["n_estimators"]),
                        "max_depth": int(row["max_depth"])
                        if row["max_depth"] != "None"
                        else None,
                        "min_samples_split": int(row["min_samples_split"]),
                        "min_samples_leaf": int(row["min_samples_leaf"]),
                        "num_samples": int(row["num_samples"]),
                        "train_mae": float(row["train_mae"]),
                        "train_rmse": float(row["train_rmse"]),
                        "val_mae": float(row["val_mae"]),
                        "val_rmse": float(row["val_rmse"]),
                        "timestamp": row["timestamp"],
                    }
                    results.append(result)
                    # Create key for completed combinations
                    key = (result["name"], result["num_samples"])
                    completed_combinations.add(key)
            print(f"Loaded {len(results)} existing results")
        except Exception as e:
            print(f"Error loading existing CSV: {e}")
            results = []
            completed_combinations = set()
    else:
        # Create new CSV file with header
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    total_experiments = len(training_sizes) * len(experiments)
    experiment_count = 0

    for training_size in training_sizes:
        for params in experiments:
            experiment_count += 1

            # Check if this combination has already been completed
            key = (params["name"], training_size)
            if key in completed_combinations:
                print(
                    f"Skipping experiment {experiment_count}/{total_experiments} (already completed): {params['name']} model with {training_size:,} samples"
                )
                continue

            print("\n" + "=" * 80)
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(
                f"Training '{params['name']}' random forest model with {training_size:,} samples"
            )
            print(f"Parameters: {params}")
            print("=" * 80)

            try:
                # Extract parameters without the name field
                model_params = {k: v for k, v in params.items() if k != "name"}
                _, metrics = train_random_forest_model(
                    num_train_samples=training_size,
                    include_mates=include_mates,
                    **model_params,
                )

                result = {
                    "name": params["name"],
                    "n_estimators": params["n_estimators"],
                    "max_depth": params["max_depth"],
                    "min_samples_split": params["min_samples_split"],
                    "min_samples_leaf": params["min_samples_leaf"],
                    "num_samples": training_size,
                    "train_mae": metrics["train_mae"],
                    "train_rmse": metrics["train_rmse"],
                    "val_mae": metrics["val_mae"],
                    "val_rmse": metrics["val_rmse"],
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)
                completed_combinations.add(key)

                # Append to CSV file immediately
                with open(csv_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)

                print(f"Results saved to {csv_file}")

                # Create updated visualizations after each training
                if len(results) > 0:
                    create_sweep_visualizations(results, timestamp)

            except Exception as e:
                print(f"Error training with params {params}, size {training_size}: {e}")
                continue

    # Final summary
    print(f"\nCompleted parameter sweep with {len(results)} total results")
    return results


if __name__ == "__main__":
    arguments = docopt(__doc__)

    if arguments["single"]:
        # Train single model
        print("Training single random forest model...")
        model, metrics = train_random_forest_model(
            num_train_samples=100_000, include_mates=False
        )
    else:
        # Run parameter sweep
        print("Running parameter sweep...")
        results = run_parameter_sweep()
        print(f"\nCompleted sweep with {len(results)} successful runs")
