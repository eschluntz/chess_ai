#!/usr/bin/env python

"""
Train a linear model to learn piece-square tables from chess positions.
The model learns deltas from standard piece values for each square.
"""

import csv
import pickle
from datetime import datetime

import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for better display support
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from core.eval import PIECE_VALUES

# Import helper functions
from learning.eval_accuracy_helpers import get_train_df
from learning.feature_extraction import extract_features_piece_square
from learning.piece_value_table_viz import (
    build_feature_names,
    print_all_piece_square_tables,
)


class LinearPieceSquareModel:
    """Linear model that learns piece values based on their squares."""

    def __init__(self, alpha=1.0):
        # Use standard piece values from core.eval
        self.base_piece_values = PIECE_VALUES

        # Initialize the Ridge regression model with regularization
        self.model = Ridge(alpha=alpha)

        # Feature names for interpretation
        self.feature_names = build_feature_names()

    def fit(self, X, y):
        """Train the linear model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the model."""
        return self.model.predict(X)


def train_linear_piece_square_model(
    num_train_samples: int = 100000, alpha: float = 100.0, include_mates: bool = True
):
    """Train a linear model to learn piece-square tables."""

    # Get preprocessed training data
    train_df = get_train_df(num_train_samples, include_mates=include_mates)

    # Initialize model with regularization
    model = LinearPieceSquareModel(alpha=alpha)
    print(f"\nUsing Ridge regression with alpha={alpha} for regularization")

    # Extract features
    print("Extracting features...")
    X_train = np.array(
        [extract_features_piece_square(board) for board in train_df["board"]]
    )
    y_train = train_df["true_score"].values

    print(f"Feature shape: {X_train.shape}")

    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Train the model
    print("\nTraining linear model...")
    model.fit(X_train_split, y_train_split)

    # Evaluate
    train_predictions = model.predict(X_train_split)
    train_mae = np.mean(np.abs(train_predictions - y_train_split))
    train_rmse = np.sqrt(np.mean((train_predictions - y_train_split) ** 2))

    val_predictions = model.predict(X_val_split)
    val_mae = np.mean(np.abs(val_predictions - y_val_split))
    val_rmse = np.sqrt(np.mean((val_predictions - y_val_split) ** 2))

    print("\nTraining set performance:")
    print(f"  MAE: {train_mae:.2f} centipawns")
    print(f"  RMSE: {train_rmse:.2f} centipawns")
    print("\nValidation set performance:")
    print(f"  MAE: {val_mae:.2f} centipawns")
    print(f"  RMSE: {val_rmse:.2f} centipawns")

    # Save the model
    mates_suffix = "with_mates" if include_mates else "no_mates"
    model_path = f"learning/models/linear_piece_square_model_{num_train_samples}_alpha{alpha}_{mates_suffix}_shuff.pkl"
    print(f"\nSaving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, {
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
    }


def run_parameter_sweep():
    """Run 2D parameter sweep over alphas and sample sizes."""
    alphas = [10, 33, 100, 330]
    sample_sizes = [10_000, 33_000, 100_000, 330_000, 1_000_000]

    # Initialize CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = "learning/results/linear_piece_square_sweep_no_mate_shuff.csv"

    # Create results directory if it doesn't exist
    import os

    os.makedirs("learning/results", exist_ok=True)

    fieldnames = [
        "alpha",
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
                        "alpha": int(row["alpha"]),
                        "num_samples": int(row["num_samples"]),
                        "train_mae": float(row["train_mae"]),
                        "train_rmse": float(row["train_rmse"]),
                        "val_mae": float(row["val_mae"]),
                        "val_rmse": float(row["val_rmse"]),
                        "timestamp": row["timestamp"],
                    }
                    results.append(result)
                    completed_combinations.add((result["alpha"], result["num_samples"]))
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

    create_sweep_visualizations(results, timestamp)

    for alpha in alphas:
        for num_samples in sample_sizes:
            # Skip if this combination has already been completed
            if (alpha, num_samples) in completed_combinations:
                print(
                    f"Skipping alpha={alpha}, samples={num_samples:,} (already completed)"
                )
                continue

            print(f"\n{'=' * 80}")
            print(f"Training with alpha={alpha}, samples={num_samples:,}")
            print("=" * 80)

            try:
                _, metrics = train_linear_piece_square_model(
                    num_train_samples=num_samples, alpha=alpha, include_mates=False
                )

                result = {
                    "alpha": alpha,
                    "num_samples": num_samples,
                    "train_mae": metrics["train_mae"],
                    "train_rmse": metrics["train_rmse"],
                    "val_mae": metrics["val_mae"],
                    "val_rmse": metrics["val_rmse"],
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)
                completed_combinations.add((alpha, num_samples))

                # Append to CSV file immediately
                with open(csv_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)

                print(f"Results saved to {csv_file}")

                # Create updated visualizations after each training
                if len(results) > 0:
                    create_sweep_visualizations(results, timestamp)

            except Exception as e:
                print(f"Error training with alpha={alpha}, samples={num_samples}: {e}")
                continue

    # Final visualization summary
    print(f"\nCompleted parameter sweep with {len(results)} total results")
    return results


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
        _sweep_fig.suptitle("Linear Piece-Square Model Parameter Sweep", fontsize=16)
        plt.show(block=False)

    # Clear the axis for redrawing
    _sweep_ax.clear()

    # Combined train/val MAE plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(df["alpha"].unique())))
    for i, alpha in enumerate(df["alpha"].unique()):
        alpha_data = df[df["alpha"] == alpha]
        color = colors[i]
        _sweep_ax.plot(
            alpha_data["num_samples"],
            alpha_data["train_mae"],
            marker="o",
            linestyle="-",
            color=color,
            label=f"Train α={alpha}",
        )
        _sweep_ax.plot(
            alpha_data["num_samples"],
            alpha_data["val_mae"],
            marker="s",
            linestyle="--",
            color=color,
            label=f"Val α={alpha}",
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
    plot_file = f"learning/results/linear_piece_square_sweep_{timestamp}.png"
    _sweep_fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {plot_file}")


def show_existing_results():
    """Load and display results from existing CSV file."""
    csv_file = "learning/results/linear_piece_square_sweep_no_mate.csv"

    if not os.path.exists(csv_file):
        print(f"No existing results found at {csv_file}")
        return

    print(f"Loading existing results from {csv_file}")
    results = []

    try:
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result = {
                    "alpha": int(row["alpha"]),
                    "num_samples": int(row["num_samples"]),
                    "train_mae": float(row["train_mae"]),
                    "train_rmse": float(row["train_rmse"]),
                    "val_mae": float(row["val_mae"]),
                    "val_rmse": float(row["val_rmse"]),
                    "timestamp": row["timestamp"],
                }
                results.append(result)

        print(f"Loaded {len(results)} existing results")

        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            create_sweep_visualizations(results, timestamp)
            input("Press Enter to close...")
        else:
            print("No results to display")

    except Exception as e:
        print(f"Error loading CSV: {e}")


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            # Train single model with visualization
            print("Training single linear piece-square model...")
            model, metrics = train_linear_piece_square_model(
                num_train_samples=100_000, alpha=100.0
            )
            print_all_piece_square_tables(model)
        elif sys.argv[1] == "show":
            # Just show existing results
            show_existing_results()
        else:
            print("Usage:")
            print("  python 31-train-piece-value-table.py        # Run parameter sweep")
            print("  python 31-train-piece-value-table.py single # Train single model")
            print(
                "  python 31-train-piece-value-table.py show   # Show existing results"
            )
    else:
        # Run parameter sweep
        print("Running 2D parameter sweep...")
        results = run_parameter_sweep()
        print(f"\nCompleted sweep with {len(results)} successful runs")
