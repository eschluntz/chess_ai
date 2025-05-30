#!/usr/bin/env python

"""
Train a simple random forest model to evaluate chess positions.
"""

import os
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Import helper functions from the eval accuracy script
from learning.eval_accuracy_helpers import (
    create_board_from_fen,
    extract_centipawn_score,
    should_skip_position,
)
from learning.feature_extraction import (
    extract_features_basic,
    extract_features_piece_square,
)


def train_random_forest_model(
    num_train_samples: int = 50000,
    num_val_samples: int = 10000,
    feature_type: str = "basic",
    model_suffix: str = "",
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
):
    """
    Train a random forest model on chess positions.

    Args:
        num_train_samples: Number of training samples to use
        num_val_samples: Number of validation samples to use
        feature_type: 'basic' or 'piece_square' feature extraction
        model_suffix: Additional suffix for model filename
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node

    Returns:
        tuple: (model, metrics_dict) where metrics_dict contains train and validation errors
    """
    print("Loading Lichess chess position evaluations dataset (streaming)...")
    ds = load_dataset("Lichess/chess-position-evaluations", streaming=True)

    # Skip first 10k (reserved for evaluation), then load training data
    print("\nData split:")
    print("  Positions 0-9,999: Reserved for evaluation (skipped)")
    print(f"  Positions 10,000-{10000 + num_train_samples - 1:,}: Training data")
    print(
        f"  Positions {10000 + num_train_samples:,}-"
        f"{10000 + num_train_samples + num_val_samples - 1:,}: Validation data"
    )

    print(f"\nLoading {num_train_samples:,} training positions...")
    train_stream = ds["train"].skip(10000)
    train_data = list(train_stream.take(num_train_samples))

    # Process training data
    print("Processing training data...")
    train_df = pd.DataFrame(train_data)
    train_df = train_df.dropna(subset=["fen"])

    # Filter positions
    train_df["should_skip"] = train_df.apply(
        lambda row: should_skip_position(row), axis=1
    )
    train_df = train_df[~train_df["should_skip"]].copy()

    # Extract scores and create boards
    train_df["true_score"] = train_df.apply(extract_centipawn_score, axis=1)
    train_df["board"] = train_df["fen"].apply(create_board_from_fen)

    # Choose feature extractor
    if feature_type == "piece_square":
        extract_features = extract_features_piece_square
        print("Using piece-square feature extraction (768+ features)")
    else:
        extract_features = extract_features_basic
        print("Using basic feature extraction (20 features)")

    print(f"Extracting features from {len(train_df)} training positions...")
    # Extract features
    X_train = np.array([extract_features(board) for board in train_df["board"]])
    y_train = train_df["true_score"].values
    print(f"Feature shape: {X_train.shape}")

    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

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
    model.fit(X_train_split, y_train_split)

    # Evaluate on training set (to check for overfitting)
    train_predictions = model.predict(X_train_split)
    train_mae = np.mean(np.abs(train_predictions - y_train_split))
    train_rmse = np.sqrt(np.mean((train_predictions - y_train_split) ** 2))

    # Evaluate on validation set
    val_predictions = model.predict(X_val_split)
    val_mae = np.mean(np.abs(val_predictions - y_val_split))
    val_rmse = np.sqrt(np.mean((val_predictions - y_val_split) ** 2))

    print("\nTraining set performance:")
    print(f"  MAE: {train_mae:.2f} centipawns")
    print(f"  RMSE: {train_rmse:.2f} centipawns")
    print("\nValidation set performance:")
    print(f"  MAE: {val_mae:.2f} centipawns")
    print(f"  RMSE: {val_rmse:.2f} centipawns")
    print("\nOverfitting check:")
    print(f"  MAE difference (val - train): {val_mae - train_mae:.2f} centipawns")
    print(f"  RMSE difference (val - train): {val_rmse - train_rmse:.2f} centipawns")

    # Load validation data (from training set, after evaluation and training data)
    print(f"\nLoading {num_val_samples} validation positions...")
    val_stream = ds["train"].skip(10000 + num_train_samples)
    val_data = list(val_stream.take(num_val_samples))

    # Process validation data
    print("Processing validation data...")
    val_df = pd.DataFrame(val_data)
    val_df = val_df.dropna(subset=["fen"])

    val_df["should_skip"] = val_df.apply(lambda row: should_skip_position(row), axis=1)
    val_df = val_df[~val_df["should_skip"]].copy()

    val_df["true_score"] = val_df.apply(extract_centipawn_score, axis=1)
    val_df["board"] = val_df["fen"].apply(create_board_from_fen)

    print(f"Extracting features from {len(val_df)} validation positions...")
    X_val = np.array([extract_features(board) for board in val_df["board"]])
    y_val = val_df["true_score"].values

    # Evaluate on independent validation set
    final_val_predictions = model.predict(X_val)
    final_val_mae = np.mean(np.abs(final_val_predictions - y_val))
    final_val_rmse = np.sqrt(np.mean((final_val_predictions - y_val) ** 2))

    # Also evaluate on full training set for comparison
    full_train_predictions = model.predict(X_train)
    full_train_mae = np.mean(np.abs(full_train_predictions - y_train))
    full_train_rmse = np.sqrt(np.mean((full_train_predictions - y_train) ** 2))

    print("\nFinal evaluation on full datasets:")
    print(f"  Full training set MAE: {full_train_mae:.2f} centipawns")
    print(f"  Full training set RMSE: {full_train_rmse:.2f} centipawns")
    print(f"  Independent validation MAE: {final_val_mae:.2f} centipawns")
    print(f"  Independent validation RMSE: {final_val_rmse:.2f} centipawns")
    print(
        f"  Generalization gap (MAE): {final_val_mae - full_train_mae:.2f} centipawns"
    )

    # Save the model with training samples and feature type in filename
    if model_suffix:
        model_filename = f"random_forest_chess_model_{num_train_samples}_{feature_type}_{model_suffix}.pkl"
    else:
        model_filename = (
            f"random_forest_chess_model_{num_train_samples}_{feature_type}.pkl"
        )
    model_path = os.path.join(os.path.dirname(__file__), "models", model_filename)
    print(f"\nSaving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_type": feature_type}, f)

    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")

    # Return model and metrics
    metrics = {
        "num_train_samples": num_train_samples,
        "feature_type": feature_type,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "train_mae": full_train_mae,
        "val_mae": final_val_mae,
    }

    return model, metrics


if __name__ == "__main__":
    # Fixed training size
    training_size = 100_000
    feature_type = "piece_square"

    # Linear sweep - all parameters scale together
    experiments = [
        # Small model (fast, less accurate)
        {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_split": 50,
            "min_samples_leaf": 25,
        },
        # Medium-small model
        {
            "n_estimators": 30,
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
        },
        # Medium model (baseline)
        {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
        },
        # Large model
        {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        # Very large model (slow, potentially overfit)
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
    ]

    # List to store results for CSV
    results = []

    csv_path = os.path.join(os.path.dirname(__file__), "data", "rf_parameter_sweep_results.csv")

    for i, params in enumerate(experiments, 1):
        print("\n" + "=" * 80)
        print(f"Experiment {i}/{len(experiments)}")
        print(f"Training {feature_type} model with {training_size:,} samples")
        print(f"Parameters: {params}")
        print("=" * 80)

        model, metrics = train_random_forest_model(
            num_train_samples=training_size, feature_type=feature_type, **params
        )
        results.append(metrics)

        # Save results to CSV after each model
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

    print(f"\n\nAll training complete! Final results saved to: {csv_path}")
