#!/usr/bin/env python

"""
Train a simple random forest model to evaluate chess positions.
"""

from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

# Import helper functions from the eval accuracy script
from eval_accuracy_helpers import (
    create_board_from_fen,
    should_skip_position,
    extract_centipawn_score
)
from feature_extraction import extract_features


def train_random_forest_model(num_train_samples: int = 50000, num_val_samples: int = 10000):
    """
    Train a random forest model on chess positions.
    """
    print("Loading Lichess chess position evaluations dataset (streaming)...")
    ds = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    
    # Skip first 10k (reserved for evaluation), then load training data
    print("\nSkipping first 10k positions (reserved for evaluation)...")
    print(f"Loading {num_train_samples} training positions...")
    train_stream = ds["train"].skip(10000)
    train_data = list(train_stream.take(num_train_samples))
    
    # Process training data
    print("Processing training data...")
    train_df = pd.DataFrame(train_data)
    train_df = train_df.dropna(subset=['fen'])
    
    # Filter positions
    train_df['should_skip'] = train_df.apply(
        lambda row: should_skip_position(row),
        axis=1
    )
    train_df = train_df[~train_df['should_skip']].copy()
    
    # Extract scores and create boards
    train_df['true_score'] = train_df.apply(extract_centipawn_score, axis=1)
    train_df['board'] = train_df['fen'].apply(create_board_from_fen)
    
    print(f"Extracting features from {len(train_df)} training positions...")
    # Extract features
    X_train = np.array([extract_features(board) for board in train_df['board']])
    y_train = train_df['true_score'].values
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    val_predictions = model.predict(X_val_split)
    val_mae = np.mean(np.abs(val_predictions - y_val_split))
    print(f"Validation MAE: {val_mae:.2f} centipawns")
    
    # Feature importance
    print("\nTop 10 most important features:")
    feature_names = [
        "white_pawns", "white_knights", "white_bishops", "white_rooks", "white_queens", "white_kings",
        "black_pawns", "black_knights", "black_bishops", "black_rooks", "black_queens", "black_kings",
        "center_control", "doubled_pawns", "king_center_distance", "turn_to_move",
        "white_castle_kingside", "white_castle_queenside", "black_castle_kingside", "black_castle_queenside"
    ]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(feature_names))):
        print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.3f}")
    
    # Load validation data (from training set, after evaluation and training data)
    print(f"\nLoading {num_val_samples} validation positions...")
    val_stream = ds["train"].skip(10000 + num_train_samples)
    val_data = list(val_stream.take(num_val_samples))
    
    # Process validation data
    print("Processing validation data...")
    val_df = pd.DataFrame(val_data)
    val_df = val_df.dropna(subset=['fen'])
    
    val_df['should_skip'] = val_df.apply(
        lambda row: should_skip_position(row),
        axis=1
    )
    val_df = val_df[~val_df['should_skip']].copy()
    
    val_df['true_score'] = val_df.apply(extract_centipawn_score, axis=1)
    val_df['board'] = val_df['fen'].apply(create_board_from_fen)
    
    print(f"Extracting features from {len(val_df)} validation positions...")
    X_val = np.array([extract_features(board) for board in val_df['board']])
    y_val = val_df['true_score'].values
    
    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_mae = np.mean(np.abs(val_predictions - y_val))
    print(f"\nFinal validation MAE: {val_mae:.2f} centipawns")
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), "random_forest_chess_model.pkl")
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")
    
    return model


if __name__ == "__main__":
    train_random_forest_model()