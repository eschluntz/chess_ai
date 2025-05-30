#!/usr/bin/env python

"""
Train a linear model to learn piece-square tables from chess positions.
The model learns deltas from standard piece values for each square.
"""

from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import pickle
import os
import chess

# Import helper functions
from learning.eval_accuracy_helpers import (
    create_board_from_fen,
    should_skip_position,
    extract_centipawn_score
)
from core.eval import PIECE_VALUES


class LinearPieceSquareModel:
    """Linear model that learns piece values based on their squares."""
    
    def __init__(self, alpha=1.0):
        # Use standard piece values from core.eval
        self.base_piece_values = PIECE_VALUES
        
        # Initialize the Ridge regression model with regularization
        self.model = Ridge(alpha=alpha)
        
        # Feature names for interpretation
        self.feature_names = []
        self._build_feature_names()
        
    def _build_feature_names(self):
        """Build human-readable feature names."""
        piece_names = {
            chess.PAWN: "Pawn",
            chess.KNIGHT: "Knight", 
            chess.BISHOP: "Bishop",
            chess.ROOK: "Rook",
            chess.QUEEN: "Queen",
            chess.KING: "King"
        }
        
        # Piece-square features
        for piece_type, piece_name in piece_names.items():
            for square in range(64):
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                square_name = f"{chr(ord('a') + file)}{rank + 1}"
                self.feature_names.append(f"White_{piece_name}_{square_name}")
                self.feature_names.append(f"Black_{piece_name}_{square_name}")
        
        # Additional features
        self.feature_names.extend([
            "Turn_White",
            "Castling_White_Kingside",
            "Castling_White_Queenside", 
            "Castling_Black_Kingside",
            "Castling_Black_Queenside"
        ])
    
    def extract_features(self, board: chess.Board) -> np.ndarray:
        """
        Extract features for linear piece-square model.
        For each piece on each square, we create a feature.
        """
        # 6 piece types * 64 squares * 2 colors = 768 piece features
        # + 5 additional features (turn, castling rights)
        features = np.zeros(773, dtype=np.float32)
        
        # Extract piece positions
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        
        for piece_idx, piece_type in enumerate(piece_types):
            # White pieces
            for square in board.pieces(piece_type, chess.WHITE):
                feature_idx = piece_idx * 128 + square * 2
                features[feature_idx] = 1
            
            # Black pieces  
            for square in board.pieces(piece_type, chess.BLACK):
                feature_idx = piece_idx * 128 + square * 2 + 1
                features[feature_idx] = 1
        
        # Additional features
        base_idx = 768
        features[base_idx] = 1 if board.turn == chess.WHITE else 0
        features[base_idx + 1] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        features[base_idx + 2] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        features[base_idx + 3] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        features[base_idx + 4] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
        
        return features
    
    def fit(self, X, y):
        """Train the linear model."""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Make predictions using the model."""
        return self.model.predict(X)
    
    def get_piece_square_tables(self):
        """
        Extract learned piece-square tables from the model coefficients.
        Returns a dictionary mapping piece type to 8x8 array of values.
        """
        coefficients = self.model.coef_
        piece_square_tables = {}
        
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        piece_names = {
            chess.PAWN: "Pawn",
            chess.KNIGHT: "Knight",
            chess.BISHOP: "Bishop", 
            chess.ROOK: "Rook",
            chess.QUEEN: "Queen",
            chess.KING: "King"
        }
        
        for piece_idx, piece_type in enumerate(piece_types):
            # Initialize tables for white and black
            white_table = np.zeros((8, 8))
            black_table = np.zeros((8, 8))
            
            for square in range(64):
                rank = square // 8
                file = square % 8
                
                # Get coefficient for white piece on this square
                white_coef_idx = piece_idx * 128 + square * 2
                white_value = self.base_piece_values[piece_type] + coefficients[white_coef_idx]
                white_table[rank, file] = white_value
                
                # Get coefficient for black piece on this square  
                black_coef_idx = piece_idx * 128 + square * 2 + 1
                # Note: Black coefficients are negative in the model
                black_value = -self.base_piece_values[piece_type] + coefficients[black_coef_idx]
                black_table[rank, file] = black_value
            
            piece_square_tables[piece_names[piece_type]] = {
                'white': white_table,
                'black': black_table,
                'deltas_white': white_table - self.base_piece_values[piece_type],
                'deltas_black': black_table + self.base_piece_values[piece_type]
            }
        
        # Add other feature coefficients
        base_idx = 768
        other_features = {
            'turn_bonus': coefficients[base_idx],
            'white_kingside_castling': coefficients[base_idx + 1],
            'white_queenside_castling': coefficients[base_idx + 2],
            'black_kingside_castling': coefficients[base_idx + 3],
            'black_queenside_castling': coefficients[base_idx + 4],
            'intercept': self.model.intercept_
        }
        
        return piece_square_tables, other_features
    
    def visualize_piece_square_table(self, piece_name, color='white', show_deltas=True):
        """
        Print ASCII visualization of a piece-square table.
        """
        tables, other_features = self.get_piece_square_tables()
        
        if show_deltas:
            table = tables[piece_name][f'deltas_{color}']
            print(f"\n{piece_name} Position Value Deltas ({color}):")
        else:
            table = tables[piece_name][color]
            print(f"\n{piece_name} Total Position Values ({color}):")
        
        print("    a    b    c    d    e    f    g    h")
        print("  +----+----+----+----+----+----+----+----+")
        
        for rank in range(7, -1, -1):  # Start from rank 8 down to rank 1
            print(f"{rank+1} |", end="")
            for file in range(8):
                value = table[rank, file]
                # Format with sign for deltas, limit to 3 digits
                if show_deltas and value >= 0:
                    print(f" {value:+3.0f}|", end="")
                else:
                    print(f" {value:3.0f}|", end="")
            print(f" {rank+1}")
            if rank > 0:
                print("  +----+----+----+----+----+----+----+----+")
        
        print("  +----+----+----+----+----+----+----+----+")
        print("    a    b    c    d    e    f    g    h")
        
        # Print statistics about the values
        if show_deltas:
            abs_values = np.abs(table.flatten())
            print(f"  Average magnitude: {np.mean(abs_values):.1f} cp")
            print(f"  Max magnitude: {np.max(abs_values):.1f} cp")
            print(f"  Std deviation: {np.std(table.flatten()):.1f} cp")


def train_linear_piece_square_model(num_train_samples: int = 100000, num_val_samples: int = 20000, alpha: float = 100.0):
    """Train a linear model to learn piece-square tables."""
    
    print("Loading Lichess chess position evaluations dataset...")
    ds = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    
    # Skip first 10k positions (reserved for evaluation)
    print(f"\nLoading {num_train_samples:,} training positions...")
    train_stream = ds["train"].skip(10000)
    train_data = list(train_stream.take(num_train_samples))
    
    # Process training data
    print("Processing training data...")
    train_df = pd.DataFrame(train_data)
    train_df = train_df.dropna(subset=['fen'])
    
    # Filter positions
    train_df['should_skip'] = train_df.apply(lambda row: should_skip_position(row), axis=1)
    train_df = train_df[~train_df['should_skip']].copy()
    
    # Extract scores and create boards
    train_df['true_score'] = train_df.apply(extract_centipawn_score, axis=1)
    train_df['board'] = train_df['fen'].apply(create_board_from_fen)
    
    print(f"Filtered to {len(train_df)} valid positions")
    
    # Initialize model with regularization
    model = LinearPieceSquareModel(alpha=alpha)
    print(f"\nUsing Ridge regression with alpha={alpha} for regularization")
    
    # Extract features
    print("Extracting features...")
    X_train = np.array([model.extract_features(board) for board in train_df['board']])
    y_train = train_df['true_score'].values
    
    print(f"Feature shape: {X_train.shape}")
    
    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("\nTraining linear model...")
    model.fit(X_train_split, y_train_split)
    
    # Evaluate
    train_predictions = model.predict(X_train_split)
    train_mae = np.mean(np.abs(train_predictions - y_train_split))
    train_rmse = np.sqrt(np.mean((train_predictions - y_train_split)**2))
    
    val_predictions = model.predict(X_val_split)
    val_mae = np.mean(np.abs(val_predictions - y_val_split))
    val_rmse = np.sqrt(np.mean((val_predictions - y_val_split)**2))
    
    print("\nTraining set performance:")
    print(f"  MAE: {train_mae:.2f} centipawns")
    print(f"  RMSE: {train_rmse:.2f} centipawns")
    print("\nValidation set performance:")
    print(f"  MAE: {val_mae:.2f} centipawns")
    print(f"  RMSE: {val_rmse:.2f} centipawns")
    
    # Load independent validation data
    print(f"\nLoading {num_val_samples} independent validation positions...")
    val_stream = ds["train"].skip(10000 + num_train_samples)
    val_data = list(val_stream.take(num_val_samples))
    
    # Process validation data
    val_df = pd.DataFrame(val_data)
    val_df = val_df.dropna(subset=['fen'])
    val_df['should_skip'] = val_df.apply(lambda row: should_skip_position(row), axis=1)
    val_df = val_df[~val_df['should_skip']].copy()
    val_df['true_score'] = val_df.apply(extract_centipawn_score, axis=1)
    val_df['board'] = val_df['fen'].apply(create_board_from_fen)
    
    X_val = np.array([model.extract_features(board) for board in val_df['board']])
    y_val = val_df['true_score'].values
    
    final_val_predictions = model.predict(X_val)
    final_val_mae = np.mean(np.abs(final_val_predictions - y_val))
    final_val_rmse = np.sqrt(np.mean((final_val_predictions - y_val)**2))
    
    print("\nIndependent validation performance:")
    print(f"  MAE: {final_val_mae:.2f} centipawns")
    print(f"  RMSE: {final_val_rmse:.2f} centipawns")
    
    # Visualize learned piece-square tables
    print("\n" + "="*60)
    print("LEARNED PIECE-SQUARE TABLES (Deltas from base values)")
    print("="*60)
    
    piece_names = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]
    for piece_name in piece_names:
        model.visualize_piece_square_table(piece_name, color='white', show_deltas=True)
    
    # Show other learned features
    _, other_features = model.get_piece_square_tables()
    print("\n" + "="*60)
    print("OTHER LEARNED FEATURES")
    print("="*60)
    print(f"Turn bonus (white to move): {other_features['turn_bonus']:.1f} cp")
    print(f"White kingside castling: {other_features['white_kingside_castling']:.1f} cp")
    print(f"White queenside castling: {other_features['white_queenside_castling']:.1f} cp")
    print(f"Black kingside castling: {other_features['black_kingside_castling']:.1f} cp")
    print(f"Black queenside castling: {other_features['black_queenside_castling']:.1f} cp")
    print(f"Model intercept: {other_features['intercept']:.1f} cp")
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), "linear_piece_square_model_{}.pkl".format(num_train_samples))
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model, {'val_mae': final_val_mae, 'val_rmse': final_val_rmse}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        # Try different regularization strengths
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
        results = []
        
        for alpha in alphas:
            print(f"\n\n{'='*80}")
            print(f"Training with regularization alpha={alpha}")
            print('='*80)
            model, metrics = train_linear_piece_square_model(num_train_samples=100000, num_val_samples=20000, alpha=alpha)
            
            # Get validation performance and magnitude statistics
            tables, other_features = model.get_piece_square_tables()
            all_deltas = []
            for piece_name in ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]:
                deltas = tables[piece_name]['deltas_white'].flatten()
                all_deltas.extend(deltas)
            
            avg_magnitude = np.mean(np.abs(all_deltas))
            max_magnitude = np.max(np.abs(all_deltas))
            
            results.append({
                'alpha': alpha,
                'val_mae': metrics['val_mae'],
                'val_rmse': metrics['val_rmse'],
                'avg_magnitude': avg_magnitude,
                'max_magnitude': max_magnitude
            })
        
        # Print summary
        print("\n\n" + "="*80)
        print("REGULARIZATION SUMMARY")
        print("="*80)
        print(f"{'Alpha':>10} | {'Val MAE':>10} | {'Avg Mag':>10} | {'Max Mag':>10}")
        print("-"*50)
        for r in results:
            print(f"{r['alpha']:>10.1f} | {r['val_mae']:>10.0f} | {r['avg_magnitude']:>10.1f} | {r['max_magnitude']:>10.1f}")
        
        # Find best alpha (balance between performance and reasonable magnitudes)
        # Prefer alpha with low MAE but also reasonable piece-square magnitudes (not too extreme)
        best_result = min(results, key=lambda x: x['val_mae'])
        print(f"\nBest validation MAE: {best_result['val_mae']:.0f} cp with alpha={best_result['alpha']}")
        
        # Also consider magnitude - very large values might be overfitting
        reasonable_results = [r for r in results if r['avg_magnitude'] < 50]
        if reasonable_results:
            best_reasonable = min(reasonable_results, key=lambda x: x['val_mae'])
            print(f"Best with reasonable magnitudes (<50 cp avg): alpha={best_reasonable['alpha']} (MAE={best_reasonable['val_mae']:.0f} cp)")
    else:
        # Train with optimal settings using more data
        print("Training linear piece-square model with optimal settings...")
        print("Using 1M training samples as suggested")
        model, metrics = train_linear_piece_square_model(
            num_train_samples=1_000_000, 
            num_val_samples=50_000, 
            alpha=100.0  # Good balance between performance and reasonable values
        )