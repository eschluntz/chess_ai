#!/usr/bin/env python

"""
Evaluate the accuracy of chess evaluation functions using the Lichess position evaluations dataset.
Always uses the first 10,000 positions for evaluation.
"""

from datasets import load_dataset
import sys
import os
import pickle
from collections.abc import Callable
import chess

# Add parent directory to path to import eval functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import piece_value_eval, piece_position_eval

# Import helper functions
from eval_accuracy_helpers import process_dataset_batch
from eval_common import (
    evaluate_all_functions,
    create_combined_scatter_plot,
    print_results_summary
)
from feature_extraction import extract_features_basic, extract_features_piece_square


def load_random_forest_eval(model_path: str) -> Callable[[chess.Board], tuple[int, bool]]:
    """Load the trained random forest model and return an evaluation function."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        
    # Handle both old and new model formats
    if isinstance(data, dict):
        model = data['model']
        feature_type = data['feature_type']
    else:
        # Old format - assume basic features
        model = data
        feature_type = 'basic'
    
    # Choose the appropriate feature extractor
    if feature_type == 'piece_square':
        extract_features = extract_features_piece_square
    else:
        extract_features = extract_features_basic
    
    def random_forest_eval(board: chess.Board) -> tuple[int, bool]:
        """Evaluate a position using the trained random forest model."""
        features = extract_features(board)
        score = int(model.predict([features])[0])
        return score, False
    
    return random_forest_eval


def main(num_samples: int = 2000) -> None:
    """
    Evaluate chess evaluation functions.
    Uses the first positions from the dataset for evaluation.
    
    Args:
        num_samples: Number of positions to evaluate (default: 2000)
    """
    # Define evaluation functions
    baseline_name = 'piece_value_eval'
    
    # Start with hardcoded functions
    all_functions = [
        (piece_value_eval, 'piece_value_eval'),
        (piece_position_eval, 'piece_position_eval'),
    ]
    
    # Load all trained models
    model_dir = os.path.dirname(__file__)
    model_files = sorted([f for f in os.listdir(model_dir) if f.startswith('random_forest_chess_model_') and f.endswith('.pkl')])
    
    if model_files:
        print(f"\nFound {len(model_files)} trained models:")
        for model_file in model_files:
            # Extract info from filename
            # Format: random_forest_chess_model_<size>_<feature_type>.pkl
            parts = model_file.replace('random_forest_chess_model_', '').replace('.pkl', '').split('_')
            if len(parts) >= 2:
                # New format with feature type
                model_name = f'rf_{parts[0]}_{parts[1]}'
            else:
                # Old format
                model_name = f'rf_model_{parts[0]}'
            
            model_path = os.path.join(model_dir, model_file)
            print(f"  Loading {model_file}...")
            rf_eval = load_random_forest_eval(model_path)
            all_functions.append((rf_eval, model_name))
    else:
        print("No trained models found. Evaluating only hardcoded functions.")
    
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
    print("\n" + "="*50)
    print("Evaluating all functions...")
    all_results = evaluate_all_functions(all_functions, processed_df)
    
    # Print results
    print_results_summary(all_results, baseline_name)
    
    # Create combined scatter plot
    print("\nCreating scatter plots...")
    create_combined_scatter_plot(all_results, baseline_name)


if __name__ == "__main__":
    main()