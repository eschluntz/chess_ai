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
    print_results_summary
)
from learning.feature_extraction import extract_features_basic, extract_features_piece_square

# Import neural network loader
try:
    # Import from the correct module name (4-train-neural-network)
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_neural_network", 
                                                  os.path.join(os.path.dirname(__file__), "40-train-neural-network.py"))
    train_neural_network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_neural_network)
    load_neural_network_eval = train_neural_network.load_neural_network_eval
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available - neural network models will be skipped")


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
    
    # Load Random Forest models
    rf_model_files = sorted([f for f in os.listdir(model_dir) if f.startswith('random_forest_chess_model_') and f.endswith('.pkl')])
    
    if rf_model_files:
        print(f"\nFound {len(rf_model_files)} Random Forest models:")
        for model_file in rf_model_files:
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
        print("No Random Forest models found.")
    
    # Load Neural Network models if PyTorch is available
    if PYTORCH_AVAILABLE:
        nn_model_files = sorted([f for f in os.listdir(model_dir) if f.startswith('neural_network_chess_model_') and f.endswith('.pkl')])
        
        if nn_model_files:
            print(f"\nFound {len(nn_model_files)} Neural Network models:")
            for model_file in nn_model_files:
                # Extract info from filename
                # Format: neural_network_chess_model_<size>_<suffix>.pkl
                parts = model_file.replace('neural_network_chess_model_', '').replace('.pkl', '').split('_')
                if len(parts) >= 2:
                    model_name = f'nn_{parts[0]}_{parts[1]}'
                else:
                    model_name = f'nn_{parts[0]}'
                
                model_path = os.path.join(model_dir, model_file)
                print(f"  Loading {model_file}...")
                nn_eval = load_neural_network_eval(model_path)
                all_functions.append((nn_eval, model_name))
        else:
            print("No Neural Network models found.")
    
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