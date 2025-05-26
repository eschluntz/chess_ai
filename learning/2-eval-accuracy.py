#!/usr/bin/env python

"""
Evaluate the accuracy of chess evaluation functions using the Lichess position evaluations dataset.
"""

import chess
from datasets import load_dataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import eval functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import piece_value_eval, piece_position_eval


def create_board_from_fen(fen):
    """Create a chess.Board object from FEN string."""
    return chess.Board(fen)


def convert_mate_to_cp(mate_in_moves):
    """
    Convert mate-in-N moves to centipawn equivalent.
    Uses a formula that gives very high scores for forced mates.
    
    Common approach: 20000 - (300 * mate_in_moves) for positive mates
    This gives mate-in-1 = 19700 cp, mate-in-10 = 17000 cp, etc.
    """
    if mate_in_moves > 0:
        # Positive means white has mate
        return 20000 - (300 * mate_in_moves)
    else:
        # Negative means black has mate
        return -20000 + (300 * abs(mate_in_moves))


def should_skip_position(example, include_mates, max_mate_distance, max_cp):
    """
    Determine if a position should be skipped based on filtering criteria.
    
    Returns:
        (should_skip, reason)
        - should_skip: Boolean indicating if position should be skipped
        - reason: String indicating why it was skipped ('mate_excluded', 'long_mate', 'extreme_position', or None)
    """
    if example['mate'] is not None:
        if not include_mates:
            return True, 'mate_excluded'
        
        # Filter out very long mates
        if abs(example['mate']) > max_mate_distance:
            return True, 'long_mate'
        
        return False, None
    else:
        # Filter out extreme positions
        if abs(example['cp']) > max_cp:
            return True, 'extreme_position'
        
        return False, None


def get_true_score(example):
    """
    Get the true score from the example, converting mate scores to centipawns if needed.
    
    This function should only be called after should_skip_position returns False,
    so we can safely assume any mate score needs conversion.
    
    Returns:
        true_score: The true centipawn score (or converted mate score)
    """
    if example['mate'] is not None:
        return convert_mate_to_cp(example['mate'])
    else:
        return example['cp']


def evaluate_accuracy(eval_function, dataset_stream, num_samples=10000, include_mates=True, max_mate_distance=15, max_cp=750):
    """
    Evaluate the accuracy of an evaluation function against Stockfish evaluations.
    
    Args:
        eval_function: Function that takes a chess.Board and returns (score, done)
        dataset_stream: Streaming dataset
        num_samples: Number of positions to evaluate
        include_mates: Whether to include mate positions in evaluation
        max_mate_distance: Maximum mate distance to include (filters out mates > N moves)
        max_cp: Maximum centipawn value to include (filters out heavily winning/losing positions)
    
    Returns:
        dict: Dictionary containing MAE, MSE, and correlation metrics
    """
    predictions = []
    true_values = []
    mate_positions = 0
    filtered_long_mates = 0
    filtered_extreme_positions = 0
    
    print(f"Evaluating {eval_function.__name__} on {num_samples} positions...")
    
    # Use tqdm for progress bar
    for i, example in enumerate(tqdm(dataset_stream, total=num_samples, desc=f"Evaluating {eval_function.__name__}")):
        if i >= num_samples:
            break
        
        # Check if position should be skipped
        should_skip, skip_reason = should_skip_position(
            example, include_mates, max_mate_distance, max_cp
        )
        
        if should_skip:
            if skip_reason == 'long_mate':
                filtered_long_mates += 1
            elif skip_reason == 'extreme_position':
                filtered_extreme_positions += 1
            continue
        
        # Get the true score (after filtering, so mates are safe to convert)
        true_score = get_true_score(example)
        
        # Track mate positions
        if example['mate'] is not None:
            mate_positions += 1
        
        # Create board from FEN
        board = create_board_from_fen(example['fen'])
        
        # Get prediction from our evaluation function
        pred_score, _ = eval_function(board)
        
        # Store values
        predictions.append(pred_score)
        true_values.append(true_score)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # Calculate metrics
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    
    # Calculate correlation
    correlation = np.corrcoef(predictions, true_values)[0, 1]
    
    # Calculate percentage of correct sign predictions (who's winning)
    correct_sign = np.sum((predictions > 0) == (true_values > 0)) / len(predictions) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation,
        'correct_sign_percentage': correct_sign,
        'num_evaluated': len(predictions),
        'mate_positions': mate_positions,
        'filtered_long_mates': filtered_long_mates,
        'filtered_extreme_positions': filtered_extreme_positions,
        'predictions': predictions,
        'true_values': true_values
    }


def create_scatter_plots_with_and_without_mates(results_with_mates, results_without_mates, eval_name, save_path):
    """Create scatter plots comparing predicted vs true scores, with and without mates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Including mates
    predictions_with = results_with_mates['predictions']
    true_values_with = results_with_mates['true_values']
    
    ax1.scatter(true_values_with, predictions_with, alpha=0.5, s=1)
    
    # Add diagonal line (perfect predictions)
    min_val = min(true_values_with.min(), predictions_with.min())
    max_val = max(true_values_with.max(), predictions_with.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    ax1.set_xlabel('True Score (Stockfish centipawns)', fontsize=12)
    ax1.set_ylabel('Predicted Score (centipawns)', fontsize=12)
    ax1.set_title(f'{eval_name} vs Stockfish (Including Mates)\nCorrelation: {results_with_mates["correlation"]:.3f}, MAE: {results_with_mates["mae"]:.0f}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    textstr = f'N = {results_with_mates["num_evaluated"]:,}\nMate positions: {results_with_mates["mate_positions"]:,}'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Excluding mates
    predictions_without = results_without_mates['predictions']
    true_values_without = results_without_mates['true_values']
    
    ax2.scatter(true_values_without, predictions_without, alpha=0.5, s=1, color='green')
    
    # Add diagonal line
    min_val = min(true_values_without.min(), predictions_without.min())
    max_val = max(true_values_without.max(), predictions_without.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    ax2.set_xlabel('True Score (Stockfish centipawns)', fontsize=12)
    ax2.set_ylabel('Predicted Score (centipawns)', fontsize=12)
    ax2.set_title(f'{eval_name} vs Stockfish (Excluding Mates)\nCorrelation: {results_without_mates["correlation"]:.3f}, MAE: {results_without_mates["mae"]:.0f}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    textstr = f'N = {results_without_mates["num_evaluated"]:,}'
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved to {save_path}")


def main():
    # Load the streaming dataset
    print("Loading Lichess chess position evaluations dataset (streaming)...")
    ds_full = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    dataset_stream = ds_full["train"]
    
    # Number of samples to evaluate
    num_samples = 10000
    
    # Evaluate piece_value_eval WITH mates
    print("\n" + "="*50)
    print("Evaluating piece_value_eval() WITH mates...")
    piece_value_with_mates = evaluate_accuracy(piece_value_eval, dataset_stream.take(num_samples), num_samples, include_mates=True)
    
    # Evaluate piece_value_eval WITHOUT mates
    print("\nEvaluating piece_value_eval() WITHOUT mates...")
    dataset_stream = ds_full["train"]
    piece_value_without_mates = evaluate_accuracy(piece_value_eval, dataset_stream.take(num_samples), num_samples, include_mates=False)
    
    print("\nResults for piece_value_eval() WITH mates:")
    print(f"  Mean Absolute Error: {piece_value_with_mates['mae']:.2f} centipawns")
    print(f"  Root Mean Squared Error: {piece_value_with_mates['rmse']:.2f} centipawns")
    print(f"  Correlation with Stockfish: {piece_value_with_mates['correlation']:.3f}")
    print(f"  Correct winner prediction: {piece_value_with_mates['correct_sign_percentage']:.1f}%")
    print(f"  Positions evaluated: {piece_value_with_mates['num_evaluated']}")
    print(f"  Mate positions included: {piece_value_with_mates['mate_positions']}")
    print(f"  Long mates filtered (>15 moves): {piece_value_with_mates['filtered_long_mates']}")
    print(f"  Extreme positions filtered (|cp| > 750): {piece_value_with_mates['filtered_extreme_positions']}")
    
    print("\nResults for piece_value_eval() WITHOUT mates:")
    print(f"  Mean Absolute Error: {piece_value_without_mates['mae']:.2f} centipawns")
    print(f"  Root Mean Squared Error: {piece_value_without_mates['rmse']:.2f} centipawns")
    print(f"  Correlation with Stockfish: {piece_value_without_mates['correlation']:.3f}")
    print(f"  Correct winner prediction: {piece_value_without_mates['correct_sign_percentage']:.1f}%")
    print(f"  Positions evaluated: {piece_value_without_mates['num_evaluated']}")
    print(f"  Extreme positions filtered (|cp| > 750): {piece_value_without_mates['filtered_extreme_positions']}")
    
    # Create scatter plots for piece_value_eval
    create_scatter_plots_with_and_without_mates(piece_value_with_mates, piece_value_without_mates,
                                               'piece_value_eval', 
                                               '/Users/erik/code/chess_ai/learning/piece_value_eval_scatter.png')
    
    # Evaluate piece_position_eval WITH mates
    print("\n" + "="*50)
    print("Evaluating piece_position_eval() WITH mates...")
    dataset_stream = ds_full["train"]
    piece_position_with_mates = evaluate_accuracy(piece_position_eval, dataset_stream.take(num_samples), num_samples, include_mates=True)
    
    # Evaluate piece_position_eval WITHOUT mates
    print("\nEvaluating piece_position_eval() WITHOUT mates...")
    dataset_stream = ds_full["train"]
    piece_position_without_mates = evaluate_accuracy(piece_position_eval, dataset_stream.take(num_samples), num_samples, include_mates=False)
    
    print("\nResults for piece_position_eval() WITH mates:")
    print(f"  Mean Absolute Error: {piece_position_with_mates['mae']:.2f} centipawns")
    print(f"  Root Mean Squared Error: {piece_position_with_mates['rmse']:.2f} centipawns")
    print(f"  Correlation with Stockfish: {piece_position_with_mates['correlation']:.3f}")
    print(f"  Correct winner prediction: {piece_position_with_mates['correct_sign_percentage']:.1f}%")
    print(f"  Positions evaluated: {piece_position_with_mates['num_evaluated']}")
    print(f"  Mate positions included: {piece_position_with_mates['mate_positions']}")
    print(f"  Long mates filtered (>15 moves): {piece_position_with_mates['filtered_long_mates']}")
    print(f"  Extreme positions filtered (|cp| > 750): {piece_position_with_mates['filtered_extreme_positions']}")
    
    print("\nResults for piece_position_eval() WITHOUT mates:")
    print(f"  Mean Absolute Error: {piece_position_without_mates['mae']:.2f} centipawns")
    print(f"  Root Mean Squared Error: {piece_position_without_mates['rmse']:.2f} centipawns")
    print(f"  Correlation with Stockfish: {piece_position_without_mates['correlation']:.3f}")
    print(f"  Correct winner prediction: {piece_position_without_mates['correct_sign_percentage']:.1f}%")
    print(f"  Positions evaluated: {piece_position_without_mates['num_evaluated']}")
    print(f"  Extreme positions filtered (|cp| > 750): {piece_position_without_mates['filtered_extreme_positions']}")
    
    # Create scatter plots for piece_position_eval
    create_scatter_plots_with_and_without_mates(piece_position_with_mates, piece_position_without_mates,
                                               'piece_position_eval', 
                                               '/Users/erik/code/chess_ai/learning/piece_position_eval_scatter.png')
    
    # Compare results
    print("\n" + "="*50)
    print("Comparison (WITH mates):")
    print(f"  MAE improvement: {piece_value_with_mates['mae'] - piece_position_with_mates['mae']:.2f} centipawns")
    print(f"  Correlation improvement: {piece_position_with_mates['correlation'] - piece_value_with_mates['correlation']:.3f}")
    print(f"  Winner prediction improvement: {piece_position_with_mates['correct_sign_percentage'] - piece_value_with_mates['correct_sign_percentage']:.1f}%")
    
    print("\nComparison (WITHOUT mates):")
    print(f"  MAE improvement: {piece_value_without_mates['mae'] - piece_position_without_mates['mae']:.2f} centipawns")
    print(f"  Correlation improvement: {piece_position_without_mates['correlation'] - piece_value_without_mates['correlation']:.3f}")
    print(f"  Winner prediction improvement: {piece_position_without_mates['correct_sign_percentage'] - piece_value_without_mates['correct_sign_percentage']:.1f}%")
    
    # Save results (exclude large arrays from JSON)
    results = {
        'piece_value_eval': {
            'with_mates': {k: v for k, v in piece_value_with_mates.items() if k not in ['predictions', 'true_values']},
            'without_mates': {k: v for k, v in piece_value_without_mates.items() if k not in ['predictions', 'true_values']}
        },
        'piece_position_eval': {
            'with_mates': {k: v for k, v in piece_position_with_mates.items() if k not in ['predictions', 'true_values']},
            'without_mates': {k: v for k, v in piece_position_without_mates.items() if k not in ['predictions', 'true_values']}
        }
    }
    
    import json
    with open('/Users/erik/code/chess_ai/learning/eval_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to eval_accuracy_results.json")


if __name__ == "__main__":
    main()