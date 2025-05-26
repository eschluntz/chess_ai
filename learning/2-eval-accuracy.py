#!/usr/bin/env python

"""
Evaluate the accuracy of chess evaluation functions using the Lichess position evaluations dataset.
"""

import chess
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from collections.abc import Callable
from typing import Any
from scipy.stats import spearmanr

# Add parent directory to path to import eval functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import piece_value_eval, piece_position_eval


def create_board_from_fen(fen: str) -> chess.Board:
    """Create a chess.Board object from FEN string."""
    return chess.Board(fen)


def should_skip_position(example: dict[str, Any], include_mates: bool, max_mate_distance: int, max_cp: int) -> bool:
    """
    Determine if a position should be skipped based on filtering criteria.
    
    Returns:
        Boolean indicating if position should be skipped
    """
    # Skip if neither mate nor cp value exists
    if pd.isna(example.get('mate')) and pd.isna(example.get('cp')):
        return True
        
    if pd.notna(example.get('mate')):
        if not include_mates:
            return True
        
        # Filter out very long mates
        if abs(example['mate']) > max_mate_distance:
            return True
        
        return False
    else:
        # Skip if cp is missing
        if pd.isna(example.get('cp')):
            return True
            
        # Filter out extreme positions
        if abs(example['cp']) > max_cp:
            return True
        
        return False


def extract_centipawn_score(example: dict[str, Any]) -> int:
    """
    Extract the centipawn score from the example, converting mate scores if needed.
    
    For mate positions, uses the formula: 20000 - (300 * mate_in_moves)
    This gives mate-in-1 = 19700 cp, mate-in-10 = 17000 cp, etc.
    
    This function should only be called after should_skip_position returns False,
    so we can safely assume any mate score needs conversion.
    
    Returns:
        centipawn_score: The centipawn score (either direct or converted from mate)
    """
    if pd.notna(example.get('mate')):
        mate_in_moves = example['mate']
        if mate_in_moves > 0:
            # Positive means white has mate
            return 20000 - (300 * mate_in_moves)
        else:
            # Negative means black has mate
            return -20000 + (300 * abs(mate_in_moves))
    else:
        # Return cp value, or 0 if it's missing (should be rare after filtering)
        return example.get('cp', 0)


def process_dataset_batch(dataset_stream: Any, num_samples: int, include_mates: bool, max_mate_distance: int, max_cp: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Process dataset and return filtered examples as a DataFrame.
    Uses functional approach with dataset.take() and DataFrame operations.
    """
    # Load data using take()
    print(f"Loading {num_samples} positions...")
    data = list(tqdm(dataset_stream.take(num_samples), total=num_samples, desc="Loading positions"))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    initial_count = len(df)
    
    # First, filter out any rows with NaN values in critical columns
    df = df.dropna(subset=['fen'])
    
    # Apply filtering using the helper function
    df['should_skip'] = df.apply(
        lambda row: should_skip_position(row, include_mates, max_mate_distance, max_cp),
        axis=1
    )
    
    # Filter out skipped positions
    df_filtered = df[~df['should_skip']].copy()
    
    # Add true score and metadata for remaining positions
    df_filtered['true_score'] = df_filtered.apply(extract_centipawn_score, axis=1)
    df_filtered['is_mate'] = df_filtered['mate'].notna()
    
    # Calculate simple stats
    stats = {
        'total_loaded': initial_count,
        'total_filtered': initial_count - len(df_filtered),
        'mate_positions': df_filtered['is_mate'].sum(),
        'total_processed': len(df_filtered)
    }
    
    # Return only needed columns
    processed_df = df_filtered[['fen', 'true_score', 'is_mate']]
    
    return processed_df, stats


def evaluate_accuracy(eval_function: Callable[[chess.Board], tuple[int, bool]], dataset_stream: Any, num_samples: int = 10000, include_mates: bool = True, max_mate_distance: int = 15, max_cp: int = 750) -> dict[str, Any]:
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
    print(f"Evaluating {eval_function.__name__} on {num_samples} positions...")
    
    # Process dataset first
    processed_df, stats = process_dataset_batch(
        dataset_stream, num_samples, include_mates, max_mate_distance, max_cp
    )
    
    # Create boards column
    print(f"Creating chess boards for {len(processed_df)} positions...")
    processed_df['board'] = processed_df['fen'].apply(create_board_from_fen)
    
    # Evaluate all positions using apply
    print(f"Evaluating positions with {eval_function.__name__}...")
    tqdm.pandas(desc=f"Running {eval_function.__name__}")
    processed_df['prediction'] = processed_df['board'].progress_apply(
        lambda board: eval_function(board)[0]
    )
    
    # Extract arrays for metrics
    predictions = processed_df['prediction'].values
    true_values = processed_df['true_score'].values
    
    # Check for NaN values and filter them out if any exist
    mask = ~(np.isnan(predictions) | np.isnan(true_values))
    if not mask.all():
        print(f"Warning: Found {(~mask).sum()} NaN values, filtering them out")
        predictions = predictions[mask]
        true_values = true_values[mask]
    
    # Calculate metrics
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    
    # Calculate correlations
    pearson_correlation = np.corrcoef(predictions, true_values)[0, 1]
    spearman_correlation, _ = spearmanr(predictions, true_values)
    
    # Calculate percentage of correct sign predictions (who's winning)
    correct_sign = np.sum((predictions > 0) == (true_values > 0)) / len(predictions) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'pearson_correlation': pearson_correlation,
        'spearman_correlation': spearman_correlation,
        'correct_sign_percentage': correct_sign,
        'num_evaluated': len(predictions),
        'mate_positions': stats['mate_positions'],
        'total_filtered': stats['total_filtered'],
        'predictions': predictions,
        'true_values': true_values
    }



def create_scatter_plots_with_and_without_mates(results_with_mates: dict[str, Any], results_without_mates: dict[str, Any], eval_name: str) -> None:
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
    ax1.set_title(f'{eval_name} vs Stockfish (Including Mates)\nSpearman: {results_with_mates.get("spearman_correlation", results_with_mates.get("correlation", 0)):.3f}, MAE: {results_with_mates["mae"]:.0f}', fontsize=14)
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
    ax2.set_title(f'{eval_name} vs Stockfish (Excluding Mates)\nSpearman: {results_without_mates.get("spearman_correlation", results_without_mates.get("correlation", 0)):.3f}, MAE: {results_without_mates["mae"]:.0f}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    textstr = f'N = {results_without_mates["num_evaluated"]:,}'
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Display plot
    plt.tight_layout()
    plt.show()


def main(eval_functions: list[tuple[Callable[[chess.Board], tuple[int, bool]], str]] = None) -> None:
    """
    Evaluate chess evaluation functions.
    
    Args:
        eval_functions: List of (function, name) tuples to evaluate.
                       Defaults to piece_value_eval and piece_position_eval.
    """
    if eval_functions is None:
        eval_functions = [
            (piece_value_eval, 'piece_value_eval'),
            (piece_position_eval, 'piece_position_eval')
        ]
    
    # Load the streaming dataset
    print("Loading Lichess chess position evaluations dataset (streaming)...")
    ds_full = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    
    # Number of samples to evaluate
    num_samples = 10000
    
    # Store results for all functions
    all_results = {}
    
    # Evaluate each function
    for eval_func, func_name in eval_functions:
        print("\n" + "="*50)
        print(f"Evaluating {func_name}")
        
        # Evaluate WITH mates
        print(f"\nEvaluating {func_name} WITH mates...")
        dataset_stream = ds_full["train"]
        results_with_mates = evaluate_accuracy(eval_func, dataset_stream.take(num_samples), num_samples, include_mates=True)
        
        # Evaluate WITHOUT mates
        print(f"\nEvaluating {func_name} WITHOUT mates...")
        dataset_stream = ds_full["train"]
        results_without_mates = evaluate_accuracy(eval_func, dataset_stream.take(num_samples), num_samples, include_mates=False)
        
        # Store results
        all_results[func_name] = {
            'with_mates': results_with_mates,
            'without_mates': results_without_mates
        }
        
        # Print results
        print(f"\nResults for {func_name} WITH mates:")
        print(f"  Mean Absolute Error: {results_with_mates['mae']:.2f} centipawns")
        print(f"  Root Mean Squared Error: {results_with_mates['rmse']:.2f} centipawns")
        print(f"  Pearson correlation: {results_with_mates['pearson_correlation']:.3f}")
        print(f"  Spearman correlation: {results_with_mates['spearman_correlation']:.3f}")
        print(f"  Correct winner prediction: {results_with_mates['correct_sign_percentage']:.1f}%")
        print(f"  Positions evaluated: {results_with_mates['num_evaluated']}")
        print(f"  Mate positions included: {results_with_mates['mate_positions']}")
        print(f"  Total positions filtered: {results_with_mates['total_filtered']}")
        
        print(f"\nResults for {func_name} WITHOUT mates:")
        print(f"  Mean Absolute Error: {results_without_mates['mae']:.2f} centipawns")
        print(f"  Root Mean Squared Error: {results_without_mates['rmse']:.2f} centipawns")
        print(f"  Pearson correlation: {results_without_mates['pearson_correlation']:.3f}")
        print(f"  Spearman correlation: {results_without_mates['spearman_correlation']:.3f}")
        print(f"  Correct winner prediction: {results_without_mates['correct_sign_percentage']:.1f}%")
        print(f"  Positions evaluated: {results_without_mates['num_evaluated']}")
        print(f"  Total positions filtered: {results_without_mates['total_filtered']}")
        
        # Create scatter plots
        create_scatter_plots_with_and_without_mates(results_with_mates, results_without_mates, func_name)
    
    # Compare results if we have at least 2 functions
    if len(eval_functions) >= 2:
        print("\n" + "="*50)
        print("Comparisons:")
        
        base_name = eval_functions[0][1]
        base_results = all_results[base_name]
        
        for eval_func, func_name in eval_functions[1:]:
            func_results = all_results[func_name]
            
            print(f"\n{base_name} vs {func_name}:")
            
            print("  WITH mates:")
            mae_improvement = base_results['with_mates']['mae'] - func_results['with_mates']['mae']
            spearman_improvement = func_results['with_mates']['spearman_correlation'] - base_results['with_mates']['spearman_correlation']
            win_improvement = func_results['with_mates']['correct_sign_percentage'] - base_results['with_mates']['correct_sign_percentage']
            print(f"    MAE improvement: {mae_improvement:.2f} centipawns")
            print(f"    Spearman improvement: {spearman_improvement:.3f}")
            print(f"    Winner prediction improvement: {win_improvement:.1f}%")
            
            print("  WITHOUT mates:")
            mae_improvement = base_results['without_mates']['mae'] - func_results['without_mates']['mae']
            spearman_improvement = func_results['without_mates']['spearman_correlation'] - base_results['without_mates']['spearman_correlation']
            win_improvement = func_results['without_mates']['correct_sign_percentage'] - base_results['without_mates']['correct_sign_percentage']
            print(f"    MAE improvement: {mae_improvement:.2f} centipawns")
            print(f"    Spearman improvement: {spearman_improvement:.3f}")
            print(f"    Winner prediction improvement: {win_improvement:.1f}%")
    


if __name__ == "__main__":
    main()