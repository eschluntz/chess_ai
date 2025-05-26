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


def process_dataset_batch(dataset_stream: Any, num_samples: int, max_mate_distance: int, max_cp: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Process dataset and return filtered examples as a DataFrame.
    Uses functional approach with dataset.take() and DataFrame operations.
    """
    # Load data using take()
    print(f"Loading {num_samples} positions...")
    data = list(dataset_stream.take(num_samples))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    initial_count = len(df)
    
    # First, filter out any rows with NaN values in critical columns
    df = df.dropna(subset=['fen'])
    
    # Apply filtering using the helper function (always include mates for processing)
    df['should_skip'] = df.apply(
        lambda row: should_skip_position(row, include_mates=True, max_mate_distance=max_mate_distance, max_cp=max_cp),
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



def evaluate_all_functions(eval_functions: list[tuple[Callable[[chess.Board], tuple[int, bool]], str]], processed_df: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Evaluate all functions on the same data.
    
    Args:
        eval_functions: List of (function, name) tuples to evaluate
        processed_df: DataFrame with all positions
    
    Returns:
        Dictionary mapping function names to results
    """
    all_results = {}
    
    # Create board objects once
    print(f"Creating chess boards for {len(processed_df)} positions...")
    processed_df['board'] = processed_df['fen'].apply(create_board_from_fen)
    
    for eval_func, func_name in eval_functions:
        print(f"\nEvaluating {func_name}...")
        
        # Evaluate all positions
        print(f"  Running {func_name} on {len(processed_df)} positions...")
        processed_df['prediction'] = processed_df['board'].apply(
            lambda board: eval_func(board)[0]
        )
        
        # Calculate metrics WITH mates
        predictions_all = processed_df['prediction'].values
        true_values_all = processed_df['true_score'].values
        is_mate_all = processed_df['is_mate'].values
        
        mae_with = mean_absolute_error(true_values_all, predictions_all)
        mse_with = mean_squared_error(true_values_all, predictions_all)
        rmse_with = np.sqrt(mse_with)
        pearson_with = np.corrcoef(predictions_all, true_values_all)[0, 1]
        spearman_with, _ = spearmanr(predictions_all, true_values_all)
        correct_sign_with = np.sum((predictions_all > 0) == (true_values_all > 0)) / len(predictions_all) * 100
        
        # Calculate metrics WITHOUT mates (filter using is_mate column)
        non_mate_mask = ~is_mate_all
        predictions_without = predictions_all[non_mate_mask]
        true_values_without = true_values_all[non_mate_mask]
        
        mae_without = mean_absolute_error(true_values_without, predictions_without)
        mse_without = mean_squared_error(true_values_without, predictions_without)
        rmse_without = np.sqrt(mse_without)
        pearson_without = np.corrcoef(predictions_without, true_values_without)[0, 1]
        spearman_without, _ = spearmanr(predictions_without, true_values_without)
        correct_sign_without = np.sum((predictions_without > 0) == (true_values_without > 0)) / len(predictions_without) * 100
        
        # Store results
        all_results[func_name] = {
            'with_mates': {
                'mae': mae_with,
                'mse': mse_with,
                'rmse': rmse_with,
                'pearson_correlation': pearson_with,
                'spearman_correlation': spearman_with,
                'correct_sign_percentage': correct_sign_with,
                'num_evaluated': len(predictions_all),
                'mate_positions': is_mate_all.sum(),
                'predictions': predictions_all,
                'true_values': true_values_all
            },
            'without_mates': {
                'mae': mae_without,
                'mse': mse_without,
                'rmse': rmse_without,
                'pearson_correlation': pearson_without,
                'spearman_correlation': spearman_without,
                'correct_sign_percentage': correct_sign_without,
                'num_evaluated': len(predictions_without),
                'predictions': predictions_without,
                'true_values': true_values_without
            }
        }
    
    return all_results


def create_combined_scatter_plot(all_results: dict[str, dict[str, dict[str, Any]]], baseline_name: str) -> None:
    """
    Create a combined scatter plot with all evaluation functions.
    Each row is one function, columns are with/without mates.
    """
    num_functions = len(all_results)
    fig, axes = plt.subplots(num_functions, 2, figsize=(16, 6 * num_functions))
    
    # Handle single function case
    if num_functions == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (func_name, results) in enumerate(all_results.items()):
        # Plot with mates (left column)
        ax_with = axes[idx, 0]
        predictions_with = results['with_mates']['predictions']
        true_values_with = results['with_mates']['true_values']
        
        ax_with.scatter(true_values_with, predictions_with, alpha=0.5, s=1)
        
        # Add diagonal line
        min_val = min(true_values_with.min(), predictions_with.min())
        max_val = max(true_values_with.max(), predictions_with.max())
        ax_with.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        ax_with.set_xlabel('True Score (Stockfish centipawns)', fontsize=12)
        ax_with.set_ylabel('Predicted Score (centipawns)', fontsize=12)
        
        # Add improvement in title if not baseline
        title = f'{func_name} vs Stockfish (Including Mates)\nSpearman: {results["with_mates"]["spearman_correlation"]:.3f}, MAE: {results["with_mates"]["mae"]:.0f}'
        if func_name != baseline_name and baseline_name in all_results:
            spearman_improvement = results['with_mates']['spearman_correlation'] - all_results[baseline_name]['with_mates']['spearman_correlation']
            title += f'\n(Δ Spearman: {spearman_improvement:+.3f})'
        ax_with.set_title(title, fontsize=14)
        
        ax_with.legend()
        ax_with.grid(True, alpha=0.3)
        
        # Add statistics text
        textstr = f'N = {results["with_mates"]["num_evaluated"]:,}\nMate positions: {results["with_mates"]["mate_positions"]:,}'
        ax_with.text(0.02, 0.98, textstr, transform=ax_with.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot without mates (right column)
        ax_without = axes[idx, 1]
        predictions_without = results['without_mates']['predictions']
        true_values_without = results['without_mates']['true_values']
        
        ax_without.scatter(true_values_without, predictions_without, alpha=0.5, s=1, color='green')
        
        # Add diagonal line
        min_val = min(true_values_without.min(), predictions_without.min())
        max_val = max(true_values_without.max(), predictions_without.max())
        ax_without.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        ax_without.set_xlabel('True Score (Stockfish centipawns)', fontsize=12)
        ax_without.set_ylabel('Predicted Score (centipawns)', fontsize=12)
        
        # Add improvement in title if not baseline
        title = f'{func_name} vs Stockfish (Excluding Mates)\nSpearman: {results["without_mates"]["spearman_correlation"]:.3f}, MAE: {results["without_mates"]["mae"]:.0f}'
        if func_name != baseline_name and baseline_name in all_results:
            spearman_improvement = results['without_mates']['spearman_correlation'] - all_results[baseline_name]['without_mates']['spearman_correlation']
            title += f'\n(Δ Spearman: {spearman_improvement:+.3f})'
        ax_without.set_title(title, fontsize=14)
        
        ax_without.legend()
        ax_without.grid(True, alpha=0.3)
        
        # Add statistics text
        textstr = f'N = {results["without_mates"]["num_evaluated"]:,}'
        ax_without.text(0.02, 0.98, textstr, transform=ax_without.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def main(num_samples: int = 10000) -> None:
    """
    Evaluate chess evaluation functions.
    
    Args:
        num_samples: Number of positions to evaluate
    """
    # Define evaluation functions
    baseline_func = (piece_value_eval, 'piece_value_eval')
    baseline_name = 'piece_value_eval'
    
    # All functions to evaluate (baseline first)
    all_functions = [
        baseline_func,
        (piece_position_eval, 'piece_position_eval'),
        # Add more evaluation functions here as needed
    ]
    
    # Load the streaming dataset
    print("Loading Lichess chess position evaluations dataset (streaming)...")
    ds_full = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    
    # Process data ONCE
    print("\nProcessing dataset...")
    dataset_stream = ds_full["train"]
    
    processed_df, stats = process_dataset_batch(
        dataset_stream.take(num_samples), num_samples, 
        max_mate_distance=15, max_cp=750
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
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    for func_name, results in all_results.items():
        is_baseline = func_name == baseline_name
        print(f"\n{func_name}{' (BASELINE)' if is_baseline else ''}:")
        
        print("  WITH mates:")
        print(f"    Mean Absolute Error: {results['with_mates']['mae']:.2f} centipawns")
        print(f"    Spearman correlation: {results['with_mates']['spearman_correlation']:.3f}")
        print(f"    Correct winner prediction: {results['with_mates']['correct_sign_percentage']:.1f}%")
        
        if not is_baseline and baseline_name in all_results:
            mae_improvement = all_results[baseline_name]['with_mates']['mae'] - results['with_mates']['mae']
            spearman_improvement = results['with_mates']['spearman_correlation'] - all_results[baseline_name]['with_mates']['spearman_correlation']
            win_improvement = results['with_mates']['correct_sign_percentage'] - all_results[baseline_name]['with_mates']['correct_sign_percentage']
            print(f"    vs baseline: MAE {mae_improvement:+.0f}, Spearman {spearman_improvement:+.3f}, Win% {win_improvement:+.1f}")
        
        print("  WITHOUT mates:")
        print(f"    Mean Absolute Error: {results['without_mates']['mae']:.2f} centipawns")
        print(f"    Spearman correlation: {results['without_mates']['spearman_correlation']:.3f}")
        print(f"    Correct winner prediction: {results['without_mates']['correct_sign_percentage']:.1f}%")
        
        if not is_baseline and baseline_name in all_results:
            mae_improvement = all_results[baseline_name]['without_mates']['mae'] - results['without_mates']['mae']
            spearman_improvement = results['without_mates']['spearman_correlation'] - all_results[baseline_name]['without_mates']['spearman_correlation']
            win_improvement = results['without_mates']['correct_sign_percentage'] - all_results[baseline_name]['without_mates']['correct_sign_percentage']
            print(f"    vs baseline: MAE {mae_improvement:+.0f}, Spearman {spearman_improvement:+.3f}, Win% {win_improvement:+.1f}")
    
    # Create combined scatter plot
    print("\nCreating scatter plots...")
    create_combined_scatter_plot(all_results, baseline_name)
    


if __name__ == "__main__":
    main()