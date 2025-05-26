"""
Common evaluation functions shared between evaluation scripts.
"""

import chess
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from collections.abc import Callable
from typing import Any
from scipy.stats import spearmanr
from tqdm import tqdm


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
    processed_df['board'] = processed_df['fen'].apply(lambda fen: chess.Board(fen))
    
    for eval_func, func_name in eval_functions:
        print(f"\nEvaluating {func_name}...")
        
        # Evaluate all positions
        print(f"  Running {func_name} on {len(processed_df)} positions...")
        tqdm.pandas(desc=f"  {func_name}")
        processed_df['prediction'] = processed_df['board'].progress_apply(
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


def print_results_summary(all_results: dict[str, dict[str, dict[str, Any]]], baseline_name: str) -> None:
    """
    Print a summary of evaluation results.
    """
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