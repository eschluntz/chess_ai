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


def evaluate_all_functions(eval_functions: list[tuple[Callable[[chess.Board], tuple[int, bool]], str]], processed_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
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
        
        # Calculate metrics
        predictions = processed_df['prediction'].values
        true_values = processed_df['true_score'].values
        is_mate = processed_df['is_mate'].values
        
        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        pearson = np.corrcoef(predictions, true_values)[0, 1]
        spearman, _ = spearmanr(predictions, true_values)
        correct_sign = np.sum((predictions > 0) == (true_values > 0)) / len(predictions) * 100
        
        # Store results
        all_results[func_name] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'pearson_correlation': pearson,
            'spearman_correlation': spearman,
            'correct_sign_percentage': correct_sign,
            'num_evaluated': len(predictions),
            'mate_positions': is_mate.sum(),
            'predictions': predictions,
            'true_values': true_values
        }
    
    return all_results


def create_combined_scatter_plot(all_results: dict[str, dict[str, Any]], baseline_name: str) -> None:
    """
    Create a combined scatter plot with all evaluation functions.
    Shows results including mates for all functions.
    """
    num_functions = len(all_results)
    fig, axes = plt.subplots(num_functions, 2, figsize=(16, 6 * num_functions))
    
    # Handle single function case
    if num_functions == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (func_name, results) in enumerate(all_results.items()):
        predictions = results['predictions']
        true_values = results['true_values']
        
        # Left subplot: Full view
        ax_full = axes[idx, 0]
        ax_full.scatter(true_values, predictions, alpha=0.5, s=1)
        
        # Add diagonal line
        min_val = min(true_values.min(), predictions.min())
        max_val = max(true_values.max(), predictions.max())
        ax_full.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        ax_full.set_xlabel('True Score (Stockfish centipawns)', fontsize=12)
        ax_full.set_ylabel('Predicted Score (centipawns)', fontsize=12)
        
        # Add improvement in title if not baseline
        title = f'{func_name} vs Stockfish (Full Range)\nSpearman: {results["spearman_correlation"]:.3f}, MAE: {results["mae"]:.0f}'
        if func_name != baseline_name and baseline_name in all_results:
            spearman_improvement = results['spearman_correlation'] - all_results[baseline_name]['spearman_correlation']
            title += f'\n(Δ Spearman: {spearman_improvement:+.3f})'
        ax_full.set_title(title, fontsize=14)
        
        ax_full.legend()
        ax_full.grid(True, alpha=0.3)
        
        # Add statistics text
        textstr = f'N = {results["num_evaluated"]:,}\nMate positions: {results["mate_positions"]:,}'
        ax_full.text(0.02, 0.98, textstr, transform=ax_full.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Right subplot: Zoomed view (-1000 to 1000)
        ax_zoom = axes[idx, 1]
        
        # Filter data for zoomed view
        zoom_mask = (true_values >= -1000) & (true_values <= 1000) & (predictions >= -1000) & (predictions <= 1000)
        true_values_zoom = true_values[zoom_mask]
        predictions_zoom = predictions[zoom_mask]
        
        ax_zoom.scatter(true_values_zoom, predictions_zoom, alpha=0.5, s=1, color='green')
        
        # Add diagonal line for zoomed view
        ax_zoom.plot([-1000, 1000], [-1000, 1000], 'r--', label='Perfect prediction')
        
        ax_zoom.set_xlim(-1000, 1000)
        ax_zoom.set_ylim(-1000, 1000)
        ax_zoom.set_xlabel('True Score (Stockfish centipawns)', fontsize=12)
        ax_zoom.set_ylabel('Predicted Score (centipawns)', fontsize=12)
        
        # Calculate metrics for zoomed region
        if len(predictions_zoom) > 0:
            from sklearn.metrics import mean_absolute_error
            from scipy.stats import spearmanr
            mae_zoom = mean_absolute_error(true_values_zoom, predictions_zoom)
            spearman_zoom, _ = spearmanr(predictions_zoom, true_values_zoom)
            zoom_title = f'{func_name} vs Stockfish (Zoomed: ±1000cp)\nSpearman: {spearman_zoom:.3f}, MAE: {mae_zoom:.0f}'
        else:
            zoom_title = f'{func_name} vs Stockfish (Zoomed: ±1000cp)\nNo data in range'
        
        ax_zoom.set_title(zoom_title, fontsize=14)
        ax_zoom.legend()
        ax_zoom.grid(True, alpha=0.3)
        
        # Add statistics text for zoomed view
        zoom_textstr = f'N = {len(predictions_zoom):,}\n({len(predictions_zoom)/len(predictions)*100:.1f}% of data)'
        ax_zoom.text(0.02, 0.98, zoom_textstr, transform=ax_zoom.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout(pad=1.0)
    plt.show()


def print_results_summary(all_results: dict[str, dict[str, Any]], baseline_name: str) -> None:
    """
    Print a summary of evaluation results including mates.
    """
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    for func_name, results in all_results.items():
        is_baseline = func_name == baseline_name
        print(f"\n{func_name}{' (BASELINE)' if is_baseline else ''}:")
        
        # Print results
        print(f"  Mean Absolute Error: {results['mae']:.2f} centipawns")
        print(f"  Spearman correlation: {results['spearman_correlation']:.3f}")
        print(f"  Correct winner prediction: {results['correct_sign_percentage']:.1f}%")
        print(f"  Positions evaluated: {results['num_evaluated']:,} (including {results['mate_positions']:,} mate positions)")
        
        if not is_baseline and baseline_name in all_results:
            mae_improvement = all_results[baseline_name]['mae'] - results['mae']
            spearman_improvement = results['spearman_correlation'] - all_results[baseline_name]['spearman_correlation']
            win_improvement = results['correct_sign_percentage'] - all_results[baseline_name]['correct_sign_percentage']
            print(f"  vs baseline: MAE {mae_improvement:+.0f}, Spearman {spearman_improvement:+.3f}, Win% {win_improvement:+.1f}")
    
    # Print table summary
    print("\n" + "="*50)
    print("SUMMARY TABLE")
    print("="*50)
    
    # Table header
    print(f"{'Model':<25} {'MAE':>6} {'RMSE':>6} {'Spearman':>8} {'Pearson':>8} {'Win%':>6} {'N':>8}")
    print("-" * 80)
    
    # Sort models with baseline first, then alphabetically
    sorted_models = sorted(all_results.items(), key=lambda x: (x[0] != baseline_name, x[0]))
    
    for func_name, results in sorted_models:
        is_baseline = func_name == baseline_name
        name_display = f"{func_name}{'*' if is_baseline else ''}"
        
        print(f"{name_display:<25} "
              f"{results['mae']:>6.0f} "
              f"{results['rmse']:>6.0f} "
              f"{results['spearman_correlation']:>8.3f} "
              f"{results['pearson_correlation']:>8.3f} "
              f"{results['correct_sign_percentage']:>6.1f} "
              f"{results['num_evaluated']:>8,}")
    
    print("\n* = baseline model")