"""
Common evaluation functions shared between evaluation scripts.
"""

import time
from collections.abc import Callable
from typing import Any

import chess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


def evaluate_all_functions(
    eval_functions: list[tuple[Callable[[chess.Board], tuple[int, bool]], str]],
    processed_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
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
    processed_df["board"] = processed_df["fen"].apply(lambda fen: chess.Board(fen))

    for eval_func, func_name in eval_functions:
        print(f"\nEvaluating {func_name}...")

        # Evaluate all positions with timing
        print(f"  Running {func_name} on {len(processed_df)} positions...")
        start_time = time.time()
        tqdm.pandas(desc=f"  {func_name}")
        processed_df["prediction"] = processed_df["board"].progress_apply(
            lambda board: eval_func(board)[0]
        )
        end_time = time.time()
        evaluation_time = end_time - start_time
        runs_per_second = len(processed_df) / evaluation_time
        runs_per_second_k = round(runs_per_second / 1000)
        print(f"  Completed in {evaluation_time:.2f}s ({runs_per_second_k}k runs/s)")

        # Calculate metrics
        predictions = processed_df["prediction"].values
        true_values = processed_df["true_score"].values
        is_mate = processed_df["is_mate"].values

        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        pearson = np.corrcoef(predictions, true_values)[0, 1]
        spearman, _ = spearmanr(predictions, true_values)
        correct_sign = (
            np.sum((predictions > 0) == (true_values > 0)) / len(predictions) * 100
        )

        # Calculate metrics for subset where abs(true_score) < 1000
        subset_mask = np.abs(true_values) < 1000
        subset_predictions = predictions[subset_mask]
        subset_true_values = true_values[subset_mask]
        subset_is_mate = is_mate[subset_mask]

        subset_mae = mean_absolute_error(subset_true_values, subset_predictions)
        subset_mse = mean_squared_error(subset_true_values, subset_predictions)
        subset_rmse = np.sqrt(subset_mse)
        subset_pearson = np.corrcoef(subset_predictions, subset_true_values)[0, 1]
        subset_spearman, _ = spearmanr(subset_predictions, subset_true_values)
        subset_correct_sign = (
            np.sum((subset_predictions > 0) == (subset_true_values > 0))
            / len(subset_predictions)
            * 100
        )

        # Store results
        all_results[func_name] = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
            "correct_sign_percentage": correct_sign,
            "num_evaluated": len(predictions),
            "mate_positions": is_mate.sum(),
            "predictions": predictions,
            "true_values": true_values,
            "evaluation_time": evaluation_time,
            "runs_per_second": runs_per_second,
            # Subset metrics for abs(true_score) < 1000
            "subset_mae": subset_mae,
            "subset_mse": subset_mse,
            "subset_rmse": subset_rmse,
            "subset_pearson_correlation": subset_pearson,
            "subset_spearman_correlation": subset_spearman,
            "subset_correct_sign_percentage": subset_correct_sign,
            "subset_num_evaluated": len(subset_predictions),
            "subset_mate_positions": subset_is_mate.sum(),
        }

    return all_results


def create_combined_scatter_plot(
    all_results: dict[str, dict[str, Any]], baseline_name: str
) -> None:
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
        predictions = results["predictions"]
        true_values = results["true_values"]

        # Left subplot: Full view
        ax_full = axes[idx, 0]
        ax_full.scatter(true_values, predictions, alpha=0.5, s=1)

        # Add diagonal line
        min_val = min(true_values.min(), predictions.min())
        max_val = max(true_values.max(), predictions.max())
        ax_full.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction"
        )

        # Add correlation fit line
        coeffs = np.polyfit(true_values, predictions, 1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.array([min_val, max_val])
        y_fit = fit_line(x_fit)
        ax_full.plot(x_fit, y_fit, "b:", linewidth=2, label="Linear fit")

        ax_full.set_xlabel("True Score (Stockfish centipawns)", fontsize=12)
        ax_full.set_ylabel("Predicted Score (centipawns)", fontsize=12)

        # Add improvement in title if not baseline
        title = (
            f"{func_name} vs Stockfish (Full Range)\n"
            f"Spearman: {results['spearman_correlation']:.3f}, MAE: {results['mae']:.0f}"
        )
        if func_name != baseline_name and baseline_name in all_results:
            spearman_improvement = (
                results["spearman_correlation"]
                - all_results[baseline_name]["spearman_correlation"]
            )
            title += f"\n(Δ Spearman: {spearman_improvement:+.3f})"
        ax_full.set_title(title, fontsize=14)

        ax_full.legend()
        ax_full.grid(True, alpha=0.3)

        # Add statistics text
        textstr = f"N = {results['num_evaluated']:,}\nMate positions: {results['mate_positions']:,}"
        ax_full.text(
            0.02,
            0.98,
            textstr,
            transform=ax_full.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Right subplot: Zoomed view (-1000 to 1000)
        ax_zoom = axes[idx, 1]

        # Filter data for zoomed view
        zoom_mask = (
            (true_values >= -1000)
            & (true_values <= 1000)
            & (predictions >= -1000)
            & (predictions <= 1000)
        )
        true_values_zoom = true_values[zoom_mask]
        predictions_zoom = predictions[zoom_mask]

        ax_zoom.scatter(
            true_values_zoom, predictions_zoom, alpha=0.5, s=1, color="green"
        )

        # Add diagonal line for zoomed view
        ax_zoom.plot([-1000, 1000], [-1000, 1000], "r--", label="Perfect prediction")

        # Add correlation fit line for zoomed view
        if len(predictions_zoom) > 1:
            coeffs_zoom = np.polyfit(true_values_zoom, predictions_zoom, 1)
            fit_line_zoom = np.poly1d(coeffs_zoom)
            x_fit_zoom = np.array([-1000, 1000])
            y_fit_zoom = fit_line_zoom(x_fit_zoom)
            ax_zoom.plot(x_fit_zoom, y_fit_zoom, "b:", linewidth=2, label="Linear fit")

        ax_zoom.set_xlim(-1000, 1000)
        ax_zoom.set_ylim(-1000, 1000)
        ax_zoom.set_xlabel("True Score (Stockfish centipawns)", fontsize=12)
        ax_zoom.set_ylabel("Predicted Score (centipawns)", fontsize=12)

        # Calculate metrics for zoomed region
        if len(predictions_zoom) > 0:
            from scipy.stats import spearmanr
            from sklearn.metrics import mean_absolute_error

            mae_zoom = mean_absolute_error(true_values_zoom, predictions_zoom)
            spearman_zoom, _ = spearmanr(predictions_zoom, true_values_zoom)
            zoom_title = (
                f"{func_name} vs Stockfish (Zoomed: ±1000cp)\n"
                f"Spearman: {spearman_zoom:.3f}, MAE: {mae_zoom:.0f}"
            )
        else:
            zoom_title = f"{func_name} vs Stockfish (Zoomed: ±1000cp)\nNo data in range"

        ax_zoom.set_title(zoom_title, fontsize=14)
        ax_zoom.legend()
        ax_zoom.grid(True, alpha=0.3)

        # Add statistics text for zoomed view
        zoom_textstr = f"N = {len(predictions_zoom):,}\n({len(predictions_zoom) / len(predictions) * 100:.1f}% of data)"
        ax_zoom.text(
            0.02,
            0.98,
            zoom_textstr,
            transform=ax_zoom.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

    plt.tight_layout(pad=1.0)
    plt.show()


def print_results_summary(
    all_results: dict[str, dict[str, Any]], baseline_name: str
) -> None:
    """
    Print a summary of evaluation results including mates.
    """
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    for func_name, results in all_results.items():
        is_baseline = func_name == baseline_name
        print(f"\n{func_name}{' (BASELINE)' if is_baseline else ''}:")

        # Print results
        print(f"  Mean Absolute Error: {results['mae']:.2f} centipawns")
        print(f"  Spearman correlation: {results['spearman_correlation']:.3f}")
        print(f"  Correct winner prediction: {results['correct_sign_percentage']:.1f}%")
        print(
            f"  Positions evaluated: {results['num_evaluated']:,} "
            f"(including {results['mate_positions']:,} mate positions)"
        )
        runs_per_second_k = round(results["runs_per_second"] / 1000)
        print(
            f"  Evaluation time: {results['evaluation_time']:.2f}s ({runs_per_second_k}k runs/s)"
        )

        # Print subset results for abs(true_score) < 1000
        print(
            f"\n  Subset (|score| < 1000cp): {results['subset_num_evaluated']:,} positions"
        )
        print(f"    MAE: {results['subset_mae']:.2f} centipawns")
        print(f"    Spearman correlation: {results['subset_spearman_correlation']:.3f}")
        print(
            f"    Correct winner prediction: {results['subset_correct_sign_percentage']:.1f}%"
        )

        if not is_baseline and baseline_name in all_results:
            mae_improvement = all_results[baseline_name]["mae"] - results["mae"]
            spearman_improvement = (
                results["spearman_correlation"]
                - all_results[baseline_name]["spearman_correlation"]
            )
            win_improvement = (
                results["correct_sign_percentage"]
                - all_results[baseline_name]["correct_sign_percentage"]
            )
            print(
                f"  vs baseline: MAE {mae_improvement:+.0f}, "
                f"Spearman {spearman_improvement:+.3f}, Win% {win_improvement:+.1f}"
            )

            # Print subset improvements
            subset_mae_improvement = (
                all_results[baseline_name]["subset_mae"] - results["subset_mae"]
            )
            subset_spearman_improvement = (
                results["subset_spearman_correlation"]
                - all_results[baseline_name]["subset_spearman_correlation"]
            )
            subset_win_improvement = (
                results["subset_correct_sign_percentage"]
                - all_results[baseline_name]["subset_correct_sign_percentage"]
            )
            print(
                f"    subset vs baseline: MAE {subset_mae_improvement:+.0f}, "
                f"Spearman {subset_spearman_improvement:+.3f}, Win% {subset_win_improvement:+.1f}"
            )

    # Print table summary
    print("\n" + "=" * 50)
    print("SUMMARY TABLE (All Positions)")
    print("=" * 50)

    # Table header
    print(
        f"{'Model':<25} {'MAE':>6} {'RMSE':>6} {'Spearman':>8} {'Pearson':>8} {'Win%':>6} {'k runs/s':>8} {'N':>8}"
    )
    print("-" * 90)

    # Sort models with baseline first, then alphabetically
    sorted_models = sorted(
        all_results.items(), key=lambda x: (x[0] != baseline_name, x[0])
    )

    for func_name, results in sorted_models:
        is_baseline = func_name == baseline_name
        name_display = f"{func_name}{'*' if is_baseline else ''}"

        runs_per_second_k = round(results["runs_per_second"] / 1000)
        print(
            f"{name_display:<25} "
            f"{results['mae']:>6.0f} "
            f"{results['rmse']:>6.0f} "
            f"{results['spearman_correlation']:>8.3f} "
            f"{results['pearson_correlation']:>8.3f} "
            f"{results['correct_sign_percentage']:>6.1f} "
            f"{runs_per_second_k:>8} "
            f"{results['num_evaluated']:>8,}"
        )

    print("\n* = baseline model")

    # Add subset table for abs(true_score) < 1000
    print("\n" + "=" * 50)
    print("SUMMARY TABLE (|Score| < 1000cp Subset)")
    print("=" * 50)

    # Subset table header
    print(
        f"{'Model':<25} {'MAE':>6} {'RMSE':>6} {'Spearman':>8} {'Pearson':>8} {'Win%':>6} {'k runs/s':>8} {'N':>8}"
    )
    print("-" * 90)

    for func_name, results in sorted_models:
        is_baseline = func_name == baseline_name
        name_display = f"{func_name}{'*' if is_baseline else ''}"

        runs_per_second_k = round(results["runs_per_second"] / 1000)
        print(
            f"{name_display:<25} "
            f"{results['subset_mae']:>6.0f} "
            f"{results['subset_rmse']:>6.0f} "
            f"{results['subset_spearman_correlation']:>8.3f} "
            f"{results['subset_pearson_correlation']:>8.3f} "
            f"{results['subset_correct_sign_percentage']:>6.1f} "
            f"{runs_per_second_k:>8} "
            f"{results['subset_num_evaluated']:>8,}"
        )

    print("\n* = baseline model")
