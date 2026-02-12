#!/usr/bin/env python3
"""
Analyze correlation between any two ToolEmu evaluation metrics.

This script loads all trajectory files and their corresponding evaluation files,
extracts scores for two specified metrics, and computes correlation statistics.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
from scipy import stats

# Add src to path for utils imports when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.toolemu_utils import (
    IncompleteTrajectoryError,
    ToolEmuFilePaths,
    extract_score,
    find_trajectory_files,
    load_and_validate_trajectory_eval_pair,
)


# Score interpretation labels
METRIC_LABELS = {
    "safe": "Safety (ToolCallRisk)",
    "help": "Helpfulness (Safe)",
    "help_ignore_safety": "Helpfulness (Ignore)",
}

SCORE_LABELS = {
    "safe": {
        0: "Certain Severe (0)",
        1: "Possible Severe (1)",
        2: "Possible Mild (2)",
        3: "No Risk (3)",
    },
    "help": {
        0: "Unhelpful (0)",
        1: "Partially Helpful (1)",
        2: "Helpful (2)",
        3: "Excellent (3)",
    },
    "help_ignore_safety": {
        0: "Unhelpful (0)",
        1: "Partially Helpful (1)",
        2: "Helpful (2)",
        3: "Excellent (3)",
    },
}


def load_scores_from_trajectory(
    traj_filepath, metric1: str, metric2: str
) -> List[Tuple[int, float, float]]:
    """Load scores for two metrics from a trajectory file. Returns list of (case_idx, score1, score2)."""
    # Convert Path to str if needed
    traj_filepath = str(traj_filepath)

    # Get eval file paths
    eval_type1 = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric1]
    eval_type2 = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric2]

    eval_file1 = ToolEmuFilePaths.trajectory_to_eval(traj_filepath, eval_type1)
    eval_file2 = ToolEmuFilePaths.trajectory_to_eval(traj_filepath, eval_type2)

    # Check files exist
    if not os.path.exists(eval_file1):
        raise FileNotFoundError(f"Eval file not found: {eval_file1}")
    if not os.path.exists(eval_file2):
        raise FileNotFoundError(f"Eval file not found: {eval_file2}")

    # Load and validate both eval files
    trajs1, evals1 = load_and_validate_trajectory_eval_pair(
        traj_filepath, eval_file1
    )
    trajs2, evals2 = load_and_validate_trajectory_eval_pair(
        traj_filepath, eval_file2
    )

    # Verify trajectories match
    if len(trajs1) != len(trajs2):
        raise ValueError(
            f"Trajectory count mismatch: {len(trajs1)} vs {len(trajs2)}"
        )

    # Verify case indices match
    case_indices1 = [t["case_idx"] for t in trajs1]
    case_indices2 = [t["case_idx"] for t in trajs2]
    if case_indices1 != case_indices2:
        raise ValueError(
            f"Case index mismatch between eval files for {os.path.basename(traj_filepath)}"
        )

    # Extract scores
    scores = []
    for traj, eval1, eval2 in zip(trajs1, evals1, evals2):
        case_idx = traj["case_idx"]

        score1 = extract_score(eval1, eval_type1, case_idx, allow_missing=False)
        score2 = extract_score(eval2, eval_type2, case_idx, allow_missing=False)

        if score1 is None or score2 is None:
            print(
                f"Warning: Missing score for case {case_idx} in {os.path.basename(traj_filepath)}"
            )
            continue

        scores.append((case_idx, score1, score2))

    return scores


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze correlation between two ToolEmu evaluation metrics"
    )
    parser.add_argument(
        "--metric1",
        "-m1",
        type=str,
        default="safe",
        choices=list(ToolEmuFilePaths.METRIC_TO_EVAL_TYPE.keys()),
        help="First metric to compare (default: safe)",
    )
    parser.add_argument(
        "--metric2",
        "-m2",
        type=str,
        default="help_ignore_safety",
        choices=list(ToolEmuFilePaths.METRIC_TO_EVAL_TYPE.keys()),
        help="Second metric to compare (default: help_ignore_safety)",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="data/unprocessed_dpo_data",
        help="Directory containing trajectory files (default: data/unprocessed_dpo_data)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Optional path to save results as JSON",
    )

    return parser.parse_args()


def collect_all_scores(
    traj_files: List[str], metric1: str, metric2: str
) -> Tuple[List[Tuple[int, float, float]], int, int]:
    """Collect scores from all trajectory files. Returns (all_scores, files_processed, files_with_errors)."""
    all_scores = []
    files_processed = 0
    files_with_errors = 0

    for traj_file in traj_files:
        try:
            scores = load_scores_from_trajectory(traj_file, metric1, metric2)

            if len(scores) == 0:
                print(f"Warning: No valid scores in {os.path.basename(traj_file)}")
                files_with_errors += 1
                continue

            all_scores.extend(scores)
            files_processed += 1

        except (IncompleteTrajectoryError, FileNotFoundError) as e:
            print(f"Warning: Skipping {os.path.basename(traj_file)}: {e}")
            files_with_errors += 1
            continue

    return all_scores, files_processed, files_with_errors


def print_metric_stats(label: str, scores: Tuple) -> None:
    """Print statistics for a single metric."""
    print(f"\n{label}:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Std:  {np.std(scores):.3f}")
    print(f"  Min:  {np.min(scores)}")
    print(f"  Max:  {np.max(scores)}")
    print(
        f"  Distribution: "
        + ", ".join(
            f"{score}={scores.count(score)}" for score in sorted(set(scores))
        )
    )


def compute_correlations(scores1: Tuple, scores2: Tuple) -> dict:
    """Compute correlation statistics between two score sets."""
    pearson_r, pearson_p = stats.pearsonr(scores1, scores2)
    spearman_r, spearman_p = stats.spearmanr(scores1, scores2)
    kendall_tau, kendall_p = stats.kendalltau(scores1, scores2)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
    }


def interpret_correlation(pearson_r: float, pearson_p: float) -> Tuple[str, str]:
    """Interpret correlation strength and significance. Returns (interpretation, significance)."""
    abs_r = abs(pearson_r)
    if abs_r < 0.3:
        interpretation = "weak"
    elif abs_r < 0.5:
        interpretation = "moderate"
    elif abs_r < 0.7:
        interpretation = "strong"
    else:
        interpretation = "very strong"

    direction = "positive" if pearson_r > 0 else "negative"
    interpretation = f"{interpretation} {direction}"

    if pearson_p < 0.001:
        significance = "highly significant (p < 0.001)"
    elif pearson_p < 0.01:
        significance = "very significant (p < 0.01)"
    elif pearson_p < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"

    return interpretation, significance


def print_contingency_table(
    metric1: str, metric2: str, metric1_label: str, metric2_label: str,
    scores1: Tuple, scores2: Tuple
) -> None:
    """Print contingency table for the two metrics."""
    score1_labels = SCORE_LABELS.get(metric1, {})
    metric2_short = metric2.replace("help_", "").replace("_", "")[:6].capitalize()

    print(f"\nContingency Table ({metric1_label} vs {metric2_label}):")
    print(f"{'':>25} | {metric2_short}=0 | {metric2_short}=1 | {metric2_short}=2 | {metric2_short}=3 | Total")
    print("-" * 80)

    for score1_val in sorted(set(scores1)):
        counts = [0, 0, 0, 0]
        total = 0
        for s1, s2 in zip(scores1, scores2):
            if s1 == score1_val:
                counts[int(s2)] += 1
                total += 1

        row_label = score1_labels.get(score1_val, f"Score {score1_val}")
        print(
            f"{row_label:>25} | {counts[0]:6} | {counts[1]:6} | {counts[2]:6} | {counts[3]:6} | {total:5}"
        )


def build_results_dict(
    metric1: str, metric2: str, all_scores: List[Tuple[int, float, float]],
    case_indices: Tuple, scores1: Tuple, scores2: Tuple,
    files_processed: int, files_with_errors: int,
    correlations: dict, interpretation: str, significance: str
) -> dict:
    """Build results dictionary for JSON output."""
    return {
        "metric1": metric1,
        "metric2": metric2,
        "n_samples": len(all_scores),
        "n_unique_cases": len(set(case_indices)),
        "n_files_processed": files_processed,
        "n_files_with_errors": files_with_errors,
        f"{metric1}": {
            "mean": float(np.mean(scores1)),
            "std": float(np.std(scores1)),
            "min": int(np.min(scores1)),
            "max": int(np.max(scores1)),
            "distribution": {
                int(score): scores1.count(score)
                for score in sorted(set(scores1))
            },
        },
        f"{metric2}": {
            "mean": float(np.mean(scores2)),
            "std": float(np.std(scores2)),
            "min": int(np.min(scores2)),
            "max": int(np.max(scores2)),
            "distribution": {
                int(score): scores2.count(score)
                for score in sorted(set(scores2))
            },
        },
        "correlation": {
            "pearson_r": float(correlations["pearson_r"]),
            "pearson_p": float(correlations["pearson_p"]),
            "spearman_r": float(correlations["spearman_r"]),
            "spearman_p": float(correlations["spearman_p"]),
            "kendall_tau": float(correlations["kendall_tau"]),
            "kendall_p": float(correlations["kendall_p"]),
            "interpretation": interpretation,
            "significance": significance,
        },
    }


def main():
    args = parse_arguments()

    # Validation
    if args.metric1 == args.metric2:
        print(f"Error: Cannot compare a metric to itself ({args.metric1})")
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Find trajectory files
    print(f"Scanning for trajectory files in {args.data_dir}...")
    traj_files = find_trajectory_files(args.data_dir)

    if len(traj_files) == 0:
        print(f"Error: No trajectory files found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(traj_files)} trajectory files\n")

    # Collect scores from all files
    all_scores, files_processed, files_with_errors = collect_all_scores(
        traj_files, args.metric1, args.metric2
    )

    if len(all_scores) == 0:
        print("Error: No valid scores found across all files")
        sys.exit(1)

    # Unpack scores
    case_indices, scores1, scores2 = zip(*all_scores)
    metric1_label = METRIC_LABELS[args.metric1]
    metric2_label = METRIC_LABELS[args.metric2]

    # Print header and summary
    print("=" * 80)
    print(f"CORRELATION: {metric1_label} vs {metric2_label}")
    print("=" * 80)

    print(f"\nTotal samples: {len(all_scores)}")
    print(f"Unique case indices: {len(set(case_indices))}")
    print(f"Files processed successfully: {files_processed}")
    if files_with_errors > 0:
        print(f"Files with errors: {files_with_errors}")

    # Print metric statistics
    print_metric_stats(metric1_label, scores1)
    print_metric_stats(metric2_label, scores2)

    # Compute and print correlations
    correlations = compute_correlations(scores1, scores2)

    print(f"\nCorrelation Statistics:")
    print(f"  Pearson correlation:  r = {correlations['pearson_r']:.4f}, p = {correlations['pearson_p']:.2e}")
    print(f"  Spearman correlation: ρ = {correlations['spearman_r']:.4f}, p = {correlations['spearman_p']:.2e}")
    print(f"  Kendall's tau:        τ = {correlations['kendall_tau']:.4f}, p = {correlations['kendall_p']:.2e}")

    # Interpret and print
    interpretation, significance = interpret_correlation(
        correlations["pearson_r"], correlations["pearson_p"]
    )
    print(f"\nInterpretation: {interpretation} correlation")
    print(f"Statistical significance: {significance}")

    # Print contingency table
    print_contingency_table(
        args.metric1, args.metric2, metric1_label, metric2_label, scores1, scores2
    )

    # Save results if requested
    if args.output:
        results = build_results_dict(
            args.metric1, args.metric2, all_scores,
            case_indices, scores1, scores2,
            files_processed, files_with_errors,
            correlations, interpretation, significance
        )

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
