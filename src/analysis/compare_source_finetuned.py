#!/usr/bin/env python3
"""
Compare source model and finetuned model outputs on ToolEmu evaluations.

Finds cases where source and finetuned models scored differently on two
specified metrics (default: safety and helpfulness most-ignore-safety).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_name_utils import nickname_to_directory, clean_model_name, expand_model_nickname
from utils.train_utils import compute_test_indices, extract_seed_from_path, DEFAULT_RANDOM_SEED
from utils.toolemu_utils import extract_score, ToolEmuFilePaths

# Valid metrics for comparison
VALID_METRICS = list(ToolEmuFilePaths.METRIC_TO_EVAL_TYPE.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare source and finetuned model evaluation scores"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="output/trajectories",
        help="Directory containing model output subdirectories (default: output/trajectories)",
    )
    parser.add_argument(
        "--source-model", "-b",
        type=str,
        required=True,
        help="Source model identifier or nickname (e.g., 'Qwen3-8B', 'Qwen-8B', 'Qwen_Qwen3-8B')",
    )
    parser.add_argument(
        "--finetuned-model", "-f",
        type=str,
        required=True,
        help="Finetuned model identifier (e.g., 'Qwen-8B_int4_most')",
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=None,
        help="Output file for comparison results (default: print to stdout)",
    )
    parser.add_argument(
        "--show-all", "-a",
        action="store_true",
        help="Show all cases, including those with identical scores",
    )
    parser.add_argument(
        "--min-score-diff", "-m",
        type=int,
        default=1,
        help="Minimum score difference to report (default: 1)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity level: -v for evaluator thoughts, -vv for thoughts + trajectories",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Filter to only include test set indices (infers seed from finetuned model name)",
    )
    parser.add_argument(
        "--emulator-model", "-s",
        type=str,
        required=True,
        help="Emulator model (e.g., 'Qwen_Qwen3-32B')",
    )
    parser.add_argument(
        "--evaluator-model", "-e",
        type=str,
        required=True,
        help="Evaluator model (e.g., 'Qwen_Qwen3-32B')",
    )
    parser.add_argument(
        "--metric1",
        type=str,
        default="safe",
        choices=VALID_METRICS,
        help=f"First metric to compare (default: safe). Choices: {', '.join(VALID_METRICS)}",
    )
    parser.add_argument(
        "--metric2",
        type=str,
        default="help_ignore_safety",
        choices=VALID_METRICS,
        help=f"Second metric to compare (default: help_ignore_safety). Choices: {', '.join(VALID_METRICS)}",
    )

    return parser.parse_args()




def load_eval_file(filepath: str, test_seed: Optional[int] = None) -> Dict[int, float]:
    """Load evaluation file and return mapping from case_idx to score.

    Raises:
        FileNotFoundError: If eval or trajectory file doesn't exist
        IncompleteTrajectoryError: If trajectory is incomplete (propagated to caller)
        ValueError: If validation fails
    """
    from utils.toolemu_utils import load_and_validate_all_eval_files, ToolEmuFilePaths, extract_score

    scores = {}

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Evaluation file not found: {filepath}")

    # Find corresponding trajectory file using centralized utility
    traj_filepath = ToolEmuFilePaths.eval_to_trajectory(filepath)

    if not os.path.exists(traj_filepath):
        raise FileNotFoundError(
            f"Corresponding trajectory file not found: {traj_filepath}\n"
            f"(derived from eval file: {filepath})"
        )

    # Determine eval type from filename using centralized utility
    eval_type = ToolEmuFilePaths.extract_eval_type(filepath)
    if eval_type is None:
        raise ValueError(f"Cannot determine eval type from filename: {filepath}")

    # Load and validate ALL eval files (may raise IncompleteTrajectoryError)
    trajectories, evaluations_dict = load_and_validate_all_eval_files(
        traj_filepath, test_seed=test_seed
    )

    # Get the evaluations for the requested eval type
    if eval_type not in evaluations_dict:
        raise ValueError(f"Eval type {eval_type} not found in evaluations_dict keys: {list(evaluations_dict.keys())}")

    evaluations = evaluations_dict[eval_type]

    # Extract scores from evaluations using centralized utility
    for traj, eval_data in zip(trajectories, evaluations):
        case_idx = traj['case_idx']

        # Extract score using centralized utility
        score = extract_score(eval_data, eval_type, case_idx, allow_missing=True)
        if score is None:
            print(f"Warning: Skipping case_idx {case_idx} in {os.path.basename(filepath)} (missing score)")
            continue

        scores[case_idx] = score

    return scores


def extract_evaluator_thoughts(eval_data: Optional[Dict], verbosity: int) -> str:
    """Extract evaluator thought(s) from eval data, supporting both single and multi-replicate formats.

    For verbosity=1: returns thought from replicate closest to median score (for consistency with displayed median).
    For verbosity>=2: returns all thoughts for multi-replicate, single thought for single-replicate.
    """
    if eval_data is None:
        return "N/A"

    # Multi-replicate format
    if "replicates" in eval_data:
        replicates = eval_data["replicates"]
        if not replicates:
            return "N/A"

        if verbosity == 1:
            # Find replicate closest to median score (for consistency with displayed median)
            import statistics
            scores_with_idx = []
            for i, rep in enumerate(replicates):
                score = rep.get('eval_scores', {})
                score_val = score.get('ToolCallRisk') or score.get('Helpfulness')
                if score_val is not None:
                    scores_with_idx.append((score_val, i))

            if not scores_with_idx:
                return replicates[0].get('Evaluator Thought', 'N/A')

            median_score = statistics.median([s for s, _ in scores_with_idx])
            # Find replicate with score closest to median
            best_idx = min(scores_with_idx, key=lambda x: abs(x[0] - median_score))[1]
            return replicates[best_idx].get('Evaluator Thought', 'N/A')
        else:
            # Show all replicates' thoughts
            thoughts = []
            for i, rep in enumerate(replicates):
                thought = rep.get('Evaluator Thought', 'N/A')
                score = rep.get('eval_scores', {})
                # Get the score (could be ToolCallRisk or Helpfulness)
                score_val = score.get('ToolCallRisk') or score.get('Helpfulness')
                score_str = f" (score: {score_val})" if score_val is not None else ""
                thoughts.append(f"--- Replicate {i+1}{score_str} ---\n{thought}")
            return "\n\n".join(thoughts)

    # Single-replicate format
    return eval_data.get('Evaluator Thought', 'N/A')


def load_model_data_verbose(
    output_dir: str,
    model_name: str,
    emulator_model: str,
    evaluator_model: str,
    metric1: str,
    metric2: str,
    test_seed: Optional[int] = None
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Dict], Dict[int, Dict], Dict[int, Dict]]:
    """Load all data for verbose comparison. Returns (metric1_scores, metric2_scores, trajectories, metric1_evals, metric2_evals)."""
    from utils.toolemu_utils import load_and_validate_all_eval_files, ToolEmuFilePaths, find_trajectory_files, IncompleteTrajectoryError

    # Convert metric names to eval types
    eval_type1 = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric1]
    eval_type2 = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric2]

    model_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    all_traj_files = find_trajectory_files(model_dir)

    # Filter by emulator and evaluator models
    pattern = f"*_emu-{emulator_model}_eval-{evaluator_model}_*"
    traj_files = [f for f in all_traj_files if Path(f).match(pattern)]

    if len(traj_files) == 0:
        raise FileNotFoundError(f"No trajectory files found for {model_name}")

    # Load all data
    all_trajectories = {}
    all_metric1_scores = {}
    all_metric2_scores = {}
    all_metric1_evals = {}
    all_metric2_evals = {}

    for traj_file in sorted(traj_files):
        traj_filepath = str(traj_file)

        # Check that all eval files exist
        metric1_file = ToolEmuFilePaths.trajectory_to_eval(traj_filepath, eval_type1)
        metric2_file = ToolEmuFilePaths.trajectory_to_eval(traj_filepath, eval_type2)

        if not os.path.exists(metric1_file) or not os.path.exists(metric2_file):
            continue

        # Load and validate all eval files
        try:
            trajectories, evaluations_dict = load_and_validate_all_eval_files(
                traj_filepath, test_seed=test_seed
            )
        except IncompleteTrajectoryError as e:
            print(f"Warning: Skipping incomplete trajectory {os.path.basename(traj_filepath)}: {e}")
            continue

        metric1_evals = evaluations_dict.get(eval_type1, [])
        metric2_evals = evaluations_dict.get(eval_type2, [])

        # Store data by case_idx
        for i, traj in enumerate(trajectories):
            case_idx = traj['case_idx']

            # Store trajectory
            all_trajectories[case_idx] = traj

            # Store metric1 eval and score (supports both single and multi-replicate)
            if i < len(metric1_evals):
                metric1_eval = metric1_evals[i]
                all_metric1_evals[case_idx] = metric1_eval
                metric1_score = extract_score(metric1_eval, eval_type1, case_idx, allow_missing=True)
                if metric1_score is not None:
                    all_metric1_scores[case_idx] = metric1_score

            # Store metric2 eval and score (supports both single and multi-replicate)
            if i < len(metric2_evals):
                metric2_eval = metric2_evals[i]
                all_metric2_evals[case_idx] = metric2_eval
                metric2_score = extract_score(metric2_eval, eval_type2, case_idx, allow_missing=True)
                if metric2_score is not None:
                    all_metric2_scores[case_idx] = metric2_score

    return all_metric1_scores, all_metric2_scores, all_trajectories, all_metric1_evals, all_metric2_evals


def find_eval_files(
    output_dir: str,
    model_name: str,
    eval_type: str,
    emulator_model: str,
    evaluator_model: str
) -> List[str]:
    """Find all evaluation files for a given model and eval type. Returns list of file paths."""
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pattern = f"*_emu-{emulator_model}_eval-{evaluator_model}_*_eval_{eval_type}.jsonl"
    files = list(Path(model_dir).glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No evaluation files found for {model_name} with type {eval_type}, "
            f"emulator={emulator_model}, evaluator={evaluator_model}"
        )

    return [str(f) for f in sorted(files)]


def merge_scores(file_list: List[str], test_seed: Optional[int] = None) -> Dict[int, float]:
    """Merge scores from multiple evaluation files into a single dict mapping case_idx to score."""
    from utils.toolemu_utils import IncompleteTrajectoryError

    merged = {}

    for filepath in file_list:
        try:
            scores = load_eval_file(filepath, test_seed=test_seed)
        except IncompleteTrajectoryError as e:
            print(f"Warning: Skipping incomplete file {os.path.basename(filepath)}: {e}")
            continue

        for case_idx, score in scores.items():
            if case_idx in merged:
                raise ValueError(
                    f"Duplicate case_idx {case_idx} found in {filepath}. "
                    f"Previous score: {merged[case_idx]}, new score: {score}"
                )
            merged[case_idx] = score

    return merged


def compute_mean_delta(
    base_scores: Dict[int, float],
    finetuned_scores: Dict[int, float]
) -> Tuple[float, int]:
    """Compute mean delta over all cases where both models have scores.

    Returns (mean_delta, num_cases).
    """
    matched_cases = set(base_scores.keys()) & set(finetuned_scores.keys())
    if not matched_cases:
        return 0.0, 0
    total_diff = sum(finetuned_scores[idx] - base_scores[idx] for idx in matched_cases)
    return total_diff / len(matched_cases), len(matched_cases)


def compare_scores(
    base_scores: Dict[int, float],
    finetuned_scores: Dict[int, float],
    metric_name: str,
    min_diff: int = 1
) -> List[Tuple[int, float, float, float]]:
    """Compare scores between base and finetuned models. Returns list of (case_idx, base, ft, diff) for cases with |diff| >= min_diff."""
    differences = []

    all_case_ids = set(base_scores.keys()) | set(finetuned_scores.keys())

    for case_idx in sorted(all_case_ids):
        base_score = base_scores.get(case_idx)
        finetuned_score = finetuned_scores.get(case_idx)

        if base_score is None:
            print(f"Warning: case_idx {case_idx} missing from source model {metric_name}")
            continue
        if finetuned_score is None:
            print(f"Warning: case_idx {case_idx} missing from finetuned model {metric_name}")
            continue

        diff = finetuned_score - base_score

        if abs(diff) >= min_diff:
            differences.append((case_idx, base_score, finetuned_score, diff))

    return differences


def format_case_verbose(
    case_idx: int,
    base_traj: Optional[Dict],
    ft_traj: Optional[Dict],
    base_metric1_eval: Optional[Dict],
    ft_metric1_eval: Optional[Dict],
    base_metric2_eval: Optional[Dict],
    ft_metric2_eval: Optional[Dict],
    metric1_scores: Tuple[Optional[float], Optional[float], Optional[float]],
    metric2_scores: Tuple[Optional[float], Optional[float], Optional[float]],
    metric1: str,
    metric2: str,
    verbosity: int = 2
) -> str:
    """Format verbose details for a single case. verbosity=1 for evals only, 2 for evals + trajectories."""
    lines = []
    lines.append("=" * 100)
    lines.append(f"CASE {case_idx}")
    lines.append("=" * 100)

    # Score summary
    base_m1, ft_m1, m1_diff = metric1_scores
    base_m2, ft_m2, m2_diff = metric2_scores

    # Format scores with proper handling of None values
    base_m1_str = f"{base_m1:>4.1f}" if base_m1 is not None else "  --"
    ft_m1_str = f"{ft_m1:>4.1f}" if ft_m1 is not None else "  --"
    m1_diff_str = f"{m1_diff:>+5.1f}" if m1_diff is not None else "   --"

    base_m2_str = f"{base_m2:>4.1f}" if base_m2 is not None else "  --"
    ft_m2_str = f"{ft_m2:>4.1f}" if ft_m2 is not None else "  --"
    m2_diff_str = f"{m2_diff:>+5.1f}" if m2_diff is not None else "   --"

    # Pad metric names for alignment
    m1_label = f"{metric1}:".ljust(30)
    m2_label = f"{metric2}:".ljust(30)

    lines.append(f"{m1_label}Base={base_m1_str}  FT={ft_m1_str}  Diff={m1_diff_str}")
    lines.append(f"{m2_label}Base={base_m2_str}  FT={ft_m2_str}  Diff={m2_diff_str}")
    lines.append("")

    # Trajectories (only with -vv)
    if verbosity >= 2:
        # Source model trajectory
        lines.append("-" * 100)
        lines.append("SOURCE MODEL TRAJECTORY")
        lines.append("-" * 100)
        if base_traj:
            lines.append(f"Input: {base_traj.get('input', 'N/A')}")
            lines.append(f"\nOutput: {base_traj.get('output', 'N/A')}")

            intermediate_steps = base_traj.get('intermediate_steps', [])
            if intermediate_steps:
                lines.append(f"\nIntermediate Steps ({len(intermediate_steps)} steps):")
                for i, step in enumerate(intermediate_steps):
                    action, observation = step
                    lines.append(f"\n  Step {i+1}:")
                    lines.append(f"    Action: {action}")
                    lines.append(f"    Observation: {observation[:500]}..." if len(str(observation)) > 500 else f"    Observation: {observation}")
        else:
            lines.append("N/A")
        lines.append("")

        # Finetuned model trajectory
        lines.append("-" * 100)
        lines.append("FINETUNED MODEL TRAJECTORY")
        lines.append("-" * 100)
        if ft_traj:
            lines.append(f"Input: {ft_traj.get('input', 'N/A')}")
            lines.append(f"\nOutput: {ft_traj.get('output', 'N/A')}")

            intermediate_steps = ft_traj.get('intermediate_steps', [])
            if intermediate_steps:
                lines.append(f"\nIntermediate Steps ({len(intermediate_steps)} steps):")
                for i, step in enumerate(intermediate_steps):
                    action, observation = step
                    lines.append(f"\n  Step {i+1}:")
                    lines.append(f"    Action: {action}")
                    lines.append(f"    Observation: {observation[:500]}..." if len(str(observation)) > 500 else f"    Observation: {observation}")
        else:
            lines.append("N/A")
        lines.append("")

    # Convert metric names to eval types for score extraction
    eval_type1 = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric1]
    eval_type2 = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric2]

    # Base metric1 evaluation    lines.append("-" * 100)
    base_m1_score = extract_score(base_metric1_eval, eval_type1, case_idx, allow_missing=True) if base_metric1_eval else None
    score_str = f": Score {base_m1_score:.2f}" if base_m1_score is not None else ""
    lines.append(f"SOURCE MODEL {metric1.upper()} EVALUATION{score_str}")
    lines.append("-" * 100)
    lines.append(extract_evaluator_thoughts(base_metric1_eval, verbosity))
    lines.append("")

    # Finetuned metric1 evaluation    lines.append("-" * 100)
    ft_m1_score = extract_score(ft_metric1_eval, eval_type1, case_idx, allow_missing=True) if ft_metric1_eval else None
    score_str = f": Score {ft_m1_score:.2f}" if ft_m1_score is not None else ""
    lines.append(f"FINETUNED MODEL {metric1.upper()} EVALUATION{score_str}")
    lines.append("-" * 100)
    lines.append(extract_evaluator_thoughts(ft_metric1_eval, verbosity))
    lines.append("")

    # Base metric2 evaluation    lines.append("-" * 100)
    base_m2_score = extract_score(base_metric2_eval, eval_type2, case_idx, allow_missing=True) if base_metric2_eval else None
    score_str = f": Score {base_m2_score:.2f}" if base_m2_score is not None else ""
    lines.append(f"SOURCE MODEL {metric2.upper()} EVALUATION{score_str}")
    lines.append("-" * 100)
    lines.append(extract_evaluator_thoughts(base_metric2_eval, verbosity))
    lines.append("")

    # Finetuned metric2 evaluation    lines.append("-" * 100)
    ft_m2_score = extract_score(ft_metric2_eval, eval_type2, case_idx, allow_missing=True) if ft_metric2_eval else None
    score_str = f": Score {ft_m2_score:.2f}" if ft_m2_score is not None else ""
    lines.append(f"FINETUNED MODEL {metric2.upper()} EVALUATION{score_str}")
    lines.append("-" * 100)
    lines.append(extract_evaluator_thoughts(ft_metric2_eval, verbosity))
    lines.append("")

    return "\n".join(lines)


def format_comparison_report(
    metric1_diffs: List[Tuple[int, float, float, float]],
    metric2_diffs: List[Tuple[int, float, float, float]],
    source_model: str,
    finetuned_model: str,
    metric1_mean_delta: float,
    metric1_total_cases: int,
    metric2_mean_delta: float,
    metric2_total_cases: int,
    metric1: str,
    metric2: str,
    show_all: bool = False
) -> str:
    """Format comparison results as a readable report string."""
    lines = []
    lines.append("=" * 80)
    lines.append("SOURCE vs FINETUNED MODEL COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Source model:    {source_model}")
    lines.append(f"Finetuned model: {finetuned_model}")
    lines.append(f"Metrics:         {metric1}, {metric2}")
    lines.append("")

    # Summary statistics
    m1_improved = sum(1 for _, _, _, d in metric1_diffs if d > 0)  # Higher is better
    m1_worse = sum(1 for _, _, _, d in metric1_diffs if d < 0)
    m2_improved = sum(1 for _, _, _, d in metric2_diffs if d > 0)  # Higher is better
    m2_worse = sum(1 for _, _, _, d in metric2_diffs if d < 0)

    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"{metric1} (higher is better):")
    lines.append(f"  Improved:     {m1_improved} cases")
    lines.append(f"  Worse:        {m1_worse} cases")
    lines.append(f"  Total diffs:  {len(metric1_diffs)} cases")
    lines.append(f"  Mean delta:   {metric1_mean_delta:+.2f} (over {metric1_total_cases} cases)")
    lines.append("")
    lines.append(f"{metric2} (higher is better):")
    lines.append(f"  Improved:     {m2_improved} cases")
    lines.append(f"  Worse:        {m2_worse} cases")
    lines.append(f"  Total diffs:  {len(metric2_diffs)} cases")
    lines.append(f"  Mean delta:   {metric2_mean_delta:+.2f} (over {metric2_total_cases} cases)")
    lines.append("")

    # Create combined view by case_idx
    m1_map = {case_idx: (base, ft, diff) for case_idx, base, ft, diff in metric1_diffs}
    m2_map = {case_idx: (base, ft, diff) for case_idx, base, ft, diff in metric2_diffs}

    all_case_ids = sorted(set(m1_map.keys()) | set(m2_map.keys()))

    if len(all_case_ids) == 0:
        lines.append("No score differences found.")
        return "\n".join(lines)

    # Truncate metric names for table headers (max 10 chars)
    m1_short = metric1[:10] if len(metric1) > 10 else metric1
    m2_short = metric2[:10] if len(metric2) > 10 else metric2

    # Detailed comparison
    lines.append("DETAILED COMPARISON")
    lines.append("-" * 80)
    lines.append(f"{'Case':>6} | {m1_short:^20} | {m2_short:^20} | Change")
    lines.append(f"{'ID':>6} | {'Base':>6} {'FT':>6} {'Diff':>6} | {'Base':>6} {'FT':>6} {'Diff':>6} |")
    lines.append("-" * 80)

    for case_idx in all_case_ids:
        m1_info = m1_map.get(case_idx, (None, None, None))
        m2_info = m2_map.get(case_idx, (None, None, None))

        m1_base, m1_ft, m1_diff = m1_info
        m2_base, m2_ft, m2_diff = m2_info

        # Skip if both are None (shouldn't happen given how we built all_case_ids)
        if m1_base is None and m2_base is None:
            continue

        # Determine change type
        changes = []
        if m1_diff is not None and m1_diff > 0:
            changes.append(f"+{m1_short}")
        elif m1_diff is not None and m1_diff < 0:
            changes.append(f"-{m1_short}")

        if m2_diff is not None and m2_diff > 0:
            changes.append(f"+{m2_short}")
        elif m2_diff is not None and m2_diff < 0:
            changes.append(f"-{m2_short}")

        change_str = ", ".join(changes) if changes else "No change"

        # Format metric1 scores
        if m1_base is not None:
            m1_str = f"{m1_base:>6.1f} {m1_ft:>6.1f} {m1_diff:>+6.1f}"
        else:
            m1_str = f"{'--':>6} {'--':>6} {'--':>6}"

        # Format metric2 scores
        if m2_base is not None:
            m2_str = f"{m2_base:>6.1f} {m2_ft:>6.1f} {m2_diff:>+6.1f}"
        else:
            m2_str = f"{'--':>6} {'--':>6} {'--':>6}"

        lines.append(f"{case_idx:>6} | {m1_str} | {m2_str} | {change_str}")

    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    args = parse_args()

    # Infer test_seed from finetuned model name if --test-only is used
    test_seed = None
    if args.test_only:
        test_seed = extract_seed_from_path(args.finetuned_model)
        if test_seed is None:
            raise ValueError(
                f"--test-only requires seed in finetuned model name (e.g., '_s0'), "
                f"but could not extract seed from '{args.finetuned_model}'"
            )
        print(f"Using test seed {test_seed} (from finetuned model name)")

    # Get metric names and their eval types
    metric1 = args.metric1
    metric2 = args.metric2
    eval_type1 = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric1]
    eval_type2 = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric2]

    # Expand nicknames to directory names for lookups
    # Source/emulator/evaluator use short nicknames that need expansion
    source_model_dir = nickname_to_directory(expand_model_nickname(args.source_model))
    emulator_model_dir = nickname_to_directory(expand_model_nickname(args.emulator_model))
    evaluator_model_dir = nickname_to_directory(expand_model_nickname(args.evaluator_model))
    # Finetuned models use full directory names with seeds - don't expand (which strips seeds)
    finetuned_model_dir = args.finetuned_model

    # Get display names for output
    source_display = clean_model_name(args.source_model)
    finetuned_display = clean_model_name(args.finetuned_model)

    print(f"Comparing metrics: {metric1}, {metric2}")

    # Load data based on verbose mode
    if args.verbose:
        # Verbose mode: load full data
        print(f"Loading data for source model (verbose mode): {source_display}")
        source_m1_scores, source_m2_scores, source_trajectories, source_m1_evals, source_m2_evals = \
            load_model_data_verbose(args.data_dir, source_model_dir, emulator_model_dir, evaluator_model_dir,
                                    metric1, metric2, test_seed=test_seed)
        print(f"  Found {len(source_m1_scores)} cases with {metric1} scores")
        print(f"  Found {len(source_m2_scores)} cases with {metric2} scores")

        print(f"Loading data for finetuned model (verbose mode): {finetuned_display}")
        ft_m1_scores, ft_m2_scores, ft_trajectories, ft_m1_evals, ft_m2_evals = \
            load_model_data_verbose(args.data_dir, finetuned_model_dir, emulator_model_dir, evaluator_model_dir,
                                    metric1, metric2, test_seed=test_seed)
        print(f"  Found {len(ft_m1_scores)} cases with {metric1} scores")
        print(f"  Found {len(ft_m2_scores)} cases with {metric2} scores")
    else:
        # Normal mode: load only scores
        print(f"Loading {metric1} evaluations for source model: {source_display}")
        source_m1_files = find_eval_files(
            args.data_dir, source_model_dir, eval_type1, emulator_model_dir, evaluator_model_dir
        )
        source_m1_scores = merge_scores(source_m1_files, test_seed=test_seed)
        print(f"  Found {len(source_m1_scores)} cases")

        print(f"Loading {metric1} evaluations for finetuned model: {finetuned_display}")
        ft_m1_files = find_eval_files(
            args.data_dir, finetuned_model_dir, eval_type1, emulator_model_dir, evaluator_model_dir
        )
        ft_m1_scores = merge_scores(ft_m1_files, test_seed=test_seed)
        print(f"  Found {len(ft_m1_scores)} cases")

        print(f"Loading {metric2} evaluations for source model: {source_display}")
        source_m2_files = find_eval_files(
            args.data_dir, source_model_dir, eval_type2, emulator_model_dir, evaluator_model_dir
        )
        source_m2_scores = merge_scores(source_m2_files, test_seed=test_seed)
        print(f"  Found {len(source_m2_scores)} cases")

        print(f"Loading {metric2} evaluations for finetuned model: {finetuned_display}")
        ft_m2_files = find_eval_files(
            args.data_dir, finetuned_model_dir, eval_type2, emulator_model_dir, evaluator_model_dir
        )
        ft_m2_scores = merge_scores(ft_m2_files, test_seed=test_seed)
        print(f"  Found {len(ft_m2_scores)} cases")

    # Filter to test indices if requested
    if args.test_only:
        # Compute test indices using fixed 144-case split (consistent across all datasets)
        test_indices = set(compute_test_indices(test_seed))
        print(f"\nFiltering to {len(test_indices)} test indices (seed={test_seed})")

        # Filter all score dictionaries
        source_m1_scores = {k: v for k, v in source_m1_scores.items() if k in test_indices}
        source_m2_scores = {k: v for k, v in source_m2_scores.items() if k in test_indices}
        ft_m1_scores = {k: v for k, v in ft_m1_scores.items() if k in test_indices}
        ft_m2_scores = {k: v for k, v in ft_m2_scores.items() if k in test_indices}

        # Also filter verbose mode data if present
        if args.verbose:
            source_trajectories = {k: v for k, v in source_trajectories.items() if k in test_indices}
            ft_trajectories = {k: v for k, v in ft_trajectories.items() if k in test_indices}
            source_m1_evals = {k: v for k, v in source_m1_evals.items() if k in test_indices}
            ft_m1_evals = {k: v for k, v in ft_m1_evals.items() if k in test_indices}
            source_m2_evals = {k: v for k, v in source_m2_evals.items() if k in test_indices}
            ft_m2_evals = {k: v for k, v in ft_m2_evals.items() if k in test_indices}

        print(f"  Source {metric1}: {len(source_m1_scores)} cases")
        print(f"  Source {metric2}: {len(source_m2_scores)} cases")
        print(f"  Finetuned {metric1}: {len(ft_m1_scores)} cases")
        print(f"  Finetuned {metric2}: {len(ft_m2_scores)} cases")

    # Compare scores
    print("\nComparing scores...")
    m1_diffs = compare_scores(
        source_m1_scores, ft_m1_scores, metric1, args.min_score_diff
    )
    m2_diffs = compare_scores(
        source_m2_scores, ft_m2_scores, metric2, args.min_score_diff
    )

    # Compute mean deltas over all matched cases (not just filtered diffs)
    m1_mean_delta, m1_total_cases = compute_mean_delta(source_m1_scores, ft_m1_scores)
    m2_mean_delta, m2_total_cases = compute_mean_delta(source_m2_scores, ft_m2_scores)

    # Generate summary report
    report = format_comparison_report(
        m1_diffs, m2_diffs, source_display, finetuned_display,
        m1_mean_delta, m1_total_cases,
        m2_mean_delta, m2_total_cases,
        metric1, metric2,
        args.show_all
    )

    # Add verbose details if requested
    if args.verbose:
        # Build maps for easy lookup - only cases with differences meeting min_diff threshold
        m1_map = {case_idx: (base, ft, diff) for case_idx, base, ft, diff in m1_diffs}
        m2_map = {case_idx: (base, ft, diff) for case_idx, base, ft, diff in m2_diffs}
        all_case_ids = sorted(set(m1_map.keys()) | set(m2_map.keys()))

        verbose_output = []
        verbose_output.append("\n\n")
        verbose_output.append("=" * 100)
        verbose_output.append("VERBOSE DETAILS FOR CASES WITH DIFFERENCES")
        verbose_output.append("=" * 100)
        verbose_output.append("\n")

        for case_idx in all_case_ids:
            # Get metric1 scores - use actual scores even if difference didn't meet threshold
            if case_idx in m1_map:
                m1_info = m1_map[case_idx]
            else:
                # Case is here due to metric2 diff; get actual metric1 scores
                source_s = source_m1_scores.get(case_idx)
                ft_s = ft_m1_scores.get(case_idx)
                diff_s = (ft_s - source_s) if (source_s is not None and ft_s is not None) else None
                m1_info = (source_s, ft_s, diff_s)

            # Get metric2 scores - use actual scores even if difference didn't meet threshold
            if case_idx in m2_map:
                m2_info = m2_map[case_idx]
            else:
                # Case is here due to metric1 diff; get actual metric2 scores
                source_h = source_m2_scores.get(case_idx)
                ft_h = ft_m2_scores.get(case_idx)
                diff_h = (ft_h - source_h) if (source_h is not None and ft_h is not None) else None
                m2_info = (source_h, ft_h, diff_h)

            case_verbose = format_case_verbose(
                case_idx,
                source_trajectories.get(case_idx),
                ft_trajectories.get(case_idx),
                source_m1_evals.get(case_idx),
                ft_m1_evals.get(case_idx),
                source_m2_evals.get(case_idx),
                ft_m2_evals.get(case_idx),
                m1_info,
                m2_info,
                metric1,
                metric2,
                verbosity=args.verbose
            )
            verbose_output.append(case_verbose)
            verbose_output.append("\n")

        report += "\n".join(verbose_output)

    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output_file}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
