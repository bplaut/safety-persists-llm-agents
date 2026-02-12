#!/usr/bin/env python3
"""
Re-run failed evaluations with temperature > 0.

This script identifies eval entries that failed to parse (and couldn't be fixed
via regex extraction), re-runs the evaluation for just those cases with temp > 0,
and merges the results back into the original file.

Usage:
    python slurm_scripts/rerun_failed_evals.py \
        --data-dir output/trajectories \
        --evaluator-model Qwen/Qwen3-32B \
        --quantization int4 \
        --temperature 0.3 \
        [--max-retries 20] \
        [--eval-types agent_safe agent_help_ignore_safety] \
        [--model-filter Qwen_Qwen3-8B] \
        [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for utils imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.toolemu_utils import (
    ToolEmuFilePaths,
    find_unfixable_entries,
    load_jsonl,
)
from utils.train_utils import DEFAULT_RANDOM_SEED


def merge_results_in_place(
    original_file: Path,
    new_results: Dict[int, Dict[str, Any]]
) -> None:
    """
    Replace entries in original file by line index.

    Uses atomic write for safety: writes to temp file then renames.

    Args:
        original_file: Path to the eval file to modify
        new_results: Dict mapping line_index -> new eval entry

    Raises:
        IndexError: If any line_index in new_results is out of range
    """
    if not new_results:
        return  # Nothing to merge

    # Load original file
    data = load_jsonl(str(original_file), description="eval file")

    # Validate line indices
    max_line = len(data) - 1
    for line_idx in new_results.keys():
        if line_idx < 0 or line_idx > max_line:
            raise IndexError(
                f"Line index {line_idx} out of range [0, {max_line}] for file {original_file}"
            )

    # Replace entries
    for line_idx, new_entry in new_results.items():
        data[line_idx] = new_entry

    # Write to temp file, then atomic rename
    temp_file = original_file.with_suffix('.tmp')
    try:
        with open(temp_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        temp_file.rename(original_file)
    except Exception:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise


def run_targeted_eval(
    traj_file: Path,
    case_idx: int,
    eval_type: str,
    temperature: float,
    evaluator_model: str,
    quantization: str,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Run evaluation for a single case using evaluate.py subprocess.

    Args:
        traj_file: Path to trajectory file
        case_idx: Case index to evaluate
        eval_type: Type of evaluation ('agent_safe' or 'agent_help_ignore_safety')
        temperature: Temperature for evaluator
        evaluator_model: Model to use for evaluation
        quantization: Quantization level (int4, int8)
        seed: Random seed for reproducibility

    Returns:
        The single evaluation result dict

    Raises:
        RuntimeError: If evaluation fails or produces unexpected output
    """
    # Parse eval_type into base type and help_safety_level
    base_eval_type, help_safety_level = ToolEmuFilePaths.parse_eval_type(eval_type)

    # Create temp directory for output
    with tempfile.TemporaryDirectory(prefix="rerun_eval_") as tmpdir:
        # Load the specific trajectory we need
        all_trajs = load_jsonl(str(traj_file), description="trajectory file")
        target_traj = None
        for traj in all_trajs:
            if traj.get("case_idx") == case_idx:
                target_traj = traj
                break
        if target_traj is None:
            raise RuntimeError(f"Case {case_idx} not found in {traj_file}")

        # Write single trajectory to temp file
        temp_traj_file = os.path.join(tmpdir, "single_traj.jsonl")
        with open(temp_traj_file, "w") as f:
            f.write(json.dumps(target_traj) + "\n")

        # Build evaluate.py command
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'evaluation', 'evaluate.py'),
            '--data-path', temp_traj_file,
            '--evaluator-model-name', evaluator_model,
            '--quantization', quantization,
            '--eval-type', base_eval_type,
            '--evaluator-temperature', str(temperature),
        ]
        if help_safety_level:
            cmd.extend(['--help-safety-level', help_safety_level])

        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"evaluate.py failed for case {case_idx}:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Find the output file (evaluate.py writes output next to input file)
        output_files = list(Path(tmpdir).glob("*_eval_*.jsonl"))
        if not output_files:
            raise RuntimeError(
                f"No output file found for case {case_idx} in {tmpdir}"
            )

        # Load and return the single result
        results = load_jsonl(str(output_files[0]), description="temp eval output")
        if len(results) != 1:
            raise RuntimeError(
                f"Expected 1 result for case {case_idx}, got {len(results)}"
            )

        return results[0]


def rerun_until_success(
    eval_file: Path,
    traj_file: Path,
    eval_type: str,
    evaluator_model: str,
    quantization: str,
    temperature: float,
    max_retries: int = 20,
    seed: int = DEFAULT_RANDOM_SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main retry loop: identify failures, re-run, merge, repeat.

    Args:
        eval_file: Path to evaluation file
        traj_file: Path to trajectory file
        eval_type: Type of evaluation
        evaluator_model: Model for evaluation
        quantization: Quantization level
        temperature: Temperature for re-runs
        max_retries: Maximum retry attempts
        seed: Random seed (passed to evaluator)
        verbose: Whether to print progress

    Returns:
        Dict with: success (bool), retries (int), fixed_count (int),
        permanently_unfixable (list), skipped (bool)
    """
    # Check for unfixable entries
    unfixable = find_unfixable_entries(eval_file, traj_file, eval_type)

    if not unfixable:
        if verbose:
            print(f"  No unfixable entries found - skipping")
        return {
            'success': True,
            'retries': 0,
            'fixed_count': 0,
            'permanently_unfixable': [],
            'skipped': True
        }

    if verbose:
        case_indices = [case_idx for _, case_idx in unfixable]
        print(f"  Found {len(unfixable)} unfixable entries: case_idx {case_indices}")

    total_fixed = 0
    retry_count = 0

    while unfixable and retry_count < max_retries:
        retry_count += 1

        if verbose:
            print(f"  Retry {retry_count} (temp={temperature}): ", end="", flush=True)

        fixed_this_round = 0
        new_results = {}

        for line_idx, case_idx in unfixable:
            try:
                # Run evaluation for this case
                result = run_targeted_eval(
                    traj_file=traj_file,
                    case_idx=case_idx,
                    eval_type=eval_type,
                    temperature=temperature,
                    evaluator_model=evaluator_model,
                    quantization=quantization,
                    seed=seed,
                )

                # Update eval_id to match line position
                result['eval_id'] = line_idx
                new_results[line_idx] = result

            except Exception as e:
                if verbose:
                    print(f"\n    Error evaluating case {case_idx}: {e}")
                continue

        # Merge results back
        if new_results:
            merge_results_in_place(eval_file, new_results)

        # Re-check for remaining unfixable entries
        still_unfixable = find_unfixable_entries(eval_file, traj_file, eval_type)
        fixed_this_round = len(unfixable) - len(still_unfixable)
        total_fixed += fixed_this_round

        if verbose:
            remaining = len(still_unfixable)
            print(f"{fixed_this_round}/{len(unfixable)} fixed, {remaining} remaining")

        unfixable = still_unfixable

    # Determine final status
    success = len(unfixable) == 0
    permanently_unfixable = [(line_idx, case_idx) for line_idx, case_idx in unfixable]

    if verbose:
        if success:
            print(f"  SUCCESS: All entries now parseable")
        else:
            case_indices = [case_idx for _, case_idx in permanently_unfixable]
            print(f"  GIVING UP: case_idx {case_indices} permanently unfixable after {max_retries} retries")

    return {
        'success': success,
        'retries': retry_count,
        'fixed_count': total_fixed,
        'permanently_unfixable': permanently_unfixable,
        'skipped': False
    }


def process_directory(
    data_dir: Path,
    evaluator_model: str,
    quantization: str,
    temperature: float,
    eval_types: List[str],
    max_retries: int,
    model_filter: Optional[str],
    seed: int,
    dry_run: bool,
) -> Dict[str, Any]:
    """Process all eval files in a directory."""
    from utils.toolemu_utils import find_trajectory_files

    # Find all trajectory files
    trajectory_files = find_trajectory_files(data_dir)

    if not trajectory_files:
        print(f"No trajectory files found in {data_dir}")
        return {'files_processed': 0}

    # Filter by model if specified
    if model_filter:
        trajectory_files = [f for f in trajectory_files if model_filter in str(f)]

    print(f"Found {len(trajectory_files)} trajectory files")

    # Summary stats
    total_files = 0
    total_fixed = 0
    total_unfixable = 0
    files_with_fixes = []

    for traj_file in trajectory_files:
        for eval_type in eval_types:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))

            if not eval_file.exists():
                continue

            print(f"\nProcessing: {eval_file.name}")

            if dry_run:
                # Just check for unfixable entries
                unfixable = find_unfixable_entries(eval_file, traj_file, eval_type)
                if unfixable:
                    case_indices = [case_idx for _, case_idx in unfixable]
                    print(f"  [DRY RUN] Would re-run {len(unfixable)} entries: case_idx {case_indices}")
                else:
                    print(f"  [DRY RUN] No unfixable entries")
                continue

            result = rerun_until_success(
                eval_file=eval_file,
                traj_file=traj_file,
                eval_type=eval_type,
                evaluator_model=evaluator_model,
                quantization=quantization,
                temperature=temperature,
                max_retries=max_retries,
                seed=seed,
            )

            total_files += 1
            total_fixed += result['fixed_count']
            total_unfixable += len(result['permanently_unfixable'])

            if result['fixed_count'] > 0:
                files_with_fixes.append(str(eval_file))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {total_files}")
    print(f"Total entries fixed: {total_fixed}")
    print(f"Total permanently unfixable: {total_unfixable}")

    if files_with_fixes:
        print(f"\nFiles with fixes ({len(files_with_fixes)}):")
        for f in files_with_fixes[:10]:
            print(f"  - {f}")
        if len(files_with_fixes) > 10:
            print(f"  ... and {len(files_with_fixes) - 10} more")

    return {
        'files_processed': total_files,
        'total_fixed': total_fixed,
        'total_unfixable': total_unfixable,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-run failed evaluations with temperature > 0"
    )

    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=Path('output/trajectories'),
        help='Directory containing trajectory and eval files'
    )

    parser.add_argument(
        '--evaluator-model', '-e',
        type=str,
        required=True,
        help='Model to use for evaluation (e.g., Qwen/Qwen3-32B)'
    )

    parser.add_argument(
        '--quantization', '-q',
        type=str,
        default='none',
        choices=['none', 'int4', 'int8', 'fp16'],
        help='Quantization level for evaluator'
    )

    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.3,
        help='Temperature for re-runs (default: 0.3)'
    )

    parser.add_argument(
        '--max-retries', '-n',
        type=int,
        default=20,
        help='Maximum retry attempts per entry (default: 20)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f'Random seed for evaluator (default: {DEFAULT_RANDOM_SEED})'
    )

    parser.add_argument(
        '--eval-types',
        type=str,
        nargs='+',
        default=['agent_safe', 'agent_help_ignore_safety'],
        help='Evaluation types to process (default: agent_safe agent_help_ignore_safety)'
    )

    parser.add_argument(
        '--model-filter',
        type=str,
        default=None,
        help='Only process files matching this model name pattern'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    print(f"Re-running failed evaluations")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Evaluator model: {args.evaluator_model}")
    print(f"  Quantization: {args.quantization}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max retries: {args.max_retries}")
    print(f"  Eval types: {args.eval_types}")
    if args.model_filter:
        print(f"  Model filter: {args.model_filter}")
    if args.dry_run:
        print(f"  DRY RUN MODE")

    result = process_directory(
        data_dir=args.data_dir,
        evaluator_model=args.evaluator_model,
        quantization=args.quantization,
        temperature=args.temperature,
        eval_types=args.eval_types,
        max_retries=args.max_retries,
        model_filter=args.model_filter,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
