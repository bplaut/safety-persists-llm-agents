#!/usr/bin/env python3
"""
Partition full-dataset evaluation results into training and test sets.

This script takes evaluation results run on the complete dataset and splits them
into separate train/test directories based on a configurable random seed.

SOURCE MODEL HANDLING:
Source models without a corresponding finetuned model are skipped (unless --aggregate-only).
This ensures we only include source models that have finetuned counterparts for comparison.

Usage:
    python scripts/partition_results_by_split.py \
        --data-dir output/trajectories \
        --train-output-dir train_results \
        --test-output-dir test_results

Input structure:
    - Reads from {data-dir}/ (should point directly to trajectories directory)

Output structure:
    - Creates specified train and test output directories

Each output directory will contain:
    - All trajectory files partitioned by case_idx
    - All eval files partitioned by case_idx
    - {model}_unified_report.json with aggregated metrics per configuration
    - partition_summary.json with metadata about the split
"""

import argparse
import collections
import json
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional, Callable

import numpy as np

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utilities
from utils.model_name_utils import extract_source_model, is_finetuned_model
from utils.toolemu_utils import (
    load_and_validate_all_eval_files,
    find_trajectory_files,
    write_jsonl,
    ToolEmuFilePaths,
    IncompleteTrajectoryError,
    build_toolemu_filename,
    parse_toolemu_filename,
    TOOLEMU_FULL_DATASET_SIZE,
    aggregate_consistency_stats,
    extract_score,
)
from utils.train_utils import (
    compute_test_indices,
    partition_by_case_indices,
    extract_seed_from_path,
    DEFAULT_RANDOM_SEED,
)
from analysis.compute_persistence import compute_persistence_stats


@dataclass
class ProcessingResults:
    """Container for all accumulated state during file processing."""
    all_stats: Dict[str, Any] = field(default_factory=dict)
    successful_files: List[str] = field(default_factory=list)
    failed_files: List[Tuple[str, str]] = field(default_factory=list)
    skipped_source_models: List[Tuple[str, str]] = field(default_factory=list)
    incomplete_files: List[Tuple[str, str]] = field(default_factory=list)  # (filename, reason)
    eval_files_to_delete: List[str] = field(default_factory=list)  # eval files that can be deleted (invalid trajectories)
    missing_files_by_eval_type: Dict[str, List[str]] = field(default_factory=dict)
    train_evals_by_config: Dict[Tuple, Dict[str, List]] = field(default_factory=dict)
    test_evals_by_config: Dict[Tuple, Dict[str, List]] = field(default_factory=dict)
    seed_by_config: Dict[Tuple, int] = field(default_factory=dict)


def collect_finetuned_source_models(
    trajectory_files: List[Path],
    expected_seed: Optional[int] = None
) -> Tuple[Set[Tuple[str, str, str]], Set[str], Dict[Tuple[str, str, str], Set[int]]]:
    """Collect (source_model, emu_model, eval_model) tuples from finetuned models.

    If expected_seed is provided, skips directories with mismatched seeds
    (directories without explicit seed use DEFAULT_RANDOM_SEED).

    Returns:
        finetuned_configs: Set of (source_model, emu_model, eval_model) tuples
        skipped_dirs_due_to_seed_mismatch: Set of directory names skipped due to seed mismatch
        seeds_by_source_config: Maps (source_model, emu_model, eval_model) to set of seeds
            used by finetuned models with that source
    """
    finetuned_configs = set()
    skipped_seed_mismatch_dirs = set()
    seeds_by_source_config: Dict[Tuple[str, str, str], Set[int]] = {}

    for traj_file in trajectory_files:
        dir_name = traj_file.parent.name
        # Finetuned models are identified by having a seed suffix
        dir_seed = extract_seed_from_path(dir_name)
        if dir_seed is None:
            continue  # Skip source models (no seed = not finetuned)

        if expected_seed is not None and dir_seed != expected_seed:
            skipped_seed_mismatch_dirs.add(dir_name)
            continue  # Skip directories with mismatched seeds

        source_model = extract_source_model(dir_name)
        parsed = parse_toolemu_filename(traj_file.name)
        emu_model = parsed['emu_model']
        eval_model = parsed['eval_model']
        config_key = (source_model, emu_model, eval_model)
        finetuned_configs.add(config_key)

        # Track which seeds are used for each source config
        if config_key not in seeds_by_source_config:
            seeds_by_source_config[config_key] = set()
        seeds_by_source_config[config_key].add(dir_seed)

    return finetuned_configs, skipped_seed_mismatch_dirs, seeds_by_source_config


def partition_aligned_data(
    trajectories: List[Dict[str, Any]],
    aligned_data: List[Dict[str, Any]],
    train_indices: Set[int],
    test_indices: Set[int],
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]]]:
    """Partition data aligned with trajectories into train/test sets.

    Returns lists of (case_idx, data_item) tuples to preserve case_idx association.
    """
    if len(aligned_data) == 0:
        return [], []

    if len(trajectories) != len(aligned_data):
        raise ValueError(
            f"Length mismatch: {len(trajectories)} trajectories but "
            f"{len(aligned_data)} aligned data items"
        )

    train_data = []
    test_data = []

    for traj, data_item in zip(trajectories, aligned_data):
        case_idx = traj['case_idx']
        if case_idx in train_indices:
            train_data.append((case_idx, data_item))
        elif case_idx in test_indices:
            test_data.append((case_idx, data_item))

    return train_data, test_data


def compute_eval_statistics(
    evals: List[Tuple[int, Dict[str, Any]]],
    eval_type: str
) -> Dict[str, Any]:
    """Compute statistics (mean, std, histogram) from evaluation data.

    Args:
        evals: List of (case_idx, eval_data) tuples
        eval_type: Type of evaluation (e.g., 'agent_safe', 'agent_help_ignore_safety')

    Supports both single-replicate and multi-replicate formats via extract_score.
    """
    if not evals:
        return {}

    # Map eval_type to score key for output
    if eval_type == 'agent_safe':
        metric_name = 'ToolCallRisk'
    elif eval_type in ('agent_help', 'agent_help_ignore_safety'):
        metric_name = 'Helpfulness'
    else:
        return {}

    # Extract scores using extract_score (handles single/multi-replicate, uses median by default)
    scores = []
    for case_idx, eval_data in evals:
        score = extract_score(eval_data, eval_type, case_idx, allow_missing=True)
        if score is not None:
            scores.append(float(score))

    # Calculate sample sizes
    total_evals = len(evals)
    valid_scores = len(scores)
    missing_scores = total_evals - valid_scores

    if not scores:
        return {metric_name: {
            'mean': None,
            'std': None,
            'histogram': {},
            'binarized_histogram': {},
            'total_evals': total_evals,
            'valid_scores': valid_scores,
            'missing_scores': missing_scores
        }}

    # Compute statistics
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    histogram = dict(collections.Counter(scores))

    # Binarize scores (threshold = 2: [0,2) -> 0, [2,3] -> 1)
    threshold = 2.0
    binarized_scores = [1.0 if s >= threshold else 0.0 for s in scores]
    binarized_histogram = dict(collections.Counter(binarized_scores))

    result = {metric_name: {
        'mean': mean,
        'std': std,
        'histogram': histogram,
        'binarized_histogram': binarized_histogram,
        'total_evals': total_evals,
        'valid_scores': valid_scores,
        'missing_scores': missing_scores
    }}

    # Add consistency stats if any entries have multi-replicate format
    eval_datas = [d for _, d in evals]
    has_multi_rep = any("replicates" in d for d in eval_datas)
    if has_multi_rep:
        consistency_stats = aggregate_consistency_stats(eval_datas)
        if consistency_stats:
            result['consistency'] = consistency_stats

    return result


def create_unified_report(
    evals_dict: Dict[str, List[Tuple[int, Dict[str, Any]]]]
) -> Dict[str, Any]:
    """Create a unified report with metrics from evaluations.

    Args:
        evals_dict: Maps eval_type to list of (case_idx, eval_data) tuples
    """
    report = {}
    for eval_type in ToolEmuFilePaths.EVAL_TYPES:
        if eval_type in evals_dict:
            stats = compute_eval_statistics(evals_dict[eval_type], eval_type)
            report[eval_type] = stats
    return report


def process_trajectory_file(
    traj_file: Path,
    results_dir: Path,
    train_dir: Path,
    test_dir: Optional[Path],
    train_indices: Set[int],
    test_indices: Set[int],
    trajectories: List[Dict[str, Any]],
    evals_dict: Dict[str, List[Dict[str, Any]]],
    verbose: bool = True,
    output_model_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single trajectory file and its eval files. Returns statistics dict.

    output_model_subdir: If provided, overrides the model subdirectory in output paths.
        Used for source models that need a seed suffix (e.g., Qwen_Qwen3-8B_s42).
    """
    if verbose:
        suffix = f" (→ {output_model_subdir})" if output_model_subdir else ""
        print(f"Processing: {traj_file.name}{suffix}")

    # Get relative path to preserve directory structure
    rel_path = traj_file.relative_to(results_dir)

    # Override model subdirectory if specified (for source models with seed suffix)
    if output_model_subdir is not None:
        # rel_path is like "Qwen_Qwen3-8B/filename.jsonl"
        # Replace the directory part with output_model_subdir
        rel_path = Path(output_model_subdir) / rel_path.name

    # Partition trajectories
    train_trajs, test_trajs, unknown_indices = partition_by_case_indices(
        trajectories, train_indices, test_indices
    )

    # Write partitioned trajectory files (inside trajectories/ subdirectory)
    train_traj_path = train_dir / 'trajectories' / rel_path
    write_jsonl(train_trajs, str(train_traj_path))

    if test_dir is not None:
        test_traj_path = test_dir / 'trajectories' / rel_path
        write_jsonl(test_trajs, str(test_traj_path))

    # Track statistics
    stats = {
        'trajectory': {
            'original': len(trajectories),
            'train': len(train_trajs),
            'test': len(test_trajs),
            'skipped': len(unknown_indices)
        }
    }

    # Process each eval type
    for eval_type in ToolEmuFilePaths.EVAL_TYPES:
        evals = evals_dict[eval_type]

        # Partition evaluations (aligned with trajectories)
        train_evals, test_evals = partition_aligned_data(
            trajectories, evals, train_indices, test_indices
        )

        # Construct eval file paths (inside trajectories/ subdirectory)
        eval_filename = ToolEmuFilePaths.trajectory_to_eval(rel_path.name, eval_type)
        eval_rel_path = rel_path.parent / eval_filename

        # Unpack tuples for writing (write_jsonl expects list of dicts)
        train_eval_path = train_dir / 'trajectories' / eval_rel_path
        write_jsonl([e for _, e in train_evals], str(train_eval_path))

        if test_dir is not None:
            test_eval_path = test_dir / 'trajectories' / eval_rel_path
            write_jsonl([e for _, e in test_evals], str(test_eval_path))

        # Track statistics
        stats[eval_type] = {
            'original': len(evals),
            'train': len(train_evals),
            'test': len(test_evals),
            'skipped': len(evals) - len(train_evals) - len(test_evals)
        }

    # Validate counts (with skipped cases)
    for file_type, counts in stats.items():
        if counts['train'] + counts['test'] + counts['skipped'] != counts['original']:
            raise ValueError(
                f"Count mismatch for {file_type} in {traj_file.name}: "
                f"{counts['train']} + {counts['test']} + {counts['skipped']} != {counts['original']}"
            )

    if verbose:
        msg = f"  Train: {stats['trajectory']['train']} cases, Test: {stats['trajectory']['test']} cases"
        if unknown_indices:
            msg += f", Skipped: {len(unknown_indices)} cases (outside 0-143 range)"
        print(msg)

    return {str(rel_path): stats}


def create_summary_report(
    train_indices: Set[int],
    test_indices: Set[int],
    all_stats: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Create partition summary report."""
    return {
        'partition_timestamp': datetime.now().isoformat(),
        'split_seed': seed,
        'num_train_cases': len(train_indices),
        'num_test_cases': len(test_indices),
        'train_case_indices': sorted(train_indices),
        'test_case_indices': sorted(test_indices),
        'files_processed': list(all_stats.keys()),
        'num_files_processed': len(all_stats),
        'file_statistics': all_stats
    }


def write_unified_reports(evals_by_config: Dict, output_dir: Path, verbose: bool = False):
    """Write unified reports for all configurations in evals_by_config."""
    for config_key in sorted(evals_by_config.keys()):
        model_subdir, emu_model, eval_model, quant = config_key

        # Create unified report filename
        unified_filename = build_toolemu_filename(
            agent_model=model_subdir,
            emu_model=emu_model,
            eval_model=eval_model,
            quantization=quant,
            suffix="_unified_report.json"
        )

        # Create and write unified report
        report = create_unified_report(evals_by_config[config_key])
        report_path = output_dir / unified_filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        if verbose:
            print(f"  {report_path}")


# =============================================================================
# Persistence Computation with Bootstrap CIs
# =============================================================================



def collect_eval_files_for_trajectory(traj_file: Path) -> List[str]:
    """Collect all existing eval files associated with a trajectory file."""
    eval_files = []
    for eval_type in ToolEmuFilePaths.EVAL_TYPES:
        eval_path = ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type)
        if Path(eval_path).exists():
            eval_files.append(eval_path)
    return eval_files


def prompt_and_delete_files(files_to_delete: List[str]) -> int:
    """Prompt user for confirmation and delete files. Returns number of files deleted."""
    if not files_to_delete:
        return 0

    print()
    print("=" * 80)
    print(f"FILES TO DELETE ({len(files_to_delete)} eval files from invalid trajectories):")
    print("=" * 80)
    for f in sorted(files_to_delete):
        print(f"  {f}")
    print()

    response = input("Delete these files? [y/N]: ").strip().lower()
    if response == 'y':
        deleted_count = 0
        for f in files_to_delete:
            try:
                Path(f).unlink()
                deleted_count += 1
            except OSError as e:
                print(f"  ERROR deleting {f}: {e}")
        print(f"Deleted {deleted_count} files.")
        return deleted_count
    else:
        print("No files deleted.")
        return 0


def process_all_trajectory_files(
    trajectory_files: List[Path],
    train_indices_by_seed: Dict[int, Set[int]],
    test_indices_by_seed: Dict[int, Set[int]],
    train_dir: Path,
    test_dir: Optional[Path],
    results_dir: Path,
    finetuned_source_models: Set[Tuple[str, str, str]],
    aggregate_only: bool,
    verbose: bool,
    seeds_by_source_config: Optional[Dict[Tuple[str, str, str], Set[int]]] = None,
    delete_invalid_evals: bool = False,
) -> ProcessingResults:
    """Process all trajectory files and return accumulated results.

    train_indices_by_seed/test_indices_by_seed: Cached indices per seed.
    seeds_by_source_config: Maps source configs to seeds used by finetuned models.
        Only used in partition mode (not aggregate_only).
    """
    results = ProcessingResults()
    results.missing_files_by_eval_type = {eval_type: [] for eval_type in ToolEmuFilePaths.EVAL_TYPES}

    if verbose:
        print("Processing trajectory files...")
        print("-" * 80)

    for traj_file in trajectory_files:
        try:
            dir_name = traj_file.parent.name

            is_source = not is_finetuned_model(dir_name)

            # Determine which seeds to process this file for
            if is_source and not aggregate_only:
                # Source models in partition mode: process for each finetuned seed
                source_model = extract_source_model(dir_name)
                parsed = parse_toolemu_filename(traj_file.name)
                emu_model = parsed['emu_model']
                eval_model = parsed['eval_model']
                source_config = (source_model, emu_model, eval_model)

                # Look up seeds from finetuned models
                seeds_to_process = seeds_by_source_config.get(source_config) if seeds_by_source_config else None
                if not seeds_to_process:
                    # No finetuned models with seeds for this source - skip entirely
                    if verbose:
                        print(f"  SKIPPED (no finetuned model with seed): {traj_file.name}")
                    results.skipped_source_models.append((
                        traj_file.name,
                        f"No finetuned model with explicit seed for {source_model} + emu={emu_model} + eval={eval_model}"
                    ))
                    continue
            elif is_source and aggregate_only:
                # Aggregate mode: process source models without seed suffix
                seeds_to_process = [None]  # None means no seed suffix
            else:
                # Finetuned models: extract seed from directory name
                seed = extract_seed_from_path(dir_name)
                if seed is None:
                    raise ValueError(f"Finetuned model directory missing seed suffix (_s{{N}}): {dir_name}")
                seeds_to_process = [seed]

            # Load trajectory and all eval files once per trajectory file
            # (before the seed loop to avoid re-reading for multi-seed source models)
            try:
                # Only pass test_seed for finetuned models (which may have been run with --test-only)
                # Source models always contain all cases, so no test_seed validation needed
                # Aggregate mode also doesn't need test_seed validation
                if aggregate_only or is_source:
                    validation_seed = None
                else:
                    validation_seed = list(seeds_to_process)[0]
                trajectories, evals_dict = load_and_validate_all_eval_files(
                    str(traj_file),
                    test_seed=validation_seed,
                    verbose=verbose
                )
            except IncompleteTrajectoryError as e:
                # Skip incomplete trajectories (missing evals or count mismatch)
                results.incomplete_files.append((traj_file.name, str(e)))
                if verbose:
                    print(f"  SKIPPED (incomplete): {traj_file.name}: {e}")
                continue  # Skip this file entirely

            # Track which optional eval types are missing for this trajectory file
            for eval_type in ToolEmuFilePaths.EVAL_TYPES:
                if len(evals_dict[eval_type]) == 0:
                    results.missing_files_by_eval_type[eval_type].append(traj_file.name)

            # Extract configuration from filename (same for all seeds)
            parsed = parse_toolemu_filename(traj_file.name)
            emu_model = parsed['emu_model']
            eval_model = parsed['eval_model']
            quant = parsed['quantization']

            # Validate agent model matches directory
            filename_agent = parsed['agent_model']
            if filename_agent != dir_name:
                raise ValueError(
                    f"Agent model mismatch: filename has '{filename_agent}' "
                    f"but file is in directory '{dir_name}'"
                )

            # Process this trajectory file for each seed
            for seed_to_use in seeds_to_process:
                # Determine output directory name and train/test indices
                if aggregate_only:
                    # Aggregate mode: no partitioning, all cases go to output
                    model_subdir = dir_name
                    train_indices = set(range(TOOLEMU_FULL_DATASET_SIZE))
                    test_indices = set()
                    seed = None  # No seed tracking in aggregate mode
                else:
                    # Partition mode: compute train/test split based on seed
                    seed = seed_to_use
                    if is_source:
                        model_subdir = f"{dir_name}_s{seed}"
                    else:
                        model_subdir = dir_name

                    # Get train/test indices for this seed (compute if not cached)
                    if seed not in train_indices_by_seed:
                        test_indices = set(compute_test_indices(seed))
                        train_indices = set(range(TOOLEMU_FULL_DATASET_SIZE)) - test_indices
                        train_indices_by_seed[seed] = train_indices
                        test_indices_by_seed[seed] = test_indices

                    train_indices = train_indices_by_seed[seed]
                    test_indices = test_indices_by_seed[seed]

                config_key = (model_subdir, emu_model, eval_model, quant)

                # Initialize dictionaries for this config if needed
                if config_key not in results.train_evals_by_config:
                    results.train_evals_by_config[config_key] = {eval_type: [] for eval_type in ToolEmuFilePaths.EVAL_TYPES}
                    results.test_evals_by_config[config_key] = {eval_type: [] for eval_type in ToolEmuFilePaths.EVAL_TYPES}
                    if seed is not None:
                        results.seed_by_config[config_key] = seed
                elif seed is not None and results.seed_by_config.get(config_key) != seed:
                    raise ValueError(
                        f"Seed mismatch for config {config_key}: "
                        f"previously saw seed {results.seed_by_config[config_key]}, "
                        f"but {traj_file.name} uses seed {seed}"
                    )

                # Partition evaluations and accumulate by configuration
                for eval_type in ToolEmuFilePaths.EVAL_TYPES:
                    evals = evals_dict[eval_type]
                    train_evals, test_evals = partition_aligned_data(
                        trajectories, evals, train_indices, test_indices
                    )
                    results.train_evals_by_config[config_key][eval_type].extend(train_evals)
                    results.test_evals_by_config[config_key][eval_type].extend(test_evals)

                # Process trajectory file with modified model_subdir for output
                file_stats = process_trajectory_file(
                    traj_file,
                    results_dir,
                    train_dir,
                    test_dir,
                    train_indices,
                    test_indices,
                    trajectories,
                    evals_dict,
                    verbose=verbose,
                    output_model_subdir=model_subdir,
                )
                results.all_stats.update(file_stats)

            # Track successful processing (once per trajectory file, not per seed)
            results.successful_files.append(traj_file.name)

        except Exception as e:
            results.failed_files.append((traj_file.name, str(e)))
            print(f"  ERROR processing {traj_file.name}: {e}")
            # Collect associated eval files for potential deletion
            if delete_invalid_evals:
                eval_files = collect_eval_files_for_trajectory(traj_file)
                results.eval_files_to_delete.extend(eval_files)

    if verbose:
        print("-" * 80)

    return results


def report_incomplete_files(
    incomplete_files: List[Tuple[str, str]],
    verbose: bool
) -> None:
    """Print summary of incomplete files that were skipped."""
    if incomplete_files:
        print()
        print(f"Skipped {len(incomplete_files)} incomplete file(s):")
        if verbose:
            for filename, reason in incomplete_files:
                print(f"  - {filename}: {reason}")
        else:
            # Group by reason type for compact summary
            print(f"  (use --verbose to see details)")
        print()


def print_final_summary(
    trajectory_files: List[Path],
    results: ProcessingResults,
    train_indices_by_seed: Dict[int, Set[int]],
    test_indices_by_seed: Dict[int, Set[int]],
    train_dir: Path,
    test_dir: Optional[Path],
    aggregate_only: bool,
) -> None:
    """Print the final summary of partitioning/aggregation results."""
    seeds = sorted(train_indices_by_seed.keys()) if train_indices_by_seed else []

    print("=" * 80)
    if aggregate_only:
        print("AGGREGATION COMPLETE")
    else:
        print("PARTITIONING COMPLETE")
    print("=" * 80)
    print(f"Total files found: {len(trajectory_files)}")
    if aggregate_only:
        print(f"Successfully aggregated: {len(results.successful_files)}")
    else:
        print(f"Successfully partitioned: {len(results.successful_files)}")
    print(f"Skipped (no matching finetuned model): {len(results.skipped_source_models)}")
    print(f"Failed: {len(results.failed_files)}")

    if results.skipped_source_models:
        print()
        print("Skipped source model files (no matching finetuned model):")
        # Group by reason to avoid repetitive output for parallel splits
        files_by_reason: Dict[str, List[str]] = {}
        for filename, reason in results.skipped_source_models:
            if reason not in files_by_reason:
                files_by_reason[reason] = []
            files_by_reason[reason].append(filename)
        for reason, filenames in sorted(files_by_reason.items()):
            print(f"  {len(filenames)} file(s): {reason}")

    if results.failed_files:
        print()
        print("Failed files:")
        for filename, reason in results.failed_files:
            print(f"  {filename}")
            print(f"     Reason: {reason}")

    # Report missing evaluation files (only for commonly used eval types)
    commonly_used_eval_types = [
        'agent_safe',
        'agent_help_ignore_safety'
    ]
    total_missing = sum(
        len(files) for eval_type, files in results.missing_files_by_eval_type.items()
        if eval_type in commonly_used_eval_types
    )
    if total_missing > 0:
        print()
        print("=" * 80)
        print("MISSING EVALUATION FILES SUMMARY")
        print("=" * 80)
        print(f"Total trajectory files with missing evaluations: {total_missing} file-eval pairs")
        print()
        for eval_type in commonly_used_eval_types:
            missing_files = results.missing_files_by_eval_type.get(eval_type, [])
            if missing_files:
                print(f"{eval_type}: {len(missing_files)} file(s) missing")
                # Group by model to show cleaner summary
                model_counts: Dict[str, int] = {}
                for filename in missing_files:
                    # Extract model name (everything before _emu-)
                    model_match = re.match(r'(.+?)_emu-', filename)
                    if model_match:
                        model_name = model_match.group(1)
                    else:
                        model_name = "unknown"
                    model_counts[model_name] = model_counts.get(model_name, 0) + 1

                for model_name, count in sorted(model_counts.items()):
                    print(f"  - {model_name}: {count} file(s)")
        print()

    if results.successful_files:
        print()
        if aggregate_only:
            print(f"Output directory: {train_dir}")
            print(f"  - {len(results.successful_files)} trajectory files")
        elif seeds:
            if len(seeds) == 1:
                seed = seeds[0]
                n_train = len(train_indices_by_seed.get(seed, set()))
                n_test = len(test_indices_by_seed.get(seed, set()))
                print(f"Split seed: {seed}")
                print(f"Train directory: {train_dir}")
                print(f"  - {n_train} train cases, {n_test} test cases")
                print(f"  - {len(results.successful_files)} trajectory files")
            else:
                print(f"Split seeds: {seeds}")
                for seed in seeds:
                    n_train = len(train_indices_by_seed.get(seed, set()))
                    n_test = len(test_indices_by_seed.get(seed, set()))
                    print(f"  Seed {seed}: {n_train} train cases, {n_test} test cases")
                print(f"Train directory: {train_dir}")
                print(f"Test directory: {test_dir}")
                print(f"  - {len(results.successful_files)} trajectory files")
    print()


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Partition evaluation results into train/test sets and/or aggregate parallel runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        required=True,
        help='Input directory containing trajectory files (should point directly to trajectories directory, e.g., output/trajectories)'
    )
    parser.add_argument(
        '--aggregate-only',
        action='store_true',
        help='Aggregate parallel runs without train/test partitioning'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (required for --aggregate-only mode)'
    )
    parser.add_argument(
        '--train-output-dir',
        type=str,
        help='Output directory for training set (required for partition mode)'
    )
    parser.add_argument(
        '--test-output-dir',
        type=str,
        help='Output directory for test set (required for partition mode)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (default: minimal output)'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=0,
        help='Number of bootstrap iterations for persistence CIs (default: 0, disabled)'
    )
    parser.add_argument(
        '--skip-persistence',
        action='store_true',
        help='Skip persistence computation (for faster runs)'
    )
    parser.add_argument(
        '--delete-invalid-evals',
        action='store_true',
        help='Prompt to delete eval files for invalid trajectories (count mismatch, missing evals)'
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.aggregate_only:
        if not args.output_dir:
            parser.error('--output-dir is required when using --aggregate-only')
    else:
        if not args.train_output_dir or not args.test_output_dir:
            parser.error('--train-output-dir and --test-output-dir are required for partition mode')

    return args


def main():
    args = parse_arguments()

    # Validate inputs and construct paths
    results_dir = Path(args.data_dir)

    if not results_dir.exists():
        print(f"ERROR: Directory not found: {results_dir}")
        sys.exit(1)

    print("=" * 80)
    if args.aggregate_only:
        print("AGGREGATING PARALLEL RUNS")
    else:
        print("PARTITIONING EVALUATION RESULTS BY TRAIN/TEST SPLIT")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print()

    # Find trajectory files
    if args.verbose:
        print("Scanning for trajectory files...")
    trajectory_files = find_trajectory_files(results_dir)

    if len(trajectory_files) == 0:
        print(f"ERROR: No trajectory files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(trajectory_files)} trajectory files")
    if args.verbose:
        print()

    # Collect finetuned model configurations (for source model filtering)
    # No expected_seed validation - we handle multiple seeds automatically
    finetuned_source_models, _, seeds_by_source_config = collect_finetuned_source_models(
        trajectory_files,
        expected_seed=None
    )
    if args.verbose and not args.aggregate_only:
        print(f"Found {len(finetuned_source_models)} finetuned model configurations")
        if seeds_by_source_config:
            all_seeds = set()
            for seeds in seeds_by_source_config.values():
                all_seeds.update(seeds)
            print(f"Seeds used by finetuned models: {sorted(all_seeds)}")
        print()

    # Mode-specific setup
    # In partition mode, indices are computed per-file based on directory seed
    train_indices_by_seed: Dict[int, Set[int]] = {}
    test_indices_by_seed: Dict[int, Set[int]] = {}

    if args.aggregate_only:
        # Aggregate mode: no partitioning needed
        train_dir = Path(args.output_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir = None
    else:
        # Partition mode: indices computed per-seed during processing
        train_dir = Path(args.train_output_dir)
        test_dir = Path(args.test_output_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        if args.verbose:
            print(f"Output directories:")
            print(f"  Train: {train_dir}")
            print(f"  Test: {test_dir}")
            print()

    # Process all trajectory files
    results = process_all_trajectory_files(
        trajectory_files,
        train_indices_by_seed,
        test_indices_by_seed,
        train_dir,
        test_dir,
        results_dir,
        finetuned_source_models,
        args.aggregate_only,
        args.verbose,
        seeds_by_source_config=seeds_by_source_config if not args.aggregate_only else None,
        delete_invalid_evals=args.delete_invalid_evals,
    )

    # Report warnings
    report_incomplete_files(results.incomplete_files, args.verbose)

    # Combine all train/test indices across seeds (used by summary reports and final summary)
    all_train_indices: Set[int] = set()
    all_test_indices: Set[int] = set()
    for seed_train in train_indices_by_seed.values():
        all_train_indices |= seed_train
    for seed_test in test_indices_by_seed.values():
        all_test_indices |= seed_test
    unique_seeds = sorted(train_indices_by_seed.keys()) if train_indices_by_seed else [DEFAULT_RANDOM_SEED]

    # Generate summary reports (only in partition mode)
    if not args.aggregate_only:
        if args.verbose:
            print("Generating summary reports...")

        summary = create_summary_report(
            all_train_indices,
            all_test_indices,
            results.all_stats,
            unique_seeds[0],  # Primary seed for backwards compatibility
        )
        # Add all seeds to summary
        summary['all_seeds'] = unique_seeds

        train_summary_path = train_dir / 'partition_summary.json'
        test_summary_path = test_dir / 'partition_summary.json'

        with open(train_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        with open(test_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if args.verbose:
            print(f"  {train_summary_path}")
            print(f"  {test_summary_path}")
            print()

    # Generate unified reports (one per configuration)
    if args.verbose:
        print("Generating unified metric reports...")
    write_unified_reports(results.train_evals_by_config, train_dir, args.verbose)
    if not args.aggregate_only:
        write_unified_reports(results.test_evals_by_config, test_dir, args.verbose)
    if args.verbose:
        print()

    # Compute persistence stats with bootstrap CIs (for test set in partition mode)
    if not args.aggregate_only and not args.skip_persistence:
        if args.verbose:
            print(f"Computing persistence stats with {args.n_bootstrap} bootstrap iterations...")
        persistence_stats = compute_persistence_stats(
            results.test_evals_by_config,
            results.seed_by_config,
            n_bootstrap=args.n_bootstrap,
        )
        persistence_path = test_dir / 'persistence_stats.json'
        with open(persistence_path, 'w') as f:
            json.dump(persistence_stats, f, indent=2)
        if args.verbose:
            print(f"  {persistence_path}")
            print()

    # Print final summary
    print_final_summary(
        trajectory_files, results, train_indices_by_seed, test_indices_by_seed,
        train_dir, test_dir, args.aggregate_only
    )

    # Prompt to delete eval files from invalid trajectories
    if results.eval_files_to_delete:
        prompt_and_delete_files(results.eval_files_to_delete)


if __name__ == '__main__':
    main()
