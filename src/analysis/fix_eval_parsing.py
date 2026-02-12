#!/usr/bin/env python3
"""
Fix parsing failures in all evaluation files (safety + all helpfulness variants).

Consolidates functionality from minor_scripts:
- fix_parsing_failures.py
- fix_all_eval_variants.py
- identify_all_unfixable.py
- fix_safety_parsing.py

Uses centralized validation from model_utils for robustness.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.toolemu_utils import (
    IncompleteTrajectoryError,
    ToolEmuFilePaths,
    load_and_validate_trajectory_eval_pair,
    find_trajectory_files,
    ScoreExtractor,
    compute_eval_consistency,
)
from utils.train_utils import extract_seed_from_path


def fix_helpfulness_entry(entry: Dict[str, Any], thought: str) -> bool:
    """Fix a helpfulness evaluation entry by extracting score from thought. Returns True if fixed."""
    score = ScoreExtractor.extract_helpfulness_score(thought)
    if score is None:
        return False

    # Update entry with extracted score
    entry['eval_scores'] = {'Helpfulness': score}
    entry['Overall Quantitative Score'] = score

    # Try to extract full label
    label_match = re.search(
        r'\*\*Overall Qualitative Label\*\*:\s*(.*?)\((\d+)\)',
        thought,
        re.IGNORECASE
    )
    if label_match:
        label = label_match.group(1).strip()
        entry['Overall Qualitative Label'] = f"[{label}({score})]"

    return True


def fix_safety_entry(entry: Dict[str, Any], thought: str) -> bool:
    """Fix a safety evaluation entry by extracting score from thought. Returns True if fixed."""
    score = ScoreExtractor.extract_safety_score(thought)
    if score is None:
        return False

    # Update entry with extracted score
    entry['eval_scores'] = {'ToolCallRisk': score}  # Safety uses ToolCallRisk
    entry['Overall Quantitative Score'] = score

    # Try to extract full label
    label_match = re.search(
        r'(?:\*\*)?(?:Qualitative\s+)?Label(?:\*\*)?\s*:\s*(?:\*\*)?(.*?Risk\s*\((\d+)\))(?:\*\*)?',
        thought,
        re.IGNORECASE | re.DOTALL
    )
    if label_match:
        label = label_match.group(1).strip()
        entry['Overall Qualitative Label'] = f"[{label}]"

    return True


def fix_multi_replicate_entry(entry: Dict[str, Any], eval_type: str) -> bool:
    """Fix a multi-replicate entry by extracting scores from all replicate thoughts.

    Returns True if all replicates were successfully fixed and consistency recomputed.
    """
    replicates = entry.get('replicates', [])
    if not replicates:
        return False

    # Determine the score key and fix function based on eval type
    if eval_type == 'agent_safe':
        score_key = 'ToolCallRisk'
        extract_func = ScoreExtractor.extract_safety_score
    else:
        score_key = 'Helpfulness'
        extract_func = ScoreExtractor.extract_helpfulness_score

    extracted_scores = []

    for rep in replicates:
        # Check if this replicate already has a valid score
        existing_score = rep.get('eval_scores', {}).get(score_key)
        if existing_score is not None:
            extracted_scores.append(existing_score)
            continue

        # Try to extract score from thought
        thought = rep.get('Evaluator Thought', '')
        if not thought:
            return False  # Can't fix if any replicate has no thought

        score = extract_func(thought)
        if score is None:
            return False  # Can't fix if any replicate's score can't be extracted

        # Update the replicate with extracted score
        rep['eval_scores'] = {score_key: score}
        extracted_scores.append(score)

    # Recompute consistency stats with the fixed scores
    entry['consistency'] = compute_eval_consistency(extracted_scores)
    return True


def process_eval_file(
    traj_file: Path,
    eval_type: str,
    dry_run: bool
) -> Dict[str, Any]:
    """Process a single evaluation file, fixing parsing failures. Returns statistics dict."""
    eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))

    if not eval_file.exists():
        return {'skipped': True, 'reason': 'file not found'}

    stats = {
        'total_entries': 0,
        'failures_found': 0,
        'fixed': 0,
        'unfixable': 0,
        'unfixable_details': [],
    }

    # Load with validation (extract seed from filepath for test-only runs)
    test_seed = extract_seed_from_path(str(traj_file))
    try:
        trajectories, evaluations = load_and_validate_trajectory_eval_pair(
            str(traj_file),
            str(eval_file),
            allow_empty_scores=True,  # Allow missing scores since we're fixing them
            test_seed=test_seed,
        )
    except (IncompleteTrajectoryError, ValueError) as e:
        return {'skipped': True, 'reason': f'validation error: {e}'}

    stats['total_entries'] = len(evaluations)

    # Detect multi-replicate format
    is_multi_rep = evaluations and "replicates" in evaluations[0]

    # Determine score key and fix function based on eval type (for single-replicate)
    if eval_type == 'agent_safe':
        score_key = 'ToolCallRisk'
        fix_function = fix_safety_entry
    else:
        score_key = 'Helpfulness'
        fix_function = fix_helpfulness_entry

    modified = False

    # Process each entry
    for i, (traj, entry) in enumerate(zip(trajectories, evaluations)):
        if is_multi_rep:
            # Multi-replicate format: check if any replicate is missing scores or consistency is missing
            replicates = entry.get('replicates', [])
            needs_fix = (
                not entry.get('consistency') or
                not all(r.get('eval_scores', {}).get(score_key) is not None for r in replicates)
            )

            if needs_fix:
                stats['failures_found'] += 1

                # Try to fix using multi-replicate fixer
                if fix_multi_replicate_entry(entry, eval_type):
                    stats['fixed'] += 1
                    modified = True
                else:
                    stats['unfixable'] += 1
                    # Collect snippet from first replicate without a score
                    snippet = '[EMPTY]'
                    for rep in replicates:
                        if rep.get('eval_scores', {}).get(score_key) is None:
                            thought = rep.get('Evaluator Thought') or ''
                            snippet = thought[-300:] if thought else '[EMPTY]'
                            break
                    stats['unfixable_details'].append({
                        'line': i,
                        'eval_id': entry.get('eval_id'),
                        'case_idx': traj.get('case_idx'),
                        'reason': 'Multi-replicate: could not extract all scores',
                        'snippet': snippet
                    })
        else:
            # Single-replicate format: original logic
            if not entry.get('eval_scores') or score_key not in entry.get('eval_scores', {}):
                stats['failures_found'] += 1

                thought = entry.get('Evaluator Thought', '')
                if not thought:
                    stats['unfixable'] += 1
                    stats['unfixable_details'].append({
                        'line': i,
                        'eval_id': entry.get('eval_id'),
                        'case_idx': traj.get('case_idx'),
                        'reason': 'Empty Evaluator Thought',
                        'snippet': '[EMPTY]'
                    })
                    continue

                # Try to fix
                if fix_function(entry, thought):
                    stats['fixed'] += 1
                    modified = True
                else:
                    stats['unfixable'] += 1
                    snippet = thought[-300:] if len(thought) > 300 else thought
                    stats['unfixable_details'].append({
                        'line': i,
                        'eval_id': entry.get('eval_id'),
                        'case_idx': traj.get('case_idx'),
                        'reason': 'No pattern matched',
                        'snippet': snippet
                    })

    # Write back if modified and not dry run
    if modified and not dry_run:
        try:
            with open(eval_file, 'w') as f:
                for entry in evaluations:
                    f.write(json.dumps(entry) + '\n')

            # Re-validate after writing
            try:
                load_and_validate_trajectory_eval_pair(
                    str(traj_file),
                    str(eval_file),
                    allow_empty_scores=False,  # Now require scores
                    test_seed=test_seed,
                )
                stats['revalidated'] = True
            except (IncompleteTrajectoryError, ValueError) as e:
                stats['revalidation_error'] = str(e)
                print(f"WARNING: Revalidation failed for {eval_file.name}: {e}")
        except Exception as e:
            stats['write_error'] = str(e)
            print(f"ERROR: Failed to write {eval_file.name}: {e}")
    elif modified:
        stats['dry_run'] = True

    return stats


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fix evaluation parsing failures across all eval types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix all evaluations (dry run first)
  python src/analysis/fix_eval_parsing.py --dry-run

  # Fix for real
  python src/analysis/fix_eval_parsing.py

  # Fix only Qwen 32B evaluations
  python src/analysis/fix_eval_parsing.py --model-filter Qwen_Qwen3-32B

  # Fix only safety evaluations
  python src/analysis/fix_eval_parsing.py --eval-types agent_safe

  # Delete evaluation files with unfixable entries (dry run first)
  python src/analysis/fix_eval_parsing.py --delete-unfixable --dry-run
  python src/analysis/fix_eval_parsing.py --delete-unfixable
        """
    )

    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=Path('output/trajectories'),
        help='Directory containing trajectory files (default: output/trajectories)'
    )

    parser.add_argument(
        '--model-filter',
        type=str,
        help='Only process files containing this string (e.g., "Qwen_Qwen3-32B")'
    )

    parser.add_argument(
        '--eval-types',
        nargs='+',
        choices=ToolEmuFilePaths.EVAL_TYPES,
        default=ToolEmuFilePaths.EVAL_TYPES,
        help='Which eval types to fix (default: all)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing files'
    )

    parser.add_argument(
        '--show-unfixable',
        action='store_true',
        help='Show details of unfixable entries'
    )

    parser.add_argument(
        '--delete-unfixable',
        action='store_true',
        help='Delete evaluation files that contain unfixable entries'
    )

    return parser.parse_args()


def process_all_files(
    trajectory_files: List[Path],
    eval_types: List[str],
    model_filter: Optional[str],
    dry_run: bool
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[str], Dict[Tuple[Path, str], int]]:
    """Process all trajectory files and return statistics.

    Returns (global_stats, stats_by_eval_type, files_modified, unfixable_files).
    """
    global_stats = {
        'files_processed': 0,
        'files_skipped': 0,
        'files_modified': 0,
        'total_failures': 0,
        'total_fixed': 0,
        'total_unfixable': 0,
    }

    stats_by_eval_type = {eval_type: {
        'files_processed': 0,
        'failures': 0,
        'fixed': 0,
        'unfixable': 0,
        'unfixable_details': [],
    } for eval_type in eval_types}

    files_modified = []
    unfixable_files = {}

    for traj_file in sorted(trajectory_files):
        if model_filter and model_filter not in traj_file.name:
            continue

        global_stats['files_processed'] += 1
        file_modified = False

        for eval_type in eval_types:
            result = process_eval_file(traj_file, eval_type, dry_run)

            if result.get('skipped'):
                continue

            stats = stats_by_eval_type[eval_type]
            stats['files_processed'] += 1
            stats['failures'] += result['failures_found']
            stats['fixed'] += result['fixed']
            stats['unfixable'] += result['unfixable']
            stats['unfixable_details'].extend(result['unfixable_details'])

            global_stats['total_failures'] += result['failures_found']
            global_stats['total_fixed'] += result['fixed']
            global_stats['total_unfixable'] += result['unfixable']

            if result['unfixable'] > 0:
                eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
                unfixable_files[(eval_file, eval_type)] = result['unfixable']

            if result['fixed'] > 0:
                file_modified = True

        if file_modified:
            global_stats['files_modified'] += 1
            files_modified.append(traj_file.name)

    return global_stats, stats_by_eval_type, files_modified, unfixable_files


def print_results_by_eval_type(eval_types: List[str], stats_by_eval_type: Dict[str, Dict[str, Any]]) -> None:
    """Print results broken down by evaluation type."""
    print()
    print("=" * 80)
    print("RESULTS BY EVAL TYPE")
    print("=" * 80)

    for eval_type in eval_types:
        stats = stats_by_eval_type[eval_type]
        print(f"\n{eval_type}:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Failures found: {stats['failures']}")
        print(f"  Fixed: {stats['fixed']}")
        print(f"  Unfixable: {stats['unfixable']}")

        if stats['failures'] > 0:
            success_rate = 100 * stats['fixed'] / stats['failures']
            print(f"  Success rate: {success_rate:.1f}%")


def print_overall_summary(global_stats: Dict[str, Any], dry_run: bool) -> None:
    """Print overall summary statistics."""
    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Trajectory files processed: {global_stats['files_processed']}")
    print(f"Files with fixes: {global_stats['files_modified']}")
    print(f"Total parsing failures found: {global_stats['total_failures']}")
    print(f"Total fixed: {global_stats['total_fixed']}")
    print(f"Total unfixable: {global_stats['total_unfixable']}")

    if global_stats['total_failures'] > 0:
        success_rate = 100 * global_stats['total_fixed'] / global_stats['total_failures']
        print(f"Overall success rate: {success_rate:.1f}%")

    if dry_run and global_stats['files_modified'] > 0:
        print()
        print("=" * 80)
        print(f"DRY RUN: Would modify {global_stats['files_modified']} files")
        print("Run without --dry-run to apply changes")
        print("=" * 80)


def handle_unfixable_files(
    unfixable_files: Dict[Tuple[Path, str], int],
    delete_unfixable: bool,
    dry_run: bool
) -> None:
    """Print and optionally delete files with unfixable entries."""
    if not unfixable_files:
        return

    print()
    print("=" * 80)
    print(f"FILES WITH UNFIXABLE ENTRIES ({len(unfixable_files)} files)")
    print("=" * 80)

    # Group by eval type for clarity
    files_by_eval_type: Dict[str, List[Tuple[str, int]]] = {}
    for (eval_file, eval_type), count in unfixable_files.items():
        if eval_type not in files_by_eval_type:
            files_by_eval_type[eval_type] = []
        files_by_eval_type[eval_type].append((eval_file.name, count))

    for eval_type in sorted(files_by_eval_type.keys()):
        print(f"\n{eval_type}:")
        files_list = sorted(files_by_eval_type[eval_type])
        for filename, count in files_list:
            print(f"  {filename} ({count} unfixable)")

    if delete_unfixable:
        _delete_unfixable_files(unfixable_files, dry_run)


def _delete_unfixable_files(
    unfixable_files: Dict[Tuple[Path, str], int],
    dry_run: bool
) -> None:
    """Delete files with unfixable entries (or preview in dry run mode)."""
    if dry_run:
        print()
        print("=" * 80)
        print(f"DRY RUN: Would delete {len(unfixable_files)} files with unfixable entries")
        print("=" * 80)

        files_to_delete_by_type: Dict[str, List[Tuple[str, int]]] = {}
        for (eval_file, eval_type), count in unfixable_files.items():
            if eval_type not in files_to_delete_by_type:
                files_to_delete_by_type[eval_type] = []
            files_to_delete_by_type[eval_type].append((eval_file.name, count))

        for eval_type in sorted(files_to_delete_by_type.keys()):
            print(f"\n{eval_type}:")
            files_list = sorted(files_to_delete_by_type[eval_type])
            for filename, count in files_list:
                print(f"  Would delete: {filename} ({count} unfixable)")

        print()
        print("=" * 80)
        print("Run without --dry-run to delete files")
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print(f"DELETING {len(unfixable_files)} FILES WITH UNFIXABLE ENTRIES")
        print("=" * 80)

        deleted_count = 0
        failed_deletes = []

        for (eval_file, eval_type), count in unfixable_files.items():
            try:
                eval_file.unlink()
                deleted_count += 1
                print(f"  Deleted: {eval_file.name}")
            except Exception as e:
                failed_deletes.append((eval_file.name, str(e)))
                print(f"  ERROR deleting {eval_file.name}: {e}")

        print()
        print(f"Successfully deleted: {deleted_count}/{len(unfixable_files)} files")

        if failed_deletes:
            print(f"\nFailed to delete {len(failed_deletes)} files:")
            for filename, error in failed_deletes:
                print(f"  {filename}: {error}")


def print_modified_files(files_modified: List[str], dry_run: bool) -> None:
    """Print list of modified files."""
    if not files_modified or dry_run:
        return

    print()
    print("=" * 80)
    print(f"MODIFIED FILES ({len(files_modified)})")
    print("=" * 80)
    for filename in files_modified[:20]:
        print(f"  {filename}")
    if len(files_modified) > 20:
        print(f"  ... and {len(files_modified) - 20} more")


def print_unfixable_details(
    eval_types: List[str],
    stats_by_eval_type: Dict[str, Dict[str, Any]],
    total_unfixable: int
) -> None:
    """Print details of unfixable entries."""
    if total_unfixable == 0:
        return

    print()
    print("=" * 80)
    print(f"UNFIXABLE ENTRIES ({total_unfixable} total)")
    print("=" * 80)

    shown = 0
    for eval_type in eval_types:
        stats = stats_by_eval_type[eval_type]
        for detail in stats['unfixable_details'][:5]:
            if shown >= 15:
                break
            print(f"\nEval type: {eval_type}")
            print(f"Line: {detail['line']}, Eval ID: {detail['eval_id']}, Case: {detail['case_idx']}")
            print(f"Reason: {detail['reason']}")
            print(f"Snippet (last 300 chars):")
            print(f"  {detail['snippet'][:300]}")
            print()
            shown += 1

    if total_unfixable > shown:
        print(f"... and {total_unfixable - shown} more unfixable entries")


def main():
    args = parse_arguments()

    if not args.data_dir.exists():
        print(f"Error: Output directory not found: {args.data_dir}")
        return 1

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN MODE - No files will be modified")
        print("=" * 80)

    print("=" * 80)
    print("FIXING EVALUATION PARSING FAILURES")
    print("=" * 80)
    print(f"Output directory: {args.data_dir}")
    print(f"Eval types: {', '.join(args.eval_types)}")
    if args.model_filter:
        print(f"Model filter: {args.model_filter}")
    print()

    # Process all files
    trajectory_files = find_trajectory_files(args.data_dir)
    global_stats, stats_by_eval_type, files_modified, unfixable_files = process_all_files(
        trajectory_files,
        args.eval_types,
        args.model_filter,
        args.dry_run
    )

    # Print results
    print_results_by_eval_type(args.eval_types, stats_by_eval_type)
    print_overall_summary(global_stats, args.dry_run)
    handle_unfixable_files(unfixable_files, args.delete_unfixable, args.dry_run)
    print_modified_files(files_modified, args.dry_run)

    if args.show_unfixable:
        print_unfixable_details(args.eval_types, stats_by_eval_type, global_stats['total_unfixable'])

    return 0


if __name__ == '__main__':
    sys.exit(main())
