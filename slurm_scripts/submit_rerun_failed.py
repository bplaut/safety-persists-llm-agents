#!/usr/bin/env python3
"""Submit SLURM jobs for re-running failed evaluations with temperature > 0.

Scans for eval files with unfixable entries and submits jobs to re-run them.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slurm_utils import get_evaluator_model, get_sbatch_args
from utils.toolemu_utils import (
    ToolEmuFilePaths,
    find_trajectory_files,
    find_unfixable_entries,
)


def scan_for_unfixable_files(
    data_dir: Path,
    eval_types: List[str],
    model_filter: str = None,
) -> Tuple[Dict[str, List[Tuple[Path, Path, str, int]]], List[Tuple[Path, str, str]]]:
    """Scan for eval files with unfixable entries.

    Returns:
        - Dict mapping dir_path -> list of (traj_file, eval_file, eval_type, unfixable_count).
        - List of (traj_file, eval_type, error_msg) for files that failed to process.
    """
    trajectory_files = find_trajectory_files(data_dir)

    if model_filter:
        trajectory_files = [f for f in trajectory_files if model_filter in str(f)]

    results_by_dir: Dict[str, List[Tuple[Path, Path, str, int]]] = {}
    failed_files: List[Tuple[Path, str, str]] = []

    for traj_file in trajectory_files:
        for eval_type in eval_types:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if not eval_file.exists():
                continue

            try:
                unfixable = find_unfixable_entries(eval_file, traj_file, eval_type)
            except Exception as e:
                failed_files.append((traj_file, eval_type, str(e)))
                continue

            if unfixable:
                # Group by parent directory
                dir_key = str(traj_file.parent)
                if dir_key not in results_by_dir:
                    results_by_dir[dir_key] = []
                results_by_dir[dir_key].append((traj_file, eval_file, eval_type, len(unfixable)))

    return results_by_dir, failed_files


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for re-running failed evaluations"
    )
    parser.add_argument("--data-dir", "-d", type=Path, default=Path("output/trajectories"))
    parser.add_argument("--quantization", "-q", type=str, default="none",
                        choices=["none", "int4", "int8", "fp16"])
    parser.add_argument("--temperature", "-t", type=float, default=0.3)
    parser.add_argument("--max-retries", "-n", type=int, default=20)
    parser.add_argument("--model-filter", type=str, default=None,
                        help="Only process files matching this pattern")
    parser.add_argument("--eval-types", nargs="+",
                        default=["agent_safe", "agent_help_ignore_safety"],
                        help="Evaluation types to check")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without submitting jobs")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        sys.exit(f"Error: Directory not found: {args.data_dir}")

    print(f"Scanning {args.data_dir} for unfixable entries...")
    unfixable_by_dir, failed_files = scan_for_unfixable_files(
        args.data_dir,
        args.eval_types,
        args.model_filter,
    )

    # Print failed files (similar to incomplete files in submit_eval.py)
    if failed_files:
        print(f"\nFailed to process ({len(failed_files)}):")
        for f, eval_type, err in failed_files[:5]:
            print(f"  {f.name} ({eval_type}): {err}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    if not unfixable_by_dir:
        print("No unfixable entries found!")
        sys.exit(0)

    # Summary
    total_files = sum(len(files) for files in unfixable_by_dir.values())
    total_unfixable = sum(sum(f[3] for f in files) for files in unfixable_by_dir.values())
    print(f"\nFound {total_unfixable} unfixable entries across {total_files} eval files in {len(unfixable_by_dir)} directories")

    if args.verbose:
        print("\nDetails:")
        for dir_path, files in sorted(unfixable_by_dir.items()):
            print(f"\n  {dir_path}:")
            for traj_file, eval_file, eval_type, count in files:
                print(f"    {eval_file.name} ({eval_type}): {count} unfixable")

    if args.dry_run:
        print("\n[DRY RUN] Would submit jobs for each directory:")
        for dir_path, files in sorted(unfixable_by_dir.items()):
            # Get evaluator from first trajectory file
            first_traj = files[0][0]
            evaluator = get_evaluator_model(first_traj)
            print(f"  {dir_path} (evaluator: {evaluator})")
        sys.exit(0)

    if input("\nSubmit SLURM jobs? (y/n): ").strip().lower() != "y":
        sys.exit("Aborted.")

    # Submit one job per directory (the script processes all files in directory)
    jobs = []
    run_script = str(Path(__file__).parent / "run_rerun_failed.sh")

    for i, (dir_path, files) in enumerate(sorted(unfixable_by_dir.items()), 1):
        # Get evaluator model from first trajectory file in this directory
        first_traj = files[0][0]
        evaluator = get_evaluator_model(first_traj)
        print(f"[{i}/{len(unfixable_by_dir)}] {dir_path} (evaluator: {evaluator})")
        try:
            cmd = ["sbatch"] + get_sbatch_args(evaluator, args.quantization) + [
                run_script,
                dir_path,
                evaluator,
                args.quantization,
                str(args.temperature),
                str(args.max_retries),
            ]
            if args.model_filter:
                cmd.append(args.model_filter)

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Error: {result.stderr or result.stdout or 'unknown error'}")
            else:
                job_id = result.stdout.strip()
                jobs.append(job_id)
                print(f"  -> Job {job_id}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nSubmitted {len(jobs)} jobs: {' '.join(jobs)}")


if __name__ == "__main__":
    main()
