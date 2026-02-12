#!/usr/bin/env python3
"""Submit SLURM jobs for re-evaluating complete trajectory files without existing evals."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slurm_utils import get_evaluator_model, get_sbatch_args
from utils.toolemu_utils import (
    ToolEmuFilePaths,
    find_trajectory_files,
    load_jsonl,
    validate_trajectory_completeness,
)
from utils.train_utils import extract_seed_from_path

# Eval types to run (mapped to ToolEmuFilePaths.EVAL_TYPES format)
EVAL_TYPE_MAP = {
    "agent_safe": "agent_safe",
    "ignore_safety": "agent_help_ignore_safety",
}


def get_eval_path(traj_file: Path, eval_type: str, temperature: Optional[float]) -> Path:
    """Get eval file path using ToolEmuFilePaths."""
    path = ToolEmuFilePaths.trajectory_to_eval(str(traj_file), EVAL_TYPE_MAP[eval_type])
    if temperature and temperature != 0.0:
        path = path.replace(".jsonl", f"_t{temperature}.jsonl")
    return Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", default="output/trajectories")
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--num-replicates", "-n", type=int, default=1)
    parser.add_argument("--temperature", "-t", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print all jobs to submit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Error: Directory not found: {data_dir}")

    print(f"Scanning {data_dir}...")
    all_traj_files = find_trajectory_files(data_dir)

    # Filter to complete files and check for existing evals
    pairs_to_eval, incomplete = [], []
    for traj_file in all_traj_files:
        trajectories = load_jsonl(str(traj_file))
        test_seed = extract_seed_from_path(str(traj_file))
        is_valid, mismatch_info = validate_trajectory_completeness(
            trajectories, str(traj_file), test_seed
        )
        if not is_valid:
            expected, actual = mismatch_info
            incomplete.append((traj_file, expected, actual))
            continue
        for eval_type in EVAL_TYPE_MAP:
            eval_file = get_eval_path(traj_file, eval_type, args.temperature)
            if not eval_file.exists() or args.force:
                pairs_to_eval.append((traj_file, eval_type))

    # Summary
    print(f"Found {len(all_traj_files)} trajectories, {len(all_traj_files) - len(incomplete)} complete")
    if incomplete:
        print(f"Incomplete ({len(incomplete)}):")
        for f, exp, act in incomplete[:5]:
            print(f"  {f.name}: {act}/{exp}")
        if len(incomplete) > 5:
            print(f"  ... and {len(incomplete) - 5} more")
    print(f"To evaluate: {len(pairs_to_eval)}")

    if args.verbose and pairs_to_eval:
        print("\nJobs to submit:")
        for traj_file, eval_type in pairs_to_eval:
            print(f"  {traj_file.name} ({eval_type})")

    if not pairs_to_eval:
        sys.exit(0)

    if input("Continue? (y/n): ").strip().lower() != "y":
        sys.exit("Aborted.")

    # Submit jobs
    jobs = []
    for i, (traj_file, eval_type) in enumerate(pairs_to_eval, 1):
        evaluator = get_evaluator_model(traj_file)
        print(f"[{i}/{len(pairs_to_eval)}] {traj_file.name} ({eval_type})")
        try:
            run_eval_script = str(Path(__file__).parent / "run_eval.sh")
            cmd = ["sbatch"] + get_sbatch_args(evaluator, args.quantization) + [
                run_eval_script, str(traj_file), evaluator, args.quantization,
                eval_type, str(args.num_replicates)
            ]
            if args.temperature:
                cmd.append(str(args.temperature))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Error: {result.stderr or result.stdout or 'unknown error'}")
                print(f"  Command: {' '.join(cmd)}")
            else:
                job_id = result.stdout.strip()
                jobs.append(job_id)
                print(f"  -> {job_id}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nSubmitted {len(jobs)} jobs: {' '.join(jobs)}")


if __name__ == "__main__":
    main()
