#!/usr/bin/env python3
import argparse
import datetime
import os
import os.path as osp
import json
import glob
import subprocess
import sys
from pathlib import Path

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from toolemu.agents import AGENT_TYPES, SIMULATORS
from toolemu.dataloader import DataLoader
from toolemu.evaluators import EVALUATORS
from toolemu.utils.llm import llm_register_args
from toolemu.utils.io import get_output_dir_and_prefix, get_run_id
from utils.toolemu_utils import build_toolemu_filename, check_trajectory_exists, ToolEmuFilePaths
from utils.model_name_utils import sanitize_model_name
from utils.train_utils import compute_test_indices, extract_seed_from_path

EXECUTE_SCRIPT = "src/evaluation/emulate.py"
EVALUATE_SCRIPT = "src/evaluation/evaluate.py"


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser()
    llm_register_args(parser, prefix="agent")
    DataLoader.register_args(parser, default_data_path="./assets/all_cases.json")
    parser.add_argument("--output-dir", "-o", type=str, default="./output/trajectories")
    parser.add_argument(
        "--agent-type",
        "-atp",
        type=str,
        default="naive",
        choices=AGENT_TYPES,
        help="Type of agent to use (naive, ss_only, or helpful_ss).",
    )
    parser.add_argument(
        "--emulator-type",
        "-stp",
        type=str,
        default="adv_thought",
        choices=list(SIMULATORS.keys()),  # Still uses SIMULATORS from toolemu
    )
    parser.add_argument("--batch-size", "-bs", type=int, default=1, help="batch size")
    parser.add_argument("--max-agent-tokens", type=int, default=8000,
        help="Maximum total tokens the agent can generate before stopping (default: 8000)")
    parser.add_argument("--auto", "-auto", action="store_true", help="auto mode")
    # Emulator always runs unquantized (hardcoded in emulate.py). A bit hacky but we're
    # only going to use Qwen3-8B as emulator so it's fine and easier than having a second
    # --emulator-quantization arg.
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        help="Quantization type for agent and evaluator (none, int4, int8, fp16). API models ignore this. Emulator always runs unquantized.",
    )
    parser.add_argument(
        "--help-safety-levels",
        nargs="+",
        choices=["normal", "ignore_safety"],
        default=["ignore_safety"],
        help="Which helpfulness evaluation variants to run (default: ignore_safety only)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation and only run trajectory generation",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run test cases. Extracts seed from model name (e.g., model_s42). Intersects with --case-indices range.",
    )
    # Note: --num-replicates is already added by DataLoader.register_args()
    # Use shortprefix="m" for emulator to avoid conflict with evaluator's "-e" short options
    llm_register_args(parser, prefix="emulator", shortprefix="m")
    llm_register_args(parser, prefix="evaluator")
    return parser


def run_command(cmd, args, need_confirm=True):
    """Run a command safely without shell injection risk. Prompts for confirmation if need_confirm=True."""
    print("Command:", ' '.join(cmd))
    if not args.auto and need_confirm and input("Continue? (y/n): ") != "y":
        raise RuntimeError("User aborted.")
    try:
        subprocess.run(cmd, shell=False, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}") from e


def add_evaluator_args_to_cmd(cmd, args):
    """Add common evaluator arguments (model, quantization, etc.) to a command list."""
    if hasattr(args, 'evaluator_model_name') and args.evaluator_model_name is not None:
        cmd.extend(["--evaluator-model-name", args.evaluator_model_name])
    if hasattr(args, 'evaluator_tensor_parallel_size') and args.evaluator_tensor_parallel_size is not None:
        cmd.extend(["--evaluator-tensor-parallel-size", str(args.evaluator_tensor_parallel_size)])
    if args.quantization is not None:
        cmd.extend(["--quantization", args.quantization])
    if hasattr(args, 'evaluator_max_tokens') and args.evaluator_max_tokens is not None:
        cmd.extend(["--evaluator-max-tokens", str(args.evaluator_max_tokens)])


def check_eval_output(eval_file):
    """Check if evaluation output file exists and print status."""
    if os.path.exists(eval_file):
        print(f"Eval results written to {eval_file}")
    else:
        print(f"[Warning] Eval output file not found: {eval_file}")


def main():
    """Main entry point for running ToolEmu evaluation pipeline."""
    parser = create_parser()
    args = parser.parse_args()

    # If --test-only is set, compute intersection of test indices with case range
    # Store original range for filename building; update args.case_indices to intersection
    original_case_indices = args.case_indices  # Preserve for filename
    if args.test_only:
        # Extract seed from agent model name
        test_seed = extract_seed_from_path(args.agent_model_name)
        if test_seed is None:
            raise ValueError(
                f"--test-only requires a seed in the model name (e.g., model_s42), "
                f"but none found in: {args.agent_model_name}"
            )

        test_indices = compute_test_indices(test_seed)
        if args.case_indices is not None:
            # Parse range and compute intersection
            parts = args.case_indices.split('-')
            if len(parts) != 2:
                raise ValueError(f"--case-indices must be a range (START-END) when using --test-only")
            start, end = int(parts[0]), int(parts[1])
            intersection = [i for i in test_indices if start <= i < end]
        else:
            # No range specified, use all test indices
            intersection = test_indices

        if not intersection:
            print(f"WARNING: No test cases in range. Test indices for seed {test_seed}: {test_indices[:10]}...")
            exit(0)

        # Update case_indices to comma-separated list for DataLoader and emulate.py
        args.case_indices = ",".join(map(str, intersection))
        print(f"Test-only mode (seed {test_seed}): running {len(intersection)} test cases")

    # Check if trajectory already exists for this configuration (skip early to avoid model loading)
    exists, existing_path = check_trajectory_exists(
        agent_model=args.agent_model_name,
        emulator_model=args.emulator_model_name,
        evaluator_model=args.evaluator_model_name,
        case_range=original_case_indices,
        output_dir=args.output_dir,
    )
    if exists:
        range_str = f"range {original_case_indices}" if original_case_indices else "full dataset"
        print(f"SKIPPING: Trajectory already exists for {range_str}")
        print(f"  -> Found: {existing_path}")
        print("To re-run, delete the existing trajectory file first.")
        exit(0)

    cases = DataLoader.from_args(args, item_name="case")
    print(f"You are going to run {len(cases)} cases.")
    if not args.auto and input("Continue? (y/n): ") != "y":
        exit()

    # Determine model subdirectory (sanitize model name for filesystem)
    model_subdir = sanitize_model_name(args.agent_model_name)
    output_dir = os.path.join(args.output_dir, model_subdir)
    os.makedirs(output_dir, exist_ok=True)

    NOW = get_run_id()

    # Parse case range for filename building (use original range, not intersection)
    case_range = None
    if original_case_indices is not None:
        parts = original_case_indices.split('-')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid case-indices format: '{original_case_indices}'. "
                f"Expected format: 'START-END' (e.g., '0-15')"
            )
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid case-indices: '{original_case_indices}'. "
                f"START and END must be integers."
            )
        if start < 0 or end < 0:
            raise ValueError(
                f"Invalid case-indices: '{original_case_indices}'. "
                f"START and END must be non-negative."
            )
        if start >= end:
            raise ValueError(
                f"Invalid case-indices: '{original_case_indices}'. "
                f"START ({start}) must be less than END ({end})."
            )
        case_range = (start, end)

    # Build filename using centralized utility
    run_prefix = build_toolemu_filename(
        agent_model=args.agent_model_name,
        emu_model=args.emulator_model_name,
        eval_model=args.evaluator_model_name,
        quantization=args.quantization or "",
        case_range=case_range,
        timestamp=NOW
    )
    output_prefix = os.path.join(output_dir, run_prefix)

    cmd = [
        "python", EXECUTE_SCRIPT,
        "--data-path", args.data_path,
        "-o", output_dir,
        "-outsuf", f"_{run_prefix}",
        "-atp", args.agent_type,
        "-stp", args.emulator_type,
        "-am", args.agent_model_name,
        "--emulator-model-name", args.emulator_model_name,
        "--run-id", NOW,
    ]
    if args.agent_max_tokens is not None:
        cmd.extend(["--agent-max-tokens", str(args.agent_max_tokens)])
    if args.case_indices is not None:
        cmd.extend(["--case-indices", args.case_indices])
    if args.batch_size is not None:
        cmd.extend(["-bs", str(args.batch_size)])
    if args.max_agent_tokens is not None:
        cmd.extend(["--max-agent-tokens", str(args.max_agent_tokens)])
    if args.quantization is not None:
        cmd.extend(["--quantization", args.quantization])
    if hasattr(args, 'emulator_max_seq_len') and args.emulator_max_seq_len is not None:
        cmd.extend(["--emulator-max-seq-len", str(args.emulator_max_seq_len)])
    run_command(cmd, args)
    output_path = output_prefix
    print("The trajectories are saved to", output_path)

    if args.skip_eval:
        print("Skipping evaluation (--skip-eval flag set)")
        exit(0)

    # evaluate all trajectories for different evaluators
    for ev in EVALUATORS.keys():
        if ev == 'agent_help':
            # Run all selected helpfulness variants
            for safety_level in args.help_safety_levels:
                suffix = ToolEmuFilePaths.HELP_SAFETY_SUFFIX_MAP[safety_level]
                eval_file = os.path.join(output_dir, f"{run_prefix}_eval_{ev}{suffix}.jsonl")
                cmd = [
                    "python", EVALUATE_SCRIPT,
                    "--data-path", f"{output_prefix}.jsonl",
                    "-bs", str(args.batch_size),
                    "-ev", ev,
                ]
                add_evaluator_args_to_cmd(cmd, args)
                # Pass safety level and replicates to evaluate.py
                cmd.extend(["--help-safety-level", safety_level])
                cmd.extend(["--num-replicates", str(args.num_replicates)])
                run_command(cmd, args)
                check_eval_output(eval_file)
        else:
            # Other evaluators run once normally
            eval_file = os.path.join(output_dir, f"{run_prefix}_eval_{ev}.jsonl")
            cmd = [
                "python", EVALUATE_SCRIPT,
                "--data-path", f"{output_prefix}.jsonl",
                "-bs", str(args.batch_size),
                "-ev", ev,
            ]
            add_evaluator_args_to_cmd(cmd, args)
            cmd.extend(["--num-replicates", str(args.num_replicates)])
            run_command(cmd, args)
            check_eval_output(eval_file)

if __name__ == "__main__":
    main()
