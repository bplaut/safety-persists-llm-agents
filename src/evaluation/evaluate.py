#!/usr/bin/env python3
"""
Evaluate the agent based on a set of trajectories.

Usage: python src/evaluation/evaluate.py -inp <input_trajectories_file> -ev <evaluator_type>
"""

import argparse
import random
import re
import json
import os
import sys
from pathlib import Path

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from toolemu.dataloader import DataLoader
from toolemu.evaluators import EVALUATORS
from toolemu.executors import FuncExecutorWithRetry
from toolemu.utils import (
    llm_register_args,
    load_openai_llm_with_args,
    print_intermediate_result_and_stop,
)
from toolemu.utils.my_typing import *
from utils.model_name_utils import sanitize_model_name
from utils.train_utils import DEFAULT_RANDOM_SEED
from utils.toolemu_utils import compute_eval_consistency, ToolEmuFilePaths

load_dotenv()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval-type",
    "-ev",
    type=str,
    default="agent_safe",
    choices=EVALUATORS.keys(),
)
parser.add_argument("--output-file-prefix", "-out", type=str, default=None)
parser.add_argument("--eval-results-out-suffix", "-es", type=str, default="")
parser.add_argument("--random-seed", "-seed", type=int, default=DEFAULT_RANDOM_SEED)
DataLoader.register_args(parser)  # Registers --data-path, --case-indices, --num-replicates, --shuffle
parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Batch size for evaluation")
parser.add_argument("--stop-at", type=str, default=None, help="Where to stop in the evaluation pipeline")
parser.add_argument("--threading-request-timeout", type=int, default=3000, help="Timeout for a single function execution request")
parser.add_argument("--sequential-generation", action="store_true", help="Run function sequentially over the dataset instead of in parallel")
parser.add_argument("--disable-batch-progress", action="store_true", help="Disable progress bar for multi-threaded execution")
parser.add_argument("--disable-timer", action="store_true", help="Disable timer for multi-threaded execution")
parser.add_argument("--num-retries", type=int, default=0, help="The number of retries when the function execution fails")
parser.add_argument("--help-safety-level", type=str, choices=["normal", "ignore_safety"], default="normal", help="Which helpfulness evaluation variant to use")
parser.add_argument(
    "--quantization",
    type=str,
    default="none",
    help="Quantization type for evaluator model (none, int4, int8, fp16). API models ignore this.",
)

parser.add_argument("--critique-rounds", type=int, default=0, help="Number of critique rounds for self-refinement (default: 0, disabled)")

llm_register_args(parser, prefix="evaluator")
args = parser.parse_args()

# Apply quantization to evaluator
if args.quantization:
    args.evaluator_quantization = args.quantization
random.seed(args.random_seed)


def clean_and_parse_score(raw_score):
    """Clean and robustly parse a numeric score from model output."""
    if isinstance(raw_score, (int, float)):
        return raw_score
    if not isinstance(raw_score, str):
        raise ValueError(f"Cannot parse score from non-string: {raw_score}")
    
    # First, try to extract just the first number we find
    number_match = re.search(r'(\d+(?:\.\d+)?)', raw_score)
    if number_match:
        try:
            score_str = number_match.group(1)
            if "." in score_str:
                return float(score_str)
            else:
                return int(score_str)
        except ValueError:
            pass
    
    # Fallback: Remove markdown, whitespace, and non-numeric chars except . and -
    cleaned = re.sub(r"[^0-9.\-]", "", raw_score)
    if cleaned == "":
        raise ValueError(f"Score string is empty after cleaning: {raw_score}")
    try:
        if "." in cleaned:
            return float(cleaned)
        else:
            return int(cleaned)
    except Exception as e:
        raise ValueError(f"Failed to parse cleaned score '{cleaned}' from '{raw_score}': {e}")


def main():
    # Override num_replicates=1 for dataloader since evaluate.py handles replication internally
    # via the loop in evaluate_trajec(). This prevents double-counting.
    trajs = DataLoader.from_args(args, return_mode="with_idx", item_name="trajectory", num_replicates=1)
    # Extract model name and quit type from data_path or args
    base = os.path.basename(args.data_path)
    input_dir = os.path.dirname(args.data_path)
    m = re.match(r"(.+?)_(.+?)_\d{4}_\d{6}", base)
    if m:
        modelname, quit_type = m.group(1), m.group(2)
    else:
        modelname, quit_type = "model", "quit"
    # Extract the full prefix from the input path (without extension)
    input_prefix = os.path.splitext(base)[0]
    # Add suffix based on help safety level
    eval_suffix = ""
    if args.eval_type == "agent_help":
        eval_suffix = ToolEmuFilePaths.HELP_SAFETY_SUFFIX_MAP[args.help_safety_level]
    # Add temperature to filename if nonzero (for non-deterministic evaluations)
    temp_suffix = ""
    evaluator_temp = getattr(args, 'evaluator_temperature', 0.0)
    if evaluator_temp > 0:
        temp_suffix = f"_t{evaluator_temp}"
    output_filename = f"{input_prefix}_eval_{args.eval_type}{eval_suffix}{temp_suffix}.jsonl"
    if input_dir:
        output_path = os.path.join(input_dir, output_filename)
    else:
        output_path = output_filename

    evaluator_llm = load_openai_llm_with_args(args, prefix="evaluator")
    print(f"Evaluator model name: {getattr(evaluator_llm, 'model_name', None)}")
    evaluator = EVALUATORS[args.eval_type].from_args(args, evaluator_llm)

    if evaluator._stop_at in ["preprocess", "prompt"]:
        result = evaluator({"trajectory": trajs[0]["item"]})
        print_intermediate_result_and_stop(result, evaluator._stop_at)

    # Determine metric key for consistency stats based on eval type
    metric_key = "ToolCallRisk" if args.eval_type == "agent_safe" else "Helpfulness"

    failed_indices = []
    def evaluate_trajec(traj_with_idx):
        traj_idx, traj = traj_with_idx["idx"], traj_with_idx["item"]

        if args.num_replicates == 1:
            # Single replicate: existing flat output format (backward compatible)
            try:
                results = evaluator({"trajectory": traj})
                result = results[0]
                if "eval_scores" in result:
                    for k, v in result["eval_scores"].items():
                        result["eval_scores"][k] = clean_and_parse_score(v)
            except Exception as e:
                result = {"error": str(e)}
                failed_indices.append(traj_idx)
                print(f"[ERROR] Failed at idx {traj_idx}: {e}")
                return traj_with_idx, result
            result["eval_id"] = traj_idx
            return None, result

        # Multi-replicate mode: run evaluations sequentially, store as array
        replicates = []
        for rep in range(args.num_replicates):
            try:
                results = evaluator({"trajectory": traj})
                result = results[0]
                if "eval_scores" in result:
                    for k, v in result["eval_scores"].items():
                        result["eval_scores"][k] = clean_and_parse_score(v)
                replicates.append(result)
            except Exception as e:
                replicates.append({"error": str(e)})
                print(f"[ERROR] Failed at idx {traj_idx} rep {rep}: {e}")

        # Check if all replicates failed
        errors = [r for r in replicates if "error" in r]
        if len(errors) == args.num_replicates:
            failed_indices.append(traj_idx)
            return traj_with_idx, {
                "eval_id": traj_idx,
                "replicates": replicates,
                "error": "All replicates failed"
            }

        # Compute consistency stats from successful replicates
        successful_scores = [
            r["eval_scores"][metric_key]
            for r in replicates
            if "eval_scores" in r and metric_key in r["eval_scores"]
        ]

        consistency = compute_eval_consistency(successful_scores) if successful_scores else None

        return None, {
            "eval_id": traj_idx,
            "replicates": replicates,
            "consistency": consistency
        }

    runner = FuncExecutorWithRetry.from_args(args)
    runner.run(evaluate_trajec, output_path, trajs)
    if failed_indices:
        print(f"[SUMMARY] Failed indices: {failed_indices}")
    print(
        "You may want to use scripts to convert the result jsonl file "
        f"{output_path} to json for easier reading."
    )


if __name__ == "__main__":
    main()
