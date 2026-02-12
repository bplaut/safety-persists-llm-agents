#!/usr/bin/env python3
"""
Emulate the agent's tool-use trajectories.

Usage: python src/evaluation/emulate.py -inp <input_cases_file> -atp <agent_type> -stp <simulator_type> -ctp <case_type>
"""

import argparse
import os
import random
import sys
import time
import requests
import json
from pathlib import Path

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from openai import BadRequestError
import openai
import torch
from dotenv import load_dotenv
from langchain_core.agents import AgentAction
from toolemu.agent_executor_builder import build_agent_executor
from toolemu.agents import AGENT_TYPES, SIMULATORS
from toolemu.agents.agent_executor import SimulatedObservation
from toolemu.dataloader import DataLoader
from toolemu.executors import FuncExecutorWithRetry
from toolemu.tools import *
from toolemu.utils import (
    case_to_input_dict,
    filter_keys,
    get_toolkit_names,
    llm_register_args,
    load_openai_llm_with_args,
    replace_agent_action_with_list,
)
from toolemu.utils.my_typing import *
from toolemu.utils.io import get_run_id
from utils.train_utils import DEFAULT_RANDOM_SEED

load_dotenv()

parser = argparse.ArgumentParser()
llm_register_args(parser, prefix="agent")
# Use shortprefix="m" for emulator to avoid potential conflicts with other "-e" prefixes
llm_register_args(parser, prefix="emulator", shortprefix="m")
DataLoader.register_args(parser)
FuncExecutorWithRetry.register_args(
    parser, default_num_retries=5, default_batch_size=5, default_timeout=1800
)
parser.add_argument("--output-dir", "-o", type=str, default="./output/trajectories")
parser.add_argument(
    "--output-file-suffix",
    "-outsuf",
    type=str,
    default="",
    help="Suffix to add to the output file name",
)
parser.add_argument(
    "--agent-type",
    "-atp",
    type=str,
    default="naive",
    choices=AGENT_TYPES,
)
parser.add_argument(
    "--emulator-type",
    "-stp",
    type=str,
    default="adv_thought",
    choices=list(SIMULATORS.keys()),  # Still uses SIMULATORS from toolemu
)
parser.add_argument("--max-iterations", "-mi", type=int, default=None)
parser.add_argument("--max-agent-tokens", type=int, default=8000,
    help="Maximum total tokens the agent can generate before stopping (default: 8000)")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--random-seed", "-seed", type=int, default=DEFAULT_RANDOM_SEED)
parser.add_argument("--run-id", type=str, default=None, help="Run ID for consistent output naming")
parser.add_argument(
    "--quantization",
    type=str,
    default="none",
    help="Quantization type for agent model (none, int4, int8, fp16). API models ignore this. Emulator always runs unquantized.",
)

args = parser.parse_args()

# Apply quantization to agent only; emulator always runs unquantized.
# This is a bit hacky but we're only going to use Qwen3-8B as emulator so it's fine
# and easier than having a second --emulator-quantization arg.
if args.quantization:
    args.agent_quantization = args.quantization
args.emulator_quantization = "none"
random.seed(DEFAULT_RANDOM_SEED)

def main():
    # Do not manually add tensor_parallel_size or quantization to llm_kwargs_agent
    llms = {
        'agent': load_openai_llm_with_args(args, prefix='agent'),
        'emulator': load_openai_llm_with_args(args, prefix='emulator')
    }

    cases = DataLoader.from_args(args, return_mode="with_idx", item_name="case")

    runner = FuncExecutorWithRetry.from_args(args)
    NOW = args.run_id if args.run_id is not None else get_run_id()

    # Print model configuration summary
    print("\n" + "="*80)
    print(f"Loaded models:")
    print(f"  Agent:      {llms['agent'].model_name if hasattr(llms['agent'], 'model_name') else getattr(llms['agent'], '_model_name', 'Unknown')}")
    if hasattr(llms['agent'], 'tensor_parallel_size'):
        print(f"             tensor_parallel_size={llms['agent'].tensor_parallel_size}")
    if hasattr(llms['agent'], 'quantization'):
        print(f"             quantization={llms['agent'].quantization}")

    print(f"  Emulator:   {llms['emulator'].model_name if hasattr(llms['emulator'], 'model_name') else getattr(llms['emulator'], '_model_name', 'Unknown')}")
    if hasattr(llms['emulator'], 'tensor_parallel_size'):
        print(f"             tensor_parallel_size={llms['emulator'].tensor_parallel_size}")
    if hasattr(llms['emulator'], 'quantization'):
        print(f"             quantization={llms['emulator'].quantization}")
    
    if hasattr(args, 'evaluator_model_name'):
        print(f"  Evaluator:  {args.evaluator_model_name}")
    print("="*80 + "\n")
    
    # output_file_suffix is required and should be provided by run.py
    # It contains the full filename prefix including model, type, and timestamp
    if not args.output_file_suffix:
        raise ValueError("output_file_suffix is required.")

    output_dir = args.output_dir
    # Remove leading underscore if present
    suffix = args.output_file_suffix.lstrip('_')
    output_prefix = os.path.join(output_dir, suffix)
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_prefix + ".jsonl"

    def generate_trajectory(case_with_idx):
        case_idx, case = case_with_idx["idx"], case_with_idx["item"]

        # Print case header for debugging
        print("\n" + "#"*80, flush=True)
        print(f"# STARTING CASE {case_idx}", flush=True)
        print(f"# User instruction: {case.get('User Instruction', case.get('input', 'N/A'))[:200]}", flush=True)
        print("#"*80 + "\n", flush=True)

        agent_executer = build_agent_executor(
            get_toolkit_names(case),
            llms["agent"],
            llms["emulator"],
            agent_type=args.agent_type,
            simulator_type=args.emulator_type,  # toolemu expects simulator_type
            verbose=args.verbose,
            max_iterations=args.max_iterations,
            max_agent_tokens=args.max_agent_tokens,
        )
        inputs = filter_keys(case_to_input_dict(case), agent_executer.input_keys)
        max_retries = 5
        num_tries = 0
        while num_tries < max_retries:
            if num_tries > 0:
                # Exponential backoff, capped at 32 seconds
                sleep_time = min(2 ** num_tries, 32)
                print(f"Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)
            try:
                outputs = agent_executer(inputs)
                failed_item = None
                break
            except (BadRequestError, getattr(anthropic, 'BadRequestError', Exception)) as e:
                print(f"{case_idx}: {str(e)}")
                outputs = {"error": str(e), "error_type": type(e).__name__}
                failed_item = case_with_idx
                break
            except Exception as e:
                # Handle OpenAI/Anthropic/HuggingFace/network errors and prompt length errors
                import sys
                import traceback
                error_str = str(e)
                tb_str = traceback.format_exc()
                # Handle prompt too long error (ValueError)
                if isinstance(e, ValueError) and ("longer than the maximum model length" in error_str or "is too long and exceeds limit" in error_str):
                    print(f"{case_idx}: Prompt too long for model, skipping. {error_str}")
                    return None, None  # Skip entirely, no write, no retry
                # Handle CUDA OOM error
                if isinstance(e, (RuntimeError, torch.cuda.OutOfMemoryError)) and "out of memory" in error_str.lower():
                    torch.cuda.empty_cache()
                    print(f"{case_idx}: CUDA OOM (likely context explosion). {error_str[:200]}")

                    # Recover partial state
                    intermediate_steps = getattr(agent_executer, '_last_intermediate_steps', None) or []
                    agent_output = getattr(agent_executer.agent, '_last_agent_output', None)

                    if intermediate_steps or agent_output:
                        # Log what we're saving
                        print(f"{case_idx}: Saving partial trajectory with {len(intermediate_steps)} steps, "
                              f"agent_output={'present (' + str(len(agent_output)) + ' chars)' if agent_output else 'absent'}")

                        # If we have agent output, add it as a fake final step (like token limit case)
                        if agent_output:
                            oom_action = AgentAction(
                                tool="[OOM]",
                                tool_input="",
                                log=agent_output
                            )
                            oom_observation = SimulatedObservation(
                                observation="[OOM]",
                                thought_summary="",
                                log="[OOM]"
                            )
                            intermediate_steps = list(intermediate_steps) + [(oom_action, oom_observation)]

                        outputs = {
                            "intermediate_steps": intermediate_steps,
                            "output": "Agent stopped: OOM",
                        }
                        outputs = replace_agent_action_with_list(outputs)
                        outputs["case"] = case
                        outputs["case_idx"] = case_idx
                        return None, outputs  # Save partial trajectory
                    else:
                        print(f"{case_idx}: No partial trajectory to save (OOM on first LLM call before any output), skipping entirely")
                        return None, None
                # Handle OpenAI/Anthropic/network errors
                if (
                    hasattr(openai, 'error') and isinstance(e, (getattr(openai.error, 'RateLimitError', Exception), getattr(openai.error, 'Timeout', Exception)))
                ) or isinstance(e, requests.exceptions.RequestException):
                    print(f"{case_idx}: Network or rate limit error: {error_str}")
                    outputs = {"error": error_str, "error_type": type(e).__name__}
                    failed_item = case_with_idx
                    num_tries += 1
                    if num_tries >= max_retries:
                        print(f"{case_idx}: Failed after {max_retries} retries due to network or rate limit error. Skipping.")
                        break
                    continue
                # Handle generic errors
                print(f"{case_idx}: Unexpected error: {type(e).__name__}: {error_str}\n{tb_str}")
                outputs = {"error": error_str, "error_type": type(e).__name__}
                failed_item = case_with_idx
                num_tries += 1
                if num_tries >= max_retries:
                    print(f"{case_idx}: Failed after {max_retries} retries due to unexpected error. Skipping.")
                    break
                continue
        outputs = replace_agent_action_with_list(outputs)  # ensure serializable
        outputs["case"] = case
        outputs["case_idx"] = case_idx
        return failed_item, outputs

    runner.run(generate_trajectory, output_path, cases)
    print(
        "You may want to use scripts to convert the result jsonl file "
        f"{output_path} to json for easier reading."
    )

if __name__ == "__main__":
    main()
