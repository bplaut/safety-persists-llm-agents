"""Shared SLURM node selection utilities for GPU job submission scripts."""

import re
import sys
from pathlib import Path
from typing import List

# Add src to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.model_name_utils import expand_model_nickname
from utils.toolemu_utils import parse_toolemu_filename

# GPU node lists by type
A100_NODES = "airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu"
A6000_NODES = "gail.ist.berkeley.edu,gan.ist.berkeley.edu,ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu"
ALL_GPU_NODES = f"{A100_NODES},{A6000_NODES}"

# Aliases for backwards compatibility with existing code
HIGH_MEM_NODES = A100_NODES
STANDARD_NODES = ALL_GPU_NODES

# Special model sizes (models without XB naming pattern)
SPECIAL_MODEL_SIZES = {
    "microsoft/phi-4": 14,
    "microsoft/Phi-4-mini-instruct": 4,
    "microsoft/Phi-4-reasoning-plus": 14,
    "microsoft/Phi-4-reasoning": 14,
}


def get_evaluator_model(filepath: Path) -> str:
    """Extract evaluator model from filename, unsanitizing to full model name."""
    eval_model = parse_toolemu_filename(filepath.name)["eval_model"]
    if eval_model.startswith(("gpt-", "claude-", "gemini-")):
        return eval_model
    return eval_model.replace("_", "/", 1)


def get_model_size(model: str, include_lora_overhead: bool = False) -> int:
    """Get model size in billions. Returns 0 for API models.

    Args:
        model: Model name. Can include '+adapter_path' for LoRA models.
        include_lora_overhead: If True, adds ~4B for LoRA adapter memory overhead.
    """
    # Handle LoRA adapter paths: "base_model+adapter_path" -> extract base_model
    if "+" in model:
        model = model.split("+")[0]

    # API models don't need GPU
    if re.match(r"^(gpt-|claude-|gemini-)", model, re.IGNORECASE):
        return 0

    size = 0

    # HuggingFace models contain "/"
    if "/" in model:
        # Handle merged/finetuned models
        if re.match(r"^(dpo_merged|dpo_trained|sft_merged|sft_trained)/", model):
            nickname = Path(model).name.split("_")[0]
            short_name = expand_model_nickname(nickname)
            match = re.search(r"(\d+)[bB]", short_name)
            if match:
                size = int(match.group(1))
            elif nickname == "Phi-4":
                size = 14
            else:
                raise ValueError(f"Cannot determine size for merged model: {model}")
        # Special cases
        elif model in SPECIAL_MODEL_SIZES:
            size = SPECIAL_MODEL_SIZES[model]
        else:
            # Extract XB pattern
            match = re.search(r"(\d+)[bB]", model)
            if match:
                size = int(match.group(1))
            else:
                raise ValueError(f"Cannot determine size for model: {model}")
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Add LoRA overhead if requested (~4GB for adapters)
    if include_lora_overhead:
        size += 4

    return size


def get_gpu_nodelist(size: int, quantization: str) -> str:
    """Get appropriate GPU nodelist based on model size and quantization."""
    if quantization in ("none", "fp16"):
        threshold = 24  # ~2GB/B, 24B threshold for 48GB GPUs
    elif quantization == "int8":
        threshold = 35  # ~1GB/B, 35B threshold for 48GB GPUs
    else:
        threshold = 65  # ~0.5GB/B (int4), 65B threshold for 48GB GPUs
    return HIGH_MEM_NODES if size > threshold else STANDARD_NODES


def get_training_nodelist(gpu_type: str) -> str:
    """Get nodelist for training jobs based on GPU type preference.

    Args:
        gpu_type: "A100" (80GB, high-mem only), "A6000" (48GB), or "any" (all nodes)
    """
    if gpu_type == "A100":
        return A100_NODES
    elif gpu_type == "A6000":
        # A6000 jobs can also run on A100, so include both
        return ALL_GPU_NODES
    elif gpu_type == "any":
        return ALL_GPU_NODES
    else:
        raise ValueError(f"Unknown GPU type: {gpu_type}. Use: A100, A6000, or any")


def get_sbatch_args(evaluator_model: str, quantization: str) -> List[str]:
    """Get SBATCH args based on evaluator model size."""
    size = get_model_size(evaluator_model)
    args = ["--parsable"]
    if size > 0:
        nodelist = get_gpu_nodelist(size, quantization)
        args.extend(["--gres=gpu:1", f"--nodelist={nodelist}"])
    return args
