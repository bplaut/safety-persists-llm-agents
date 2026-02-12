#!/usr/bin/env python3
"""Merge LoRA adapters into source model weights for sequential finetuning.

This script merges LoRA adapters into the source model, producing a full BF16 model
that can be used as a starting point for additional finetuning rounds.

Usage:
    python src/training/merge_lora.py \
        --adapter-path dpo_output/Qwen2.5-7B_int4_most_gpt5m_beta-0.02/final \
        [--output-dir dpo_trained/Qwen2.5-7B_int4_most_gpt5m_beta-0.02]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src directory to path for imports (parent.parent = src/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_name_utils import extract_adapter_dir_name
from utils.train_utils import load_adapter_config, get_source_model_from_config, setup_logging

logger = logging.getLogger(__name__)


def infer_output_dir(adapter_path: str) -> str:
    """Infer output directory from adapter path.

    dpo_output/Qwen2.5-7B_int4_most/final -> dpo_trained/Qwen2.5-7B_int4_most
    """
    dir_name = extract_adapter_dir_name(adapter_path)
    return f"dpo_trained/{dir_name}"


def save_merge_metadata(
    output_dir: str,
    source_model: str,
    adapter_path: str,
    adapter_config: dict
) -> None:
    """Save merge metadata to merge_config.json."""
    metadata = {
        "source_model": source_model,
        "adapter_path": adapter_path,
        "merge_timestamp": datetime.now().isoformat(),
        "adapter_config": adapter_config
    }

    config_path = Path(output_dir) / "merge_config.json"
    with open(config_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved merge metadata to: {config_path}")


def validate_merged_model(output_dir: str) -> None:
    """Validate merged model by checking config exists and is valid."""
    logger.info(f"Validating merged model at: {output_dir}")

    config_path = Path(output_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config.json not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"Model architecture: {config.get('model_type', 'unknown')}")
    logger.info("Validation passed: model config is valid")


def merge_lora_to_source(
    adapter_path: str,
    output_dir: str,
    device_map: str = "auto"
) -> None:
    """Merge LoRA adapter into source model and save the merged model."""
    logger.info("=" * 60)
    logger.info("LoRA Merge Pipeline")
    logger.info("=" * 60)

    # Normalize and validate adapter path
    adapter_path = adapter_path.rstrip('/')
    if not adapter_path.endswith('/final'):
        raise ValueError(
            f"Adapter path must end with '/final', got: {adapter_path}"
        )

    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    # Check that we won't overwrite existing model files
    # (directory can exist for slurm_output, but model files should not)
    output_path = Path(output_dir)
    files_to_check = [
        "config.json",           # model.save_pretrained()
        "merge_config.json",     # save_merge_metadata()
        "tokenizer_config.json", # tokenizer.save_pretrained()
    ]
    # Also check for model weights (could be single file or sharded)
    if output_path.exists():
        model_files = list(output_path.glob("model*.safetensors")) + \
                      list(output_path.glob("pytorch_model*.bin"))
        if model_files:
            raise FileExistsError(
                f"Model weights already exist at: {output_dir} (found {model_files[0].name}). "
                "Delete the existing model first or choose a different output directory."
            )

    for filename in files_to_check:
        filepath = output_path / filename
        if filepath.exists():
            raise FileExistsError(
                f"Model file already exists: {filepath}. "
                "Delete the existing model first or choose a different output directory."
            )

    # Load adapter config to get source model
    logger.info(f"Loading adapter config from: {adapter_path}")
    adapter_config = load_adapter_config(adapter_path)
    source_model_name = get_source_model_from_config(adapter_config)
    logger.info(f"Source model: {source_model_name}")

    # Load tokenizer (from adapter path, which should have the tokenizer)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    logger.info(f"Tokenizer loaded: {type(tokenizer).__name__}")

    # Load source model in BF16 (no quantization for clean merge)
    logger.info(f"Loading source model in BF16: {source_model_name}")
    logger.info("This may take a few minutes...")
    model = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    logger.info(f"Source model loaded: {type(model).__name__}")

    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    logger.info("LoRA adapter loaded successfully")

    # Merge adapter into base weights
    logger.info("Merging LoRA adapter into source model weights...")
    model = model.merge_and_unload()
    logger.info("Merge complete")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save merged model
    logger.info(f"Saving merged model to: {output_dir}")
    logger.info("This may take a few minutes...")
    model.save_pretrained(output_dir)
    logger.info("Model saved")

    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    logger.info("Tokenizer saved")

    # Save merge metadata
    save_merge_metadata(output_dir, source_model_name, adapter_path, adapter_config)

    # Validate the saved model
    logger.info("Validating saved model...")
    validate_merged_model(output_dir)

    # Log summary
    logger.info("=" * 60)
    logger.info("Merge Complete!")
    logger.info("=" * 60)
    logger.info(f"Source model:  {source_model_name}")
    logger.info(f"Adapter:       {adapter_path}")
    logger.info(f"Output:        {output_dir}")

    # Log size info
    total_size = sum(
        f.stat().st_size for f in Path(output_dir).rglob('*') if f.is_file()
    )
    logger.info(f"Total size:    {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into source model for sequential finetuning"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (must end with /final)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for merged model (default: inferred from adapter path)"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)"
    )

    args = parser.parse_args()

    # Infer output directory if not provided
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = infer_output_dir(args.adapter_path)

    # Set up logging to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global logger
    logger = setup_logging(Path(output_dir), timestamp)
    logger.info(f"Output directory: {output_dir}")

    # Run merge
    merge_lora_to_source(
        adapter_path=args.adapter_path,
        output_dir=output_dir,
        device_map=args.device_map
    )


if __name__ == "__main__":
    main()
