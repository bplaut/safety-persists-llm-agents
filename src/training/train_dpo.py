#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Training Script

Trains an LLM using preference pairs from agent trajectories.
Uses TRL's DPOTrainer with LoRA for parameter-efficient fine-tuning.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.toolemu_utils import load_jsonl
from utils.train_utils import (
    compute_test_indices,
    setup_logging,
    load_model_and_tokenizer,
    setup_lora,
    EpochCheckpointCallback,
    partition_by_case_indices,
    DEFAULT_RANDOM_SEED,
    SampledEvalMixin,
)

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
    TrainerCallback,
)
from trl import DPOTrainer, DPOConfig





class SampledEvalDPOTrainer(SampledEvalMixin, DPOTrainer):
    """DPOTrainer with sampled evaluation using SampledEvalMixin."""
    pass


def load_dpo_dataset(data_path: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Load DPO dataset from JSONL file.

    Expected format (from prepare_dpo_data.py):
    {
        "prompt": List[Dict[str, str]],      # Message list (system + user messages)
        "chosen": List[Dict[str, str]],      # Message list (assistant response)
        "rejected": List[Dict[str, str]],    # Message list (assistant response)
        "case_idx": int,
        "chosen_model": str,
        "rejected_model": str,
        "chosen_score": int,
        "rejected_score": int,
        "score_difference": int
    }

    Each message dict has 'role' and 'content' keys.
    DPOTrainer will automatically apply the chat template to these message lists.
    """
    logger.info(f"Loading DPO dataset from {data_path}")

    examples = load_jsonl(str(data_path), description="DPO dataset")

    # Validate each example
    for line_num, example in enumerate(examples, 1):
        # Validate required fields
        required_fields = ['prompt', 'chosen', 'rejected', 'case_idx']
        missing_fields = [f for f in required_fields if f not in example]
        if missing_fields:
            raise ValueError(
                f"Missing required fields at line {line_num}: {missing_fields}"
            )

        # Validate that prompt, chosen, rejected are message lists
        for field in ['prompt', 'chosen', 'rejected']:
            if not isinstance(example[field], list):
                raise ValueError(
                    f"Field '{field}' at line {line_num} must be a list of message dicts, "
                    f"got {type(example[field]).__name__}"
                )
            if len(example[field]) == 0:
                raise ValueError(
                    f"Field '{field}' at line {line_num} is an empty list"
                )
            for msg in example[field]:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    raise ValueError(
                        f"Field '{field}' at line {line_num} contains invalid message structure. "
                        f"Each message must be a dict with 'role' and 'content' keys"
                    )

    logger.info(f"Loaded {len(examples)} DPO examples")

    if len(examples) == 0:
        raise ValueError("No examples loaded from dataset")

    return examples

def examples_to_dataset(examples: List[Dict[str, Any]]) -> Dataset:
    """Convert examples to HuggingFace Dataset format."""
    # DPOTrainer expects these exact column names
    data = {
        'prompt': [ex['prompt'] for ex in examples],
        'chosen': [ex['chosen'] for ex in examples],
        'rejected': [ex['rejected'] for ex in examples],
    }
    return Dataset.from_dict(data)


def create_datasets(
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    logger: logging.Logger
) -> Tuple[Dataset, Optional[Dataset]]:
    """Convert train/val examples to HuggingFace Dataset format."""
    train_dataset = examples_to_dataset(train_examples)
    logger.info(f"Created train dataset with {len(train_dataset)} examples")

    val_dataset = None
    if len(val_examples) > 0:
        val_dataset = examples_to_dataset(val_examples)
        logger.info(f"Created val dataset with {len(val_dataset)} examples")

    return train_dataset, val_dataset


def compute_example_lengths(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
) -> Tuple[int, int, int]:
    """Compute token lengths for prompt, chosen, and rejected sequences."""
    prompt_len = len(tokenizer.apply_chat_template(
        example['prompt'],
        add_generation_prompt=False,
        tokenize=True
    ))
    chosen_len = len(tokenizer.apply_chat_template(
        example['prompt'] + example['chosen'],
        add_generation_prompt=False,
        tokenize=True
    ))
    rejected_len = len(tokenizer.apply_chat_template(
        example['prompt'] + example['rejected'],
        add_generation_prompt=False,
        tokenize=True
    ))
    return prompt_len, chosen_len, rejected_len


def filter_long_examples(
    examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
    max_length: int,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Filter out examples that exceed max_length instead of truncating them.

    Args:
        examples: List of DPO examples with 'prompt', 'chosen', 'rejected' fields
        tokenizer: Tokenizer with chat template
        max_prompt_length: Maximum prompt length
        max_length: Maximum total sequence length
        logger: Logger instance

    Returns:
        Filtered list of examples
    """
    logger.info(f"Filtering examples that exceed max_length={max_length}...")

    filtered_examples = []
    excluded_prompt = 0
    excluded_chosen = 0
    excluded_rejected = 0

    for example in examples:
        prompt_len, chosen_len, rejected_len = compute_example_lengths(example, tokenizer)

        if prompt_len > max_prompt_length:
            excluded_prompt += 1
            continue
        if chosen_len > max_length:
            excluded_chosen += 1
            continue
        if rejected_len > max_length:
            excluded_rejected += 1
            continue

        filtered_examples.append(example)

    total_excluded = len(examples) - len(filtered_examples)
    if total_excluded > 0:
        pct = 100 * total_excluded / len(examples)
        logger.warning(
            f"⚠ EXCLUDED {total_excluded}/{len(examples)} ({pct:.2f}%) examples exceeding length limits:"
        )
        if excluded_prompt > 0:
            logger.warning(f"  - {excluded_prompt} with prompts > {max_prompt_length}")
        if excluded_chosen > 0:
            logger.warning(f"  - {excluded_chosen} with chosen responses > {max_length}")
        if excluded_rejected > 0:
            logger.warning(f"  - {excluded_rejected} with rejected responses > {max_length}")
    else:
        logger.info(f"✓ All {len(examples)} examples within length limits")

    logger.info(f"Kept {len(filtered_examples)}/{len(examples)} examples")
    return filtered_examples


def check_sequence_truncation(
    examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
    max_length: int,
    logger: logging.Logger
) -> None:
    """
    Check a random sample of examples for sequence truncation and log warnings.

    Args:
        examples: List of DPO examples with 'prompt', 'chosen', 'rejected' fields
        tokenizer: Tokenizer with chat template
        max_prompt_length: Maximum prompt length
        max_length: Maximum total sequence length
        logger: Logger instance
    """
    # Sample 10% of examples randomly
    sample_size = max(1, int(0.1 * len(examples)))
    sampled_examples = random.sample(examples, sample_size)

    logger.info(f"Checking {sample_size}/{len(examples)} examples (10% sample) for sequence truncation...")

    truncated_prompts = 0
    truncated_chosen = 0
    truncated_rejected = 0

    for example in sampled_examples:
        prompt_len, chosen_len, rejected_len = compute_example_lengths(example, tokenizer)

        if prompt_len > max_prompt_length:
            truncated_prompts += 1
        if chosen_len > max_length:
            truncated_chosen += 1
        if rejected_len > max_length:
            truncated_rejected += 1

    # Log warnings if truncation detected
    if truncated_prompts > 0:
        pct = 100 * truncated_prompts / sample_size
        logger.warning(
            f"⚠ TRUNCATION WARNING: {truncated_prompts}/{sample_size} ({pct:.1f}%) of sampled prompts "
            f"exceed max_prompt_length={max_prompt_length} and will be truncated"
        )

    if truncated_chosen > 0:
        pct = 100 * truncated_chosen / sample_size
        logger.warning(
            f"⚠ TRUNCATION WARNING: {truncated_chosen}/{sample_size} ({pct:.1f}%) of sampled chosen responses "
            f"exceed max_length={max_length} and will be truncated"
        )

    if truncated_rejected > 0:
        pct = 100 * truncated_rejected / sample_size
        logger.warning(
            f"⚠ TRUNCATION WARNING: {truncated_rejected}/{sample_size} ({pct:.1f}%) of sampled rejected responses "
            f"exceed max_length={max_length} and will be truncated"
        )

    if truncated_prompts == 0 and truncated_chosen == 0 and truncated_rejected == 0:
        logger.info(f"✓ No truncation detected in {sample_size} sampled examples")


def save_training_config(
    output_dir: Path,
    args: argparse.Namespace,
    timestamp: str,
    train_case_indices: set,
    test_case_indices: set,
    wandb_run_id: str | None,
    logger: logging.Logger
) -> None:
    """Save training configuration to JSON."""
    config_path = output_dir / "training_config.json"

    config = {
        'timestamp': timestamp,
        'model': args.model,
        'data_path': str(args.data_path),
        'output_dir': str(args.output_dir),
        'quantization': args.quantization,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'max_steps': args.max_steps,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'dpo_beta': args.dpo_beta,
        'max_length': args.max_length,
        'max_prompt_length': args.max_prompt_length,
        'seed': args.seed,
        'train_test_split_seed': args.train_test_split_seed,
        'train_case_indices': sorted(list(train_case_indices)),
        'test_case_indices': sorted(list(test_case_indices)),
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'wandb_run_id': wandb_run_id,
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved training config to {config_path}")


def setup_training_environment(args, timestamp: str) -> Tuple[logging.Logger, Optional[str], Optional[Dict]]:
    """Setup logging and W&B, load saved config if resuming.

    Returns: (logger, wandb_run_id, saved_config)
    """
    logger = setup_logging(args.output_dir, timestamp)

    logger.info("=" * 80)
    logger.info("DPO TRAINING")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Batch size: {args.batch_size} (accumulation: {args.gradient_accumulation_steps})")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"DPO beta: {args.dpo_beta}")
    logger.info(f"Train/test split seed: {args.train_test_split_seed}")
    logger.info(f"Reproducibility seed: {args.seed}")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Set reproducibility seed to {args.seed}")

    # Load saved config if resuming
    saved_config = None
    if args.resume_from_checkpoint:
        config_path = args.output_dir / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            logger.info("Loaded saved training configuration for resume")

    # Initialize wandb
    wandb_run_id = None
    if not args.no_wandb:
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            logger.info("Logged in to wandb")
        else:
            logger.warning("WANDB_API_KEY not found in environment - attempting anonymous login")

        run_name = args.output_dir.name

        if saved_config:
            wandb_run_id = saved_config.get('wandb_run_id')

        if wandb_run_id:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=wandb_run_id,
                resume="allow",
                config=vars(args),
            )
            logger.info(f"Resumed wandb run: entity={args.wandb_entity}, project={args.wandb_project}, id={wandb_run_id}")
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
            )
            wandb_run_id = wandb.run.id
            logger.info(f"Initialized new wandb run: entity={args.wandb_entity}, project={args.wandb_project}, run={run_name}, id={wandb_run_id}")
    else:
        logger.info("Wandb logging disabled")

    return logger, wandb_run_id, saved_config


def prepare_datasets_and_splits(
    args, logger, saved_config: Optional[Dict]
) -> Tuple[Dataset, Optional[Dataset], set, set, AutoModelForCausalLM, AutoTokenizer]:
    """Load data, compute train/test split, filter, and create HF datasets.

    Returns: (train_dataset, val_dataset, train_case_indices, test_case_indices, model, tokenizer)
    """
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATASET")
    logger.info("=" * 80)

    examples = load_dpo_dataset(args.data_path, logger)

    # Compute train/test split
    all_case_indices = sorted(set(ex['case_idx'] for ex in examples))
    logger.info(f"Dataset contains {len(all_case_indices)} unique task indices")

    test_case_indices = set(compute_test_indices(args.train_test_split_seed))
    train_case_indices = set(all_case_indices) - test_case_indices

    logger.info(f"Train/test split (seed={args.train_test_split_seed}):")
    logger.info(f"  Train cases: {len(train_case_indices)} (from {len(all_case_indices)} in dataset)")
    logger.info(f"  Train indices: {sorted(train_case_indices)}")
    logger.info(f"  Test set will also be used for validation during training")

    # Verify split matches saved config if resuming
    if args.resume_from_checkpoint:
        if not saved_config:
            config_path = args.output_dir / "training_config.json"
            raise FileNotFoundError(
                f"Cannot resume: training_config.json not found at {config_path}. "
                f"This file is needed to verify train/test split consistency."
            )

        saved_train = set(saved_config.get('train_case_indices', []))
        saved_test = set(saved_config.get('test_case_indices', []))

        if saved_train != train_case_indices or saved_test != test_case_indices:
            raise ValueError(
                f"Train/test split mismatch when resuming from checkpoint!\n"
                f"Saved config has train_test_split_seed={saved_config.get('train_test_split_seed')}\n"
                f"Current args have train_test_split_seed={args.train_test_split_seed}\n"
                f"This would cause data leakage. Either:\n"
                f"  1. Use the same --train-test-split-seed as the original run ({saved_config.get('train_test_split_seed')}), OR\n"
                f"  2. Start a new training run (don't use --resume-from-checkpoint)"
            )

        logger.info("✓ Train/test split matches saved configuration")

    # Load model and tokenizer (needed for filtering)
    logger.info("\n" + "=" * 80)
    logger.info("LOADING MODEL")
    logger.info("=" * 80)

    model, tokenizer = load_model_and_tokenizer(args.model, args.quantization, logger)
    model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, logger)

    # Filter examples
    logger.info("\n" + "=" * 80)
    logger.info("FILTERING LONG EXAMPLES")
    logger.info("=" * 80)
    examples = filter_long_examples(
        examples,
        tokenizer,
        args.max_prompt_length,
        args.max_length,
        logger
    )

    # Split and create datasets
    all_case_indices = set(ex['case_idx'] for ex in examples)
    train_case_indices_for_split = all_case_indices - test_case_indices
    train_examples, val_examples, _ = partition_by_case_indices(
        examples, train_case_indices_for_split, test_case_indices,
        logger=logger, require_train=True
    )
    train_dataset, val_dataset = create_datasets(train_examples, val_examples, logger)

    return train_dataset, val_dataset, train_case_indices, test_case_indices, model, tokenizer


def create_dpo_trainer(
    model, tokenizer, train_dataset: Dataset, val_dataset: Optional[Dataset], args, logger
) -> Tuple[DPOTrainer, DPOConfig]:
    """Configure training args and initialize DPO trainer.

    Returns: (dpo_trainer, training_args)
    """
    logger.info("\n" + "=" * 80)
    logger.info("CONFIGURING TRAINING")
    logger.info("=" * 80)

    if val_dataset is None:
        logger.warning("No validation dataset - evaluation during training will be disabled")

    training_args = DPOConfig(
        output_dir=str(args.output_dir / "recent_checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="wandb" if not args.no_wandb else "none",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        seed=args.seed,
        remove_unused_columns=False,
        beta=args.dpo_beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        precompute_ref_log_probs=True,
    )

    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Training steps per epoch: ~{len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps)}")
    logger.info(f"Total training steps: {training_args.max_steps if args.max_steps > 0 else 'auto'}")

    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING DPO TRAINER")
    logger.info("=" * 80)

    # Validate first training example
    logger.info("Validating first training example...")
    first_ex = train_dataset[0]
    for key in ['prompt', 'chosen', 'rejected']:
        if key in first_ex:
            value = first_ex[key]
            if isinstance(value, list):
                logger.info(f"  {key}: list of {len(value)} messages")
            elif isinstance(value, str):
                logger.info(f"  {key}: string ({len(value)} chars)")
        else:
            logger.warning(f"  Missing key: {key}")

    # Compute eval sample size if needed
    if val_dataset and args.eval_sample_size is None:
        args.eval_sample_size = args.logging_steps * args.gradient_accumulation_steps * args.batch_size

        if args.eval_sample_size <= 0:
            logger.warning(f"Computed eval_sample_size is {args.eval_sample_size} (invalid). Using full eval dataset.")
            args.eval_sample_size = None
        else:
            logger.info(f"Computed eval_sample_size: {args.eval_sample_size} "
                       f"({args.logging_steps} steps × {args.gradient_accumulation_steps} grad_acc × {args.batch_size} batch_size)")
            logger.info(f"NOTE: Eval sample size is based on logging_steps ({args.logging_steps}), not eval_steps ({args.eval_steps})")
    elif args.eval_sample_size is None:
        logger.info("No validation dataset provided - evaluation will be skipped")

    dpo_trainer = SampledEvalDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        eval_sample_size=args.eval_sample_size if val_dataset else None,
    )

    logger.info("SampledEvalDPOTrainer initialized successfully")
    if val_dataset and args.eval_sample_size:
        logger.info(f"Evaluation will sample {args.eval_sample_size} examples from {len(val_dataset)} total")

    # Register epoch checkpoint callback
    epoch_callback = EpochCheckpointCallback(args.output_dir, logger)
    dpo_trainer.add_callback(epoch_callback)
    logger.info("Registered EpochCheckpointCallback for end-of-epoch checkpoints")

    # Diagnostic checks
    if hasattr(dpo_trainer, 'ref_model') and dpo_trainer.ref_model is not None:
        logger.info(f"Reference model type: {type(dpo_trainer.ref_model).__name__}")

        if hasattr(dpo_trainer.ref_model, 'device'):
            logger.info(f"Reference model device: {dpo_trainer.ref_model.device}")

        if hasattr(dpo_trainer.ref_model, 'config') and hasattr(dpo_trainer.ref_model.config, '_attn_implementation'):
            ref_attn = dpo_trainer.ref_model.config._attn_implementation
            logger.info(f"Reference model attention: {ref_attn}")
            if ref_attn == "flash_attention_2":
                logger.info("✓ Reference model IS using Flash Attention 2")
            else:
                logger.warning(f"✗ Reference model NOT using Flash Attention 2 (using: {ref_attn})")

        if hasattr(model, 'base_model') and hasattr(dpo_trainer.ref_model, 'base_model'):
            same_base = id(model.base_model.model) == id(dpo_trainer.ref_model.base_model.model)
            if same_base:
                logger.info("✓ Reference model shares base weights with policy model (LoRA mode)")
            else:
                logger.warning("✗ Reference model has separate base weights (higher memory usage)")

        ref_model_class = type(dpo_trainer.ref_model).__name__
        logger.info(f"Reference model class: {ref_model_class}")
    else:
        logger.info("Reference model is None (will be created internally or weights shared)")

    # Validate tokenized data
    logger.info("Checking tokenized data...")
    try:
        train_dataloader = dpo_trainer.get_train_dataloader()
        first_batch = next(iter(train_dataloader))

        for key, value in first_batch.items():
            if hasattr(value, 'shape'):
                logger.info(f"Batch[{key}] shape: {value.shape}, dtype: {value.dtype}")
                if 'input_ids' in key or 'labels' in key:
                    if hasattr(value, 'min') and hasattr(value, 'max'):
                        min_val = value.min().item()
                        max_val = value.max().item()
                        logger.info(f"  Token ID range: [{min_val}, {max_val}]")
                        if max_val >= tokenizer.vocab_size:
                            logger.error(f"  ERROR: Found token ID {max_val} >= vocab_size {tokenizer.vocab_size}")
                        if min_val < -100:
                            logger.error(f"  ERROR: Found invalid token ID {min_val} < -100")
    except Exception as e:
        logger.warning(f"Could not validate tokenized data: {e}")

    return dpo_trainer, training_args


def execute_training(dpo_trainer: DPOTrainer, args, logger) -> None:
    """Run training loop and save results."""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    try:
        train_result = dpo_trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")

        # Save final model
        final_dir = args.output_dir / "final"
        dpo_trainer.save_model(str(final_dir))
        logger.info(f"Saved final model to {final_dir}")

        # Save training metrics
        metrics_path = args.output_dir / "training_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(train_result.metrics, f, indent=2)
        logger.info(f"Saved training metrics to {metrics_path}")

        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    finally:
        if not args.no_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="DPO Training for LLM Agent Safety")

    # Model and data
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B',
                        help='Source model to finetune')
    parser.add_argument('--data-path', '-d', type=Path, required=True,
                        help='Path to DPO dataset JSONL file (e.g., data/dpo_data/most_gpt5m.jsonl)')
    parser.add_argument('--output-dir', '-o', type=Path, default=Path('./dpo_output'),
                        help='Directory to save checkpoints and logs')

    # Quantization
    parser.add_argument('--quantization', type=str, default='none',
                        choices=['none', 'int4', 'int8', 'fp16'],
                        help='Model quantization level (none = bfloat16, no quantization)')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Per-device training batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--max-steps', type=int, default=-1,
                        help='Maximum training steps (overrides num_epochs if set)')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio for learning rate schedule')
    parser.add_argument('--save-steps', type=int, default=100,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval-steps', type=int, default=999999,
                        help='Evaluate every N steps (default: 999999, effectively disabled)')
    parser.add_argument('--logging-steps', type=int, default=40,
                        help='Log metrics every N steps')
    parser.add_argument('--eval-sample-size', type=int, default=None,
                        help='Number of examples to sample for evaluation (default: logging_steps * gradient_accumulation_steps * batch_size)')

    # LoRA configuration
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha (scaling factor)')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                        help='LoRA dropout')

    # DPO-specific
    parser.add_argument('--dpo-beta', type=float, default=0.1,
                        help='DPO regularization parameter')
    parser.add_argument('--max-length', type=int, default=8192,
                        help='Maximum sequence length')
    parser.add_argument('--max-prompt-length', type=int, default=8192,
                        help='Maximum prompt length')

    # Experiment tracking
    parser.add_argument('--wandb-project', type=str, default='llm-finetune-dpo',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default='chaiberkeley',
                        help='Weights & Biases entity (username or team name)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')

    # Data split
    parser.add_argument('-s', '--train-test-split-seed', type=int, required=True,
                        help='Random seed for train/test split (required)')

    # Resume training
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help='Path to checkpoint directory to resume training from')

    # Other
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED,
                        help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})')

    args = parser.parse_args()

    # HACK: Use 3 epochs for "both" datasets (which filter on safety and helpfulness simultaneously) since they're smaller
    if "both" in args.data_path.name and args.num_epochs == 1:
        args.num_epochs = 3

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Setup training environment (logging, wandb)
    logger, wandb_run_id, saved_config = setup_training_environment(args, timestamp)

    # Step 2: Prepare datasets and compute train/test split
    train_dataset, val_dataset, train_case_indices, test_case_indices, model, tokenizer = prepare_datasets_and_splits(
        args, logger, saved_config
    )

    # Step 3: Save training configuration
    save_training_config(args.output_dir, args, timestamp, train_case_indices, test_case_indices, wandb_run_id, logger)

    # Step 4: Create DPO trainer
    dpo_trainer, training_args = create_dpo_trainer(model, tokenizer, train_dataset, val_dataset, args, logger)

    # Step 5: Execute training
    execute_training(dpo_trainer, args, logger)


if __name__ == "__main__":
    main()
