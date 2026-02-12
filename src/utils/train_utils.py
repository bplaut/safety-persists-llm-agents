"""Training and dataset utilities."""
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
)

from utils.toolemu_utils import load_jsonl, TOOLEMU_FULL_DATASET_SIZE


DEFAULT_RANDOM_SEED = 42  # Default seed for reproducible train/test splits


def get_finetune_metric(dataset_name: str) -> Optional[str]:
    """Extract the core finetune metric from a dataset name.

    Strips suffixes like _gpt5m, _q32 and returns 'help', 'safe', or 'both'.
    """
    if 'both' in dataset_name:
        return 'both'
    if 'safe' in dataset_name and 'help' in dataset_name:
        return 'both'
    if 'safe' in dataset_name:
        return 'safe'
    if 'help' in dataset_name:
        return 'help'
    return None


def extract_training_stages(model_name: str, data_dir: str = "data/dpo_data") -> List[Tuple[str, str]]:
    """Extract training stages with their beta values from model name.

    Discovers available dataset names by scanning data_dir for .jsonl files.

    Returns list of (metric, beta_value) tuples where metric is 'help', 'safe', or 'both'.
    E.g., "dpo_merged_Llama-8B_help_gpt5m_beta-0.05_safe_gpt5m_beta-0.5"
          -> [("help", "0.05"), ("safe", "0.5")]
    """
    # Get available dataset names from data_dir
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    # Get dataset names sorted by length (longest first) to match compound names first
    datasets = sorted(
        [f.stem for f in data_path.glob("*.jsonl")],
        key=len,
        reverse=True
    )

    if not datasets:
        return []

    stages = []

    # Find all dataset markers and their positions
    # Pattern: _{dataset}_ or _{dataset}_beta-
    stage_matches = []
    for dataset in datasets:
        # Escape special chars in dataset name (e.g., safe-most has a hyphen)
        pattern = f'_({re.escape(dataset)})(?:_|$)'
        for match in re.finditer(pattern, model_name):
            stage_matches.append((match.start(), match.group(1)))

    # Sort by position and remove duplicates (keep first occurrence of each position)
    stage_matches = sorted(set(stage_matches), key=lambda x: x[0])

    # Get beta values and their positions
    beta_matches = [(m.start(), m.group(1)) for m in re.finditer(r'_beta-(\d+\.?\d*)', model_name)]

    for i, (stage_pos, dataset_name) in enumerate(stage_matches):
        next_stage_pos = stage_matches[i + 1][0] if i + 1 < len(stage_matches) else len(model_name)

        # Find beta between this stage and next stage
        stage_beta = '0.1'  # default
        for beta_pos, beta_val in beta_matches:
            if stage_pos < beta_pos < next_stage_pos:
                stage_beta = beta_val
                break

        # Convert dataset name to core metric (strip suffixes like _gpt5m)
        metric = get_finetune_metric(dataset_name)
        if metric:
            stages.append((metric, stage_beta))

    return stages


def setup_logging(output_dir: Path, timestamp: str) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")

    return logger

class EpochCheckpointCallback(TrainerCallback):
    """
    Custom callback to save checkpoints at the end of each epoch.
    These epoch checkpoints are saved separately and are not subject to save_total_limit.
    """

    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Save checkpoint at the end of each epoch."""
        # Check is_world_process_zero (always True in single-process training)
        if state.is_world_process_zero:
            epoch_num = int(state.epoch)
            checkpoint_dir = self.output_dir / "epoch_checkpoints" / f"checkpoint-epoch-{epoch_num}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Saving epoch {epoch_num} checkpoint to {checkpoint_dir}")

            # Save the model and tokenizer
            if model is not None:
                model.save_pretrained(checkpoint_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_dir)

            self.logger.info(f"Epoch {epoch_num} checkpoint saved successfully")

        return control


def load_model_and_tokenizer(
    model_name: str,
    quantization: str,
    logger: logging.Logger
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with optional quantization (none/int4/int8/fp16)."""
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Quantization: {quantization}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Log tokenizer configuration for debugging
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"Tokenizer padding side: {tokenizer.padding_side}")
    logger.info(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

    # Validate chat template exists (required for DPO training)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for {model_name} does not have a chat template. "
            f"DPO training requires a tokenizer with chat template support. "
            f"Consider using a different model or manually setting the chat_template attribute."
        )
    logger.info("✓ Tokenizer has chat template")

    # Configure quantization
    quantization_config = None
    torch_dtype = torch.bfloat16  # Default for training

    if quantization == 'int4':
        logger.info("Using 4-bit quantization (NF4)")
        logger.info("Skipping quantization for embed_tokens and lm_head (keeps them in FP16, frozen)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
            llm_int8_skip_modules=["embed_tokens", "lm_head"],  # Skip embeddings to avoid dtype mismatch
        )
    elif quantization == 'int8':
        logger.info("Using 8-bit quantization")
        logger.info("Skipping quantization for embed_tokens and lm_head (keeps them in FP16, frozen)")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["embed_tokens", "lm_head"],  # Skip embeddings to avoid dtype mismatch
        )
    elif quantization == 'fp16':
        logger.info("Using FP16 (no quantization)")
        torch_dtype = torch.float16
    elif quantization == 'none':
        logger.info("No quantization, using default dtype (bfloat16)")
    else:
        raise ValueError(f"Invalid quantization: {quantization}. Must be none/int4/int8/fp16")

    # Load model
    logger.info("Loading model with Flash Attention 2 for memory efficiency and speed")
    logger.info("Using model parallelism (device_map=auto)")

    model_kwargs = {
        "quantization_config": quantization_config,
        "torch_dtype": torch_dtype,
        "use_cache": False,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Prepare model for k-bit training if quantized
    if quantization in ['int4', 'int8']:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    logger.info(f"Model loaded on device: {model.device}")
    logger.info(f"Model dtype: {model.dtype}")

    return model, tokenizer

def setup_lora(
    model: AutoModelForCausalLM,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    logger: logging.Logger
) -> AutoModelForCausalLM:
    """
    Configure and apply LoRA to the model.

    Args:
        model: Source model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: LoRA dropout
        logger: Logger instance

    Returns:
        Model with LoRA adapters
    """
    logger.info("Configuring LoRA adapters")
    logger.info(f"  LoRA rank (r): {lora_r}")
    logger.info(f"  LoRA alpha: {lora_alpha}")
    logger.info(f"  LoRA dropout: {lora_dropout}")

    # LoRA configuration
    # Target all linear layers for maximum expressiveness (includes attention, MLP, and lm_head)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")

    return model


class SampledEvalMixin:
    """
    Mixin to add sampled evaluation to any Trainer class.

    Randomly samples a subset of the eval dataset on each evaluation step.
    This makes eval metrics comparable to training metrics (same sample size and variance).

    Usage:
        class SampledEvalDPOTrainer(SampledEvalMixin, DPOTrainer):
            pass
    """

    def __init__(self, *args, eval_sample_size: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_size = eval_sample_size

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Override to return a dataloader with a randomly sampled subset of the eval dataset.
        Uses step-based seeding for reproducibility across checkpoint resume.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # If eval_sample_size is set and we have more examples than the sample size, sample
        if self.eval_sample_size is not None and eval_dataset is not None and len(eval_dataset) > self.eval_sample_size:
            # Use step-based seeding for reproducibility across checkpoint resume
            # Each step gets a deterministic but different random sample
            current_step = self.state.global_step if hasattr(self, 'state') else 0
            step_rng = random.Random(DEFAULT_RANDOM_SEED + current_step)

            # Randomly sample indices
            indices = list(range(len(eval_dataset)))
            step_rng.shuffle(indices)
            sampled_indices = indices[:self.eval_sample_size]

            # Create a subset dataset
            sampled_dataset = eval_dataset.select(sampled_indices)

            # Log the sampling
            if current_step > 0:
                # Only log after first step to avoid cluttering initial logs
                logger = logging.getLogger(__name__)
                logger.info(f"Sampled {self.eval_sample_size} examples from {len(eval_dataset)} for evaluation at step {current_step}")

            eval_dataset = sampled_dataset

        # Call parent's method with the (possibly sampled) dataset
        return super().get_eval_dataloader(eval_dataset)


def infer_dataset_type(identifier: str, include_seed: bool = True, dpo_data_dir: str = "data/dpo_data") -> Optional[str]:
    """Infer dataset type from model name/path by matching against files in dpo_data_dir. Returns None for source models."""
    # Extract seed if present
    seed_match = re.search(r'_s(\d+)', identifier)
    seed_suffix = f"_s{seed_match.group(1)}" if (seed_match and include_seed) else ""

    # Get available dataset types from actual files in dpo_data_dir
    # Sort by length (longest first) to match compound patterns before simple ones
    data_dir = Path(dpo_data_dir)
    available_datasets = []
    if data_dir.exists() and data_dir.is_dir():
        available_datasets = sorted(
            [f.stem for f in data_dir.glob("*.jsonl")],
            key=len,
            reverse=True
        )

    # For source models with split suffixes, extract the dataset from the pattern
    # Pattern: _[(dpo)_]{dataset}_[s{N}_]split
    # Build pattern from available datasets
    if available_datasets:
        dataset_options = '|'.join(re.escape(d) for d in available_datasets)
        split_pattern = f'_((?:dpo)_)?({dataset_options})(?:_s\\d+)?_split'
        match = re.search(split_pattern, identifier)
        if match:
            finetune_prefix = match.group(1) or ""  # "dpo_" or ""
            dataset = match.group(2)
            full_dataset = f'{finetune_prefix}{dataset}'
            return f'{full_dataset}{seed_suffix}' if seed_suffix else full_dataset

    # Check for dataset markers by matching against available files
    # For sequential finetuning (e.g., Qwen2.5-7B_help_gpt5m_beta-0.05_safe_gpt5m_beta-0.05),
    # we find ALL dataset matches in order and concatenate them.
    # Build a single regex pattern matching any dataset, longest first to handle overlaps
    if available_datasets:
        # Pattern: _({dataset1}|{dataset2}|...) followed by _ or . or end
        dataset_pattern = '|'.join(re.escape(d) for d in available_datasets)
        pattern = f'_({dataset_pattern})(?=[_.]|$)'
        matches = [(m.start(), m.group(1)) for m in re.finditer(pattern, identifier)]

        if matches:
            # Sort by position to preserve order (should already be in order from finditer)
            matches.sort(key=lambda x: x[0])
            combined = '_'.join(dataset for _, dataset in matches)
            return f'{combined}{seed_suffix}'

    # Source model files without dataset suffixes return None
    return None


def get_dpo_data_path(dataset_type: str, dpo_data_dir: str = "data/dpo_data") -> str:
    """Convert dataset type to DPO data file path (e.g., "safe" -> "data/dpo_data/safe.jsonl").

    For sequential finetuning (concatenated dataset types like "help_gpt5m_safe_gpt5m"),
    uses the LAST dataset to find the file (the most recent training stage).
    """
    # Strip seed suffix if present (we don't have separate files per seed)
    base_type = re.sub(r'_s\d+', '', dataset_type)

    data_dir = Path(dpo_data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"DPO data directory not found: {dpo_data_dir}")

    # Try exact match first
    filepath = data_dir / f"{base_type}.jsonl"
    if filepath.exists():
        return str(filepath)

    # Not found - try to extract the last dataset from concatenated type (sequential finetuning)
    available_datasets = sorted([f.stem for f in data_dir.glob("*.jsonl")], key=len, reverse=True)
    if available_datasets:
        # Build regex to find all dataset matches, take the last one
        dataset_pattern = '|'.join(re.escape(d) for d in available_datasets)
        pattern = f'({dataset_pattern})(?=_|$)'
        matches = list(re.finditer(pattern, base_type))
        if matches:
            last_dataset = matches[-1].group(1)
            return str(data_dir / f"{last_dataset}.jsonl")

    raise ValueError(
        f"DPO data file not found for dataset type '{dataset_type}' (looking for {base_type}.jsonl).\n"
        f"Available datasets in {dpo_data_dir}: {sorted(available_datasets, key=len)}"
    )


def extract_seed_from_path(path: str) -> Optional[int]:
    """Extract random seed from path (looks for '_s{N}' pattern). Returns None if not found.

    Raises ValueError if multiple _s{N} matches exist with different seed values.
    """
    matches = re.findall(r'_s(\d+)', path)
    if not matches:
        return None
    unique_seeds = set(int(m) for m in matches)
    if len(unique_seeds) > 1:
        raise ValueError(
            f"Ambiguous seed in path: found multiple _s{{N}} patterns with different values "
            f"{sorted(unique_seeds)} in '{path}'"
        )
    return unique_seeds.pop()


def load_case_indices_from_finetune_data(finetune_data_path: str) -> List[int]:
    """Load and return sorted unique case indices from finetuning data file."""
    finetune_examples = load_jsonl(finetune_data_path, description="finetuning data")

    if not finetune_examples:
        raise ValueError(f"No examples found in {finetune_data_path}")

    # Extract case indices
    try:
        all_case_indices = sorted(set(ex['case_idx'] for ex in finetune_examples))
    except KeyError:
        raise ValueError(f"Missing 'case_idx' field in {finetune_data_path}")

    return all_case_indices


def compute_test_indices(seed: int) -> List[int]:
    """Compute test set by shuffling all 144 indices with seed and taking the second 50%."""
    all_indices = list(range(TOOLEMU_FULL_DATASET_SIZE))
    random.seed(seed)
    random.shuffle(all_indices)
    split_point = len(all_indices) // 2
    return sorted(all_indices[split_point:])


def partition_by_case_indices(
    items: List[Dict[str, Any]],
    train_indices: set,
    test_indices: set,
    case_idx_key: str = 'case_idx',
    logger: logging.Logger = None,
    require_train: bool = False,
    require_test: bool = False,
    allow_unknown: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Partition items into train/test based on case_idx.

    Args:
        items: List of dicts with case_idx field
        train_indices: Set of case indices for training
        test_indices: Set of case indices for testing
        case_idx_key: Key to use for case index lookup
        logger: If provided, logs partition statistics
        require_train: If True, raises ValueError if train set is empty
        require_test: If True, raises ValueError if test set is empty (otherwise warns if logger provided)
        allow_unknown: If False, raises ValueError if any item has unknown case_idx

    Returns:
        (train_items, test_items, unknown_indices)
    """
    train_items = []
    test_items = []
    unknown_indices = []

    for item in items:
        case_idx = item[case_idx_key]  # Will raise KeyError if missing

        if case_idx in train_indices:
            train_items.append(item)
        elif case_idx in test_indices:
            test_items.append(item)
        else:
            if case_idx not in unknown_indices:
                unknown_indices.append(case_idx)

    # Validation
    if not allow_unknown and unknown_indices:
        raise ValueError(
            f"case_idx {unknown_indices[0]} not in train or test set. "
            f"Unknown indices: {unknown_indices}"
        )

    if require_train and len(train_items) == 0:
        raise ValueError("No training items after partition")

    if len(test_items) == 0:
        if require_test:
            raise ValueError("No test items after partition")
        elif logger:
            logger.warning("No test/validation items - will skip validation during training")

    # Logging
    if logger:
        train_cases = set(item[case_idx_key] for item in train_items)
        test_cases = set(item[case_idx_key] for item in test_items)
        logger.info(f"Train/test split: {len(train_items)} train, {len(test_items)} test")
        logger.info(f"Train tasks: {len(train_cases)} unique cases")
        logger.info(f"Test tasks: {len(test_cases)} unique cases")

    return train_items, test_items, unknown_indices


def load_adapter_config(adapter_path: str) -> dict:
    """Load and validate adapter_config.json from adapter path.

    Raises FileNotFoundError if config doesn't exist.
    Raises ValueError if config is invalid JSON.
    Raises KeyError if required 'source_model_name_or_path' (or legacy 'base_model_name_or_path') field is missing.
    """
    config_path = Path(adapter_path) / "adapter_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found at: {config_path}"
        )

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in adapter_config.json: {e}")

    # Accept both new field name and legacy field name (PEFT writes 'base_model_name_or_path')
    if "source_model_name_or_path" not in config and "base_model_name_or_path" not in config:
        raise KeyError(
            f"adapter_config.json missing required field 'source_model_name_or_path' "
            f"(or legacy 'base_model_name_or_path'): {config_path}"
        )

    return config


def get_source_model_from_config(config: dict) -> str:
    """Extract source model path from adapter config, handling legacy field name."""
    return config.get('source_model_name_or_path') or config['base_model_name_or_path']
