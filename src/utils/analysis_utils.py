"""Shared utilities for analysis scripts."""

import glob
import os
import random
import re
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from utils.model_name_utils import clean_model_name, extract_source_model
from utils.toolemu_utils import parse_report
from utils.train_utils import DEFAULT_RANDOM_SEED


def short_dataset_name(dataset_name: str) -> str:
    """Convert dataset name to short form: help→H, safe→S, both→S&H."""
    if dataset_name == 'both':
        return 'S&H'
    elif dataset_name == 'safe':
        return 'S'
    elif dataset_name == 'help':
        return 'H'
    return dataset_name  # fallback


def calc_col_width(header: str, values: list) -> int:
    """Calculate column width to fit header and all values."""
    if not values:
        return len(header)
    return max(len(header), max(len(str(v)) for v in values))


# =============================================================================
# Statistical Utilities
# =============================================================================

def bootstrap_ci(
    compute_fn: Callable[[List[int]], float],
    case_indices: List[int],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Compute bootstrap CI for a statistic. Raises ValueError if compute_fn returns None/NaN."""
    # Compute point estimate on original data
    estimate = compute_fn(case_indices)
    if estimate is None:
        raise ValueError("compute_fn returned None on original data")
    if np.isnan(estimate):
        raise ValueError("compute_fn returned NaN on original data")

    # Bootstrap resampling
    rng = random.Random(DEFAULT_RANDOM_SEED)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        resampled = rng.choices(case_indices, k=len(case_indices))
        stat = compute_fn(resampled)

        if stat is None:
            raise ValueError("compute_fn returned None during bootstrap iteration")
        if np.isnan(stat):
            raise ValueError("compute_fn returned NaN during bootstrap iteration")

        bootstrap_stats.append(stat)

    # Compute CI using percentile method
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    std_error = float(np.std(bootstrap_stats))

    return {
        'estimate': estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_error': std_error,
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_results_data(
    results_path: str,
    data_dir: str = 'data/dpo_data'
) -> List[Dict[str, Any]]:
    """Load results data from unified report files.

    Args:
        results_path: Directory or single file path containing unified reports.
        data_dir: Directory for DPO data files (for training stage inference).

    Returns:
        List of dicts with model info, metrics, and training stages.
    """
    report_files = []

    if os.path.isdir(results_path):
        # Scan directory for unified report files (both standard and aggregated)
        pattern1 = os.path.join(results_path, "*_unified_report.json")
        pattern2 = os.path.join(results_path, "*_unified_report_*.json")
        report_files.extend(glob.glob(pattern1))
        report_files.extend(glob.glob(pattern2))
    elif os.path.isfile(results_path):
        report_files.append(results_path)

    data = []
    for report_path in report_files:
        result = parse_report(report_path, data_dir)
        if result is None:
            continue

        helpfulness = result.get('avg_helpfulness_ignore_safety')
        safety = result.get('avg_safety')

        # Skip if either metric is missing
        if helpfulness is None or safety is None:
            continue

        data.append({
            'model_name': result['model_name'],
            'source_model': extract_source_model(result['model_name']),
            'helpfulness': helpfulness,
            'safety': safety,
            'training_stages': result['training_stages'],
            'n_samples': result.get('n_help_ignore'),
            'evaluator_model': result['evaluator_model'],
            'emulator_model': result['emulator_model'],
            'quantization': result['quantization'],
            'last_finetune_type': result['last_finetune_type'],
        })

    return data


# =============================================================================
# Aggregation
# =============================================================================

# Valid dimensions for aggregation group_by
VALID_GROUP_BY_DIMENSIONS = {
    'source_model', 'evaluator_model', 'emulator_model', 'quantization',
    'last_finetune_type', 'num_stages', 'training_stages', 'metric_sequence'
}

# Dimensions that are training-specific (excluded when matching finetuned to base)
TRAINING_SPECIFIC_DIMENSIONS = {'last_finetune_type', 'num_stages', 'training_stages', 'metric_sequence'}


def aggregate_scores(
    data: List[Dict[str, Any]],
    group_by: List[str],
    numeric_fields: List[str] = None,
) -> List[Dict[str, Any]]:
    """Aggregate raw scores by grouping dimensions. Dims not in group_by get averaged.

    Args:
        data: List of data point dicts.
        group_by: Dimensions to group by.
        numeric_fields: Fields to average (default: ['helpfulness', 'safety']).

    Returns data points in the same format as input, with:
    - Grouped dimensions preserved
    - Non-grouped dimensions set to None
    - Numeric fields averaged with _std stats added
    - n_models count added
    """
    if not data:
        return []

    if numeric_fields is None:
        numeric_fields = ['helpfulness', 'safety']

    # Validate dimensions
    for dim in group_by:
        if dim not in VALID_GROUP_BY_DIMENSIONS:
            raise ValueError(f"Invalid grouping dimension: {dim}. Valid: {VALID_GROUP_BY_DIMENSIONS}")

    # Group data by the specified dimensions
    groups = {}
    for d in data:
        # Build group key tuple
        key_parts = []
        for dim in group_by:
            if dim == 'num_stages':
                key_parts.append(len(d.get('training_stages', [])))
            elif dim == 'training_stages':
                key_parts.append(tuple(d.get('training_stages', [])))
            elif dim == 'metric_sequence':
                # Extract just the dataset names (without betas) from training_stages
                stages = d.get('training_stages', [])
                key_parts.append(tuple(dataset for dataset, _ in stages))
            else:
                key_parts.append(d.get(dim))
        key = tuple(key_parts)

        if key not in groups:
            groups[key] = []
        groups[key].append(d)

    # Aggregate each group
    result = []
    for key, items in groups.items():
        # Start with first item as template, then override
        agg = dict(items[0])

        # Set non-grouped fields to None
        agg['model_name'] = None
        if 'source_model' not in group_by:
            agg['source_model'] = None
        if 'evaluator_model' not in group_by:
            agg['evaluator_model'] = None
        if 'emulator_model' not in group_by:
            agg['emulator_model'] = None
        if 'quantization' not in group_by:
            agg['quantization'] = None

        # Set grouped dimension values from key
        for i, dim in enumerate(group_by):
            if dim == 'training_stages':
                # Convert back to list for compatibility with format_label etc.
                agg[dim] = list(key[i])
            elif dim == 'metric_sequence':
                # Keep as tuple (immutable, used for grouping)
                agg[dim] = key[i]
            else:
                agg[dim] = key[i]

        # Compute averaged scores and stats for each numeric field
        for field in numeric_fields:
            values = [item[field] for item in items if item.get(field) is not None]
            if values:
                agg[field] = statistics.mean(values)
                agg[f'{field}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                agg[field] = None
                agg[f'{field}_std'] = None

        agg['n_models'] = len(items)
        agg['n_samples'] = sum(item.get('n_samples', 0) for item in items)

        # Set label for aggregated data grouped by last_finetune_type (without betas)
        if 'last_finetune_type' in group_by and 'training_stages' not in group_by:
            lft = agg.get('last_finetune_type')
            if lft:
                agg['label'] = short_dataset_name(lft)

        # Set label for aggregated data grouped by metric_sequence (without betas)
        if 'metric_sequence' in group_by and 'training_stages' not in group_by:
            ds = agg.get('metric_sequence')
            if ds:
                agg['label'] = ','.join(short_dataset_name(d) for d in ds)

        result.append(agg)

    return result


def compute_deltas_from_aggregated(
    aggregated: List[Dict[str, Any]],
    group_by: List[str],
    relative_to: str = 'base',
) -> List[Dict[str, Any]]:
    """Compute deltas by matching aggregated finetuned groups to reference groups.

    Args:
        aggregated: Aggregated data points from aggregate_scores().
        group_by: Dimensions used for grouping.
        relative_to: 'base' computes delta from original source model (default),
                     'parent' computes delta from direct parent (one fewer training stage).
    """
    if not aggregated:
        return []

    if relative_to not in ('base', 'parent'):
        raise ValueError("relative_to must be 'base' or 'parent'")

    # Separate base groups (training_stages=() or last_finetune_type=None) from finetuned
    base_groups = []
    finetuned_groups = []
    for agg in aggregated:
        # Check if this is a base group
        is_base = False
        if 'training_stages' in group_by:
            is_base = agg.get('training_stages') == () or agg.get('training_stages') == []
        elif 'metric_sequence' in group_by:
            is_base = agg.get('metric_sequence') == () or agg.get('metric_sequence') == []
        elif 'last_finetune_type' in group_by:
            is_base = agg.get('last_finetune_type') is None
        elif 'num_stages' in group_by:
            is_base = agg.get('num_stages') == 0
        else:
            # If no training-specific dimension in group_by, can't distinguish base
            # Treat all as finetuned (will match to any base in the aggregated data)
            is_base = False

        if is_base:
            base_groups.append(agg)
        else:
            finetuned_groups.append(agg)

    # Build matching keys (non-training dimensions)
    match_dims = [d for d in group_by if d not in TRAINING_SPECIFIC_DIMENSIONS]

    def get_match_key(agg):
        return tuple(agg.get(dim) for dim in match_dims)

    # Index base groups by match key
    base_by_key = {}
    for base in base_groups:
        key = get_match_key(base)
        base_by_key[key] = base

    # For parent mode, also index all groups by (match_key, training_stages)
    if relative_to == 'parent' and 'training_stages' in group_by:
        all_by_stages = {}
        for agg in aggregated:
            key = get_match_key(agg)
            stages = tuple(agg.get('training_stages', []))
            all_by_stages[(key, stages)] = agg

    # Compute deltas
    deltas = []
    for ft in finetuned_groups:
        key = get_match_key(ft)

        if relative_to == 'parent' and 'training_stages' in group_by:
            # Find direct parent: same match_key, training_stages[:-1]
            ft_stages = ft.get('training_stages', [])
            parent_stages = tuple(ft_stages[:-1])
            reference = all_by_stages.get((key, parent_stages))
        else:
            # Default: use original source model
            reference = base_by_key.get(key)

        if reference is None:
            print(f"Warning: No matching {'parent' if relative_to == 'parent' else 'base'} group found for {ft}")
            continue

        delta = dict(ft)  # Copy all fields from finetuned
        delta['helpfulness_delta'] = ft['helpfulness'] - reference['helpfulness']
        delta['safety_delta'] = ft['safety'] - reference['safety']
        delta['base_helpfulness'] = reference['helpfulness']
        delta['base_safety'] = reference['safety']
        deltas.append(delta)

    return deltas


# =============================================================================
# Filtering
# =============================================================================

def _is_ancestor(ancestor: Dict[str, Any], descendant: Dict[str, Any]) -> bool:
    """Check if ancestor's stages are a proper prefix of descendant's stages."""
    if ancestor['source_model'] != descendant['source_model']:
        return False
    ancestor_stages = ancestor.get('training_stages', [])
    descendant_stages = descendant.get('training_stages', [])
    if len(ancestor_stages) >= len(descendant_stages):
        return False
    return descendant_stages[:len(ancestor_stages)] == ancestor_stages


def filter_data(
    data: List[Dict[str, Any]],
    model: str = None,
    first_finetune: str = None,
    last_finetune: str = None,
    include_intermediates: bool = False,
    disallowed_num_stages: List[int] = None,
    disallowed_betas: List[float] = None,
    disallowed_metrics: List[str] = None
) -> List[Dict[str, Any]]:
    """Filter data points by model and/or finetune attributes.

    Args:
        data: List of data point dicts.
        model: Filter by source_model (supports nicknames like "Qwen2.5-7B" or full names).
        first_finetune: Filter by first training stage dataset name.
        last_finetune: Filter by last training stage dataset name.
        include_intermediates: If True and last_finetune is set, also include
            intermediate states in finetune chains leading to matching models.
        disallowed_num_stages: List of stage counts to exclude. Excludes if number of
            training stages matches any value in the list (e.g., [2] excludes 2-stage
            finetunes, [0] excludes source models).
        disallowed_betas: List of beta values to exclude. Excludes if ANY training stage
            uses one of these betas. Source models are never excluded.
        disallowed_metrics: List of metric names to exclude. Excludes if ANY training
            stage contains any of these metrics (substring match). Source models are
            never excluded.

    Returns filtered list preserving original order.
    """
    model_filter = clean_model_name(model, format="short") if model else None
    # Convert disallowed_betas to set of strings for comparison
    disallowed_betas_set = set(str(b) for b in disallowed_betas) if disallowed_betas else None
    # Convert disallowed_num_stages to set for O(1) lookup
    disallowed_num_stages_set = set(disallowed_num_stages) if disallowed_num_stages else None

    def passes_filters(d: Dict[str, Any], check_last: bool = True) -> bool:
        """Check if data point passes model and finetune filters."""
        if model_filter is not None:
            source_model_clean = clean_model_name(d['source_model'], format="short")
            if model_filter not in source_model_clean:
                return False

        stages = d.get('training_stages', [])

        if disallowed_num_stages_set and len(stages) in disallowed_num_stages_set:
            return False

        if first_finetune is not None and stages:
            if first_finetune not in stages[0][0]:
                return False

        if check_last and last_finetune is not None and stages:
            if last_finetune not in stages[-1][0]:
                return False

        # Check disallowed_betas: exclude if ANY stage uses a disallowed beta
        if disallowed_betas_set and stages:
            for _, beta in stages:
                if str(beta) in disallowed_betas_set:
                    return False

        # Check disallowed_metrics: exclude if ANY stage matches a disallowed metric (substring match)
        if disallowed_metrics and stages:
            for dataset, _ in stages:
                for metric in disallowed_metrics:
                    if metric in dataset:
                        return False

        return True

    result = [d for d in data if passes_filters(d)]

    # If include_intermediates, add ancestors of matching models
    if include_intermediates and last_finetune is not None:
        result_set = set(id(d) for d in result)
        intermediates = []
        for d in data:
            if id(d) in result_set:
                continue
            if not passes_filters(d, check_last=False):
                continue
            if any(_is_ancestor(d, matched) for matched in result):
                intermediates.append(d)

        # Return in original order
        intermediate_set = set(id(d) for d in intermediates)
        return [d for d in data if id(d) in result_set or id(d) in intermediate_set]

    return result


# =============================================================================
# Labeling and Formatting
# =============================================================================

def format_label(data: Dict[str, Any]) -> str:
    """Format label for a finetuned model. Shows training stages with betas.

    Format: S for safe, H for help. If all betas are the same, show single value at end.
    Examples:
      - Single stage: "S-0.05" or "H-0.1"
      - Same betas: "S,H-0.1"
      - Different betas: "S-0.05,H-0.1"

    Raises ValueError if called with no training stages (source models should not call this).
    """
    training_stages = data.get('training_stages', [])
    if not training_stages:
        raise ValueError("format_label should not be called for source models (no training stages)")

    # Check if all betas are the same
    betas = [beta for _, beta in training_stages]
    all_same_beta = len(set(betas)) == 1

    if all_same_beta:
        # S,H-0.1
        names = [short_dataset_name(dataset) for dataset, _ in training_stages]
        return f"{','.join(names)}-{betas[0]}"
    else:
        # S-0.05,H-0.1
        parts = [f"{short_dataset_name(dataset)}-{beta}" for dataset, beta in training_stages]
        return ",".join(parts)


def get_evaluator_short_name(eval_model: str) -> str:
    """Convert evaluator model to short name for filenames."""
    eval_clean = clean_model_name(eval_model, format='short').lower()
    if 'gpt-5-mini' in eval_clean or 'gpt5m' in eval_clean or 'gpt-5m' in eval_clean:
        return 'eval-gpt5m'
    elif 'qwen' in eval_clean and '32' in eval_clean:
        return 'eval-q32'
    elif 'gpt' in eval_clean:
        return f'eval-{eval_clean}'
    else:
        return f'eval-{eval_clean.replace("/", "_")}'


def get_present_types(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze data to determine what types of models/finetunes are present.

    Returns dict with:
        source_models: set of source model names (excludes None)
        finetune_types: set of finetune types ('help', 'safe', 'combined')
        has_source_model_points: bool, True if any points have no finetuning
    """
    source_models = set()
    finetune_types = set()
    has_source_model_points = False

    for d in data:
        if d.get('source_model') is not None:
            source_models.add(d['source_model'])
        lft = d.get('last_finetune_type')
        if lft == 'both':
            finetune_types.add('combined')
        elif lft == 'safe':
            finetune_types.add('safe')
        elif lft == 'help':
            finetune_types.add('help')
        elif lft is None:
            has_source_model_points = True

    return {
        'source_models': source_models,
        'finetune_types': finetune_types,
        'has_source_model_points': has_source_model_points,
    }


# =============================================================================
# Training Metrics Loading (from DPO SLURM logs)
# =============================================================================

# Regex to match the first metrics log line
TRAINING_METRICS_PATTERN = re.compile(
    r"\{'loss': ([\d.]+).*?'rewards/accuracies': ([\d.]+)"
)


def get_latest_slurm_log(slurm_dir: Path) -> Path | None:
    """Find the SLURM log with the highest job ID."""
    logs = list(slurm_dir.glob("slurm-*.out"))
    if not logs:
        return None
    def get_job_id(path: Path) -> int:
        match = re.search(r"slurm-(\d+)\.out", path.name)
        return int(match.group(1)) if match else 0
    return max(logs, key=get_job_id)


def extract_first_metrics(log_path: Path) -> tuple:
    """Extract the first loss and accuracy from a SLURM log."""
    try:
        with open(log_path, 'r', errors='ignore') as f:
            for line in f:
                match = TRAINING_METRICS_PATTERN.search(line)
                if match:
                    loss = float(match.group(1))
                    accuracy = float(match.group(2))
                    return loss, accuracy
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    return None, None


def load_training_metrics(trained_models_dir: str, data_dir: str = 'data/dpo_data') -> List[Dict[str, Any]]:
    """Load starting loss and accuracy from trained model directories.

    Returns data compatible with aggregate_scores and generate_scatter_plot.
    """
    from utils.train_utils import extract_training_stages

    trained_path = Path(trained_models_dir)
    if not trained_path.exists():
        print(f"Directory not found: {trained_models_dir}")
        return []

    results = []
    for model_dir in sorted(trained_path.iterdir()):
        if not model_dir.is_dir():
            continue

        slurm_dir = model_dir / "slurm_output"
        if not slurm_dir.exists():
            continue

        log_path = get_latest_slurm_log(slurm_dir)
        if not log_path:
            continue

        loss, accuracy = extract_first_metrics(log_path)
        if loss is None or accuracy is None:
            continue

        # Extract training stages from the directory name
        training_stages = extract_training_stages(model_dir.name, data_dir)

        # Extract source model (finetuned models are identified by seed suffix)
        source_model = extract_source_model(model_dir.name)

        results.append({
            'model_name': model_dir.name,
            'training_stages': training_stages,
            'start_loss': loss,
            'start_acc': accuracy,
            # Same pattern as parse_report in toolemu_utils.py
            'last_finetune_type': training_stages[-1][0] if training_stages else None,
            'source_model': source_model,
        })

    return results
