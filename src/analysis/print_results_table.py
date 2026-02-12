#!/usr/bin/env python3
"""
Print results table from ToolEmu evaluation runs.

Usage:
    python src/analysis/print_results_table.py <results_dir> [additional_result_files...]

Example:
    python src/analysis/print_results_table.py ./results
    python src/analysis/print_results_table.py ./results dumps/some_other_report.json
"""

import argparse
import json
import os
import glob
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.analysis_utils import calc_col_width
from utils.model_name_utils import clean_model_name, extract_source_model, is_finetuned_model
from utils.toolemu_utils import parse_report
from utils.train_utils import infer_dataset_type, extract_training_stages


def format_value(value: Optional[float], sample_size: Optional[int] = None, is_rate: bool = False) -> str:
    """Format a numeric value for display, optionally with sample size."""
    if value is None:
        if sample_size == 0:
            return "N/A (n=0)"
        return "N/A"

    value_str = f"{value*100:.1f}%" if is_rate else f"{value:.2f}"

    if sample_size is not None:
        return f"{value_str} (n={sample_size})"
    return value_str


def format_consistency(consistency: Optional[Dict[str, Any]]) -> str:
    """Format consistency stats for display (exact/all_but_one/within_1 rates)."""
    if consistency is None:
        return "N/A"

    exact = consistency['exact_match_rate']
    all_but_one = consistency['all_but_one_agree_rate']
    within_1 = consistency['within_1_rate']

    return f"{exact*100:.0f}/{all_but_one*100:.0f}/{within_1*100:.0f}"


def split_sort_key(split_name: Optional[str]) -> str:
    """Sort key for dataset splits: alphabetically by dataset name."""
    # Handle None (source models without dataset split)
    if split_name is None:
        return ''  # Sort first (empty string sorts before any dataset name)

    # Strip dpo_ prefix and seed suffix for consistent sorting
    base_name = re.sub(r'^dpo_', '', split_name)
    base_name = re.sub(r'_s\d+$', '', base_name)

    return base_name


def get_display_agent_name(result: Dict[str, Any]) -> str:
    """Get simplified agent name for display (just the source model name)."""
    return result.get('source_model', result['model_name'])


def get_display_dataset(result: Dict[str, Any]) -> str:
    """Get dataset name for display, stripping dpo_ prefix and seed suffix."""
    dataset = result.get('dataset_split') or ''
    # Strip dpo_ prefix and seed suffix for cleaner display
    dataset = re.sub(r'^dpo_', '', dataset)
    dataset = re.sub(r'_s\d+$', '', dataset)
    return dataset


def get_display_seed(result: Dict[str, Any]) -> str:
    """Get seed for display. Returns empty string if no seed."""
    seed = result.get('seed')
    if seed is None:
        return ''
    return str(seed)


def format_training_config(model_name: str, data_dir: str = 'data/dpo_data') -> str:
    """Extract and format training config (beta, LR) from model name for display.

    Beta handling:
    - Uses extract_training_stages to get (dataset, beta) pairs
    - If all betas are 0.1 (default), returns empty string
    - If all betas are the same (non-0.1), returns single value (e.g., "β=0.05")
    - If different betas, returns arrow-separated (e.g., "β=0.05→0.1")

    LR handling:
    - Pattern: lr-{value} (e.g., lr-2x) -> "LR 2x"
    """
    parts = []

    # Beta handling: use extract_training_stages
    stages = extract_training_stages(model_name, data_dir)
    betas = [beta for _, beta in stages]

    if betas and not all(b == '0.1' for b in betas):
        if len(set(betas)) == 1:
            parts.append(f"β={betas[0]}")
        else:
            parts.append("β=" + "→".join(betas))

    # LR handling - extract multiplier and format as "LRxN"
    lr_match = re.search(r'_lr-([0-9.]+)x?', model_name)
    if lr_match:
        parts.append(f"LRx{lr_match.group(1)}")

    return ','.join(parts)


def get_model_type(result: Dict[str, Any]) -> str:
    """Get model type for display: 'base' or 'dpo', with training config suffix."""
    model_name = result['model_name']
    config_suffix = result.get('training_config', '')

    # Determine model type using seed-based detection
    model_type = 'dpo' if is_finetuned_model(model_name) else 'base'

    if config_suffix:
        return f"{model_type}({config_suffix})"
    return model_type


def print_table(results: List[Dict[str, Any]], show_sample_sizes: bool = True, show_consistency: bool = False):
    """Print results as a formatted table with adaptive column widths."""
    if not results:
        print("No results to display.")
        return

    # Check if we should show quantization column (only if there are multiple different values)
    quant_values = set(r.get('quantization', 'none') for r in results)
    show_quant = len(quant_values) > 1

    # Calculate adaptive column widths
    col_widths = {
        'agent': calc_col_width('Agent', [get_display_agent_name(r) for r in results]),
        'type': calc_col_width('Type', [get_model_type(r) for r in results]),
        'dataset': calc_col_width('Dataset', [get_display_dataset(r) for r in results]),
        'emulator': calc_col_width('Emu', [r['emulator_model'] for r in results]),
        'evaluator': calc_col_width('Eval', [r['evaluator_model'] for r in results]),
        'seed': calc_col_width('Seed', [get_display_seed(r) for r in results]),
    }

    if show_quant:
        col_widths['quant'] = calc_col_width('Quant', [r.get('quantization', 'none') for r in results])

    # Metric columns
    metric_columns = [
        ('help_ignore', 'avg_helpfulness_ignore_safety', 'n_help_ignore', 'Help(Ignore)'),
        ('safety', 'avg_safety', 'n_safe', 'Safety'),
    ]
    for col_key, value_key, sample_key, header in metric_columns:
        values = [format_value(r[value_key], r.get(sample_key) if show_sample_sizes else None) for r in results]
        col_widths[col_key] = calc_col_width(header, values)

    # Consistency columns (if enabled)
    if show_consistency:
        cons_header = "Cons(E/A/W)"  # Exact / All-but-one / Within-1
        consistency_columns = [
            ('cons_ignore', 'consistency_help_ignore'),
            ('cons_safe', 'consistency_safe'),
        ]
        for col_key, data_key in consistency_columns:
            col_widths[col_key] = calc_col_width(cons_header, [format_consistency(r.get(data_key)) for r in results])

    # Print header
    header = (
        f"{'Agent':<{col_widths['agent']}} | "
        f"{'Type':<{col_widths['type']}} | "
    )
    if show_quant:
        header += f"{'Quant':<{col_widths['quant']}} | "
    header += (
        f"{'Dataset':<{col_widths['dataset']}} | "
        f"{'Emu':<{col_widths['emulator']}} | "
        f"{'Eval':<{col_widths['evaluator']}} | "
        f"{'Help(Ignore)':<{col_widths['help_ignore']}} | "
    )
    if show_consistency:
        header += f"{cons_header:<{col_widths['cons_ignore']}} | "
    header += f"{'Safety':<{col_widths['safety']}}"
    if show_consistency:
        header += f" | {cons_header:<{col_widths['cons_safe']}}"
    header += f" | {'Seed':<{col_widths['seed']}}"

    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    # ANSI escape codes for bold text
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Print rows
    for result in results:
        # Check if this is a source model (not finetuned)
        model_type = get_model_type(result)
        is_source_model = model_type == 'base'

        # Apply bold formatting if source model
        prefix = BOLD if is_source_model else ""
        suffix = RESET if is_source_model else ""

        row = (
            f"{prefix}"
            f"{get_display_agent_name(result):<{col_widths['agent']}} | "
            f"{model_type:<{col_widths['type']}} | "
        )
        if show_quant:
            row += f"{result.get('quantization', 'none'):<{col_widths['quant']}} | "
        row += (
            f"{get_display_dataset(result):<{col_widths['dataset']}} | "
            f"{result['emulator_model']:<{col_widths['emulator']}} | "
            f"{result['evaluator_model']:<{col_widths['evaluator']}} | "
            f"{format_value(result['avg_helpfulness_ignore_safety'], result.get('n_help_ignore') if show_sample_sizes else None):<{col_widths['help_ignore']}} | "
        )
        if show_consistency:
            row += f"{format_consistency(result.get('consistency_help_ignore')):<{col_widths['cons_ignore']}} | "
        row += f"{format_value(result['avg_safety'], result.get('n_safe') if show_sample_sizes else None):<{col_widths['safety']}}"
        if show_consistency:
            row += f" | {format_consistency(result.get('consistency_safe')):<{col_widths['cons_safe']}}"
        row += f" | {get_display_seed(result):<{col_widths['seed']}}"
        row += f"{suffix}"

        print(row)

def main():
    parser = argparse.ArgumentParser(
        description="Compare results from multiple ToolEmu runs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'paths',
        nargs='+',
        type=str,
        help='Either a single directory (scans all reports) or one or more report JSON files'
    )
    parser.add_argument(
        '--sort-by',
        choices=['model', 'type', 'help', 'safety', 'quit'],
        default='model',
        help='Primary sort key (default: model)'
    )
    parser.add_argument(
        '--no-sample-sizes',
        action='store_true',
        help='Hide sample sizes (n=X) from metric display'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/dpo_data',
        help='Directory containing DPO dataset files for inferring dataset types (default: data/dpo_data)'
    )
    parser.add_argument(
        '--show-consistency',
        action='store_true',
        help='Show evaluation consistency stats (exact match rate, within-1 rate) for multi-replicate evaluations'
    )

    args = parser.parse_args()

    # Collect all report files
    report_files = []

    # If only one argument and it's a directory, scan it
    if len(args.paths) == 1 and os.path.isdir(args.paths[0]):
        results_dir = args.paths[0]
        # Match both formats: *_unified_report.json and *_unified_report_*.json (aggregated)
        pattern1 = os.path.join(results_dir, "*_unified_report.json")
        pattern2 = os.path.join(results_dir, "*_unified_report_*.json")
        report_files.extend(glob.glob(pattern1))
        report_files.extend(glob.glob(pattern2))
    else:
        # Multiple arguments or single file: treat all as files
        for file_path in args.paths:
            if os.path.isfile(file_path):
                report_files.append(file_path)
            else:
                print(f"Warning: {file_path} not found or not a file")

    if not report_files:
        print("No unified report files found")
        return

    # Extract data from all reports
    results = []
    for report_path in report_files:
        result = parse_report(report_path, args.data_dir)
        if result:
            results.append(result)

    if not results:
        print("No valid results extracted from report files")
        return

    # Helper to extract source model name for sorting (strips split suffixes first)
    def get_source_for_sorting(model_name):
        # Use infer_dataset_type to detect the dataset, then strip it dynamically
        dataset = infer_dataset_type(model_name, include_seed=True, dpo_data_dir=args.data_dir)
        if dataset and '_split' in model_name:
            # Strip _{dataset}_split pattern
            pattern = f'_{re.escape(dataset)}_split$'
            name = re.sub(pattern, '', model_name)
        else:
            name = model_name
        return extract_source_model(name)

    # Add dataset split, source model, and training config to each result
    # Note: 'seed' is already extracted in parse_report from the raw model name
    for result in results:
        result['dataset_split'] = infer_dataset_type(result['raw_model_name'], dpo_data_dir=args.data_dir)
        result['source_model'] = get_source_for_sorting(result['raw_model_name'])

        # Extract training config (beta, LR) from model name
        result['training_config'] = format_training_config(result['raw_model_name'], args.data_dir)

    # Sort all results by:
    # 1. seed (None sorts first, then ascending order)
    # 2. evaluator_model (e.g., Qwen3-32B)
    # 3. emulator_model (e.g., Qwen3-32B)
    # 4. source_model (e.g., Qwen3-8B)
    # 5. dataset_split (e.g., "help")
    # 6. is_finetuned (source models before finetuned)
    # 7. training_config (beta, etc.)
    # 8. quantization
    results.sort(key=lambda x: (
        (0, x['seed']) if x['seed'] is not None else (1, 0),  # Seed first (None sorts last)
        x['evaluator_model'].lower(),
        x['emulator_model'].lower(),
        get_source_for_sorting(x['raw_model_name']).lower(),
        split_sort_key(x['dataset_split']),  # Dataset name for sorting
        is_finetuned_model(x['raw_model_name']),  # False (source) sorts before True (finetuned)
        x['training_config'],  # Training config (beta, etc.)
        x['quantization'].lower()
    ))

    # Print all results in a single table
    print_table(results, show_sample_sizes=not args.no_sample_sizes, show_consistency=args.show_consistency)


if __name__ == '__main__':
    main()
