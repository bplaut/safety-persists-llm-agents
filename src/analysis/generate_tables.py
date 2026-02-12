#!/usr/bin/env python3
"""
Generate LaTeX tables for ToolEmu evaluation results and hyperparameters.

Usage:
    python src/analysis/generate_tables.py <results_dir> [--output-dir figs]
    python src/analysis/generate_tables.py <results_dir> --hyperparameters-only
    python src/analysis/generate_tables.py <results_dir> --results-only
    python src/analysis/generate_tables.py <results_dir> --aggregated-results-only
    python src/analysis/generate_tables.py <results_dir> --aggregated-deltas-only
    python src/analysis/generate_tables.py <results_dir> --persistence-only

Generates tables in the output directory:
    - hyperparams.tex: Table of all hyperparameters used in training
    - {evaluator}_results.tex: Table of agent performance (one file per evaluator)
    - aggregated_results_table.tex: Results aggregated by training config
    - aggregated_deltas_table.tex: Delta from source model by training config
    - persist.tex: Persistence of training objectives by beta
    - training_metrics.tex: Starting loss/accuracy from DPO training
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.analysis_utils import (
    short_dataset_name, format_label, load_training_metrics,
    load_results_data, aggregate_scores, compute_deltas_from_aggregated,
    get_evaluator_short_name
)
from utils.model_name_utils import clean_model_name, model_sort_key

import re


def save_table(content: str, filename: str, output_dir: str, description: str) -> None:
    """Save a LaTeX table to a file and print confirmation."""
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Generated {description} table: {output_path}")


def parse_model_seed_key(key: str) -> Tuple[str, int]:
    """Parse a model key that may include a seed suffix like 'Llama-8B_s42'.

    Returns (base_model_name, seed) where seed is None if no suffix found.
    """
    match = re.match(r'^(.+)_s(\d+)$', key)
    if match:
        return match.group(1), int(match.group(2))
    return key, None


def model_seed_sort_key(key: str):
    """Sort key for model names that may include seed suffixes.

    Sorts by base model name first, then by seed.
    """
    base_model, seed = parse_model_seed_key(key)
    return (model_sort_key(base_model), seed or 0)


# =============================================================================
# Hyperparameter Table
# =============================================================================

# All hyperparameters as a flat list: (parameter_name, value)
HYPERPARAMETERS = [
    ("Learning rate", "5e-5"),
    ("Batch size", "1"),
    ("Gradient accumulation", "8"),
    ("Num epochs", "1"),
    ("Warmup ratio", "0.1"),
    ("Beta", "various"),
    ("LoRA rank ($r$)", "16"),
    ("LoRA alpha ($\\alpha$)", "32"),
    ("LoRA dropout", "0.05"),
    ("LoRA target modules", "\\{q,k,v,o,gate,up,down\\}\\_proj"),
    ("Optimizer", "AdamW"),
    ("LR scheduler", "Cosine"),
    ("Max grad norm", "1.0"),
    ("Quantization level", "int4"),
    ("Temperature", "0.0"),
    ("Max agent tokens", "8000"),
    ("Max re-prompts for agent action", "5"),
]


def generate_hyperparameter_table() -> str:
    """Generate the LaTeX hyperparameter table string."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Hyperparameters used for training and evaluation.}",
        r"\label{tab:hyperparameters}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
    ]

    for name, value in HYPERPARAMETERS:
        lines.append(f"{name} & {value} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Results Table
# =============================================================================

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    return text.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def format_label_latex(data: Dict[str, Any]) -> str:
    """Format training config label for LaTeX. Returns 'base' for source models."""
    stages = data.get('training_stages', [])
    if not stages:
        return "base"
    return format_label(data).replace('&', r'\&')


def training_sort_key(x, include_model: bool = True):
    """Sort key for training configurations.

    Args:
        x: Dict with training_stages and optionally source_model.
        include_model: If True, group by source_model first.

    Sorting order:
    1. Source model (if include_model=True)
    2. Source models before finetuned
    3. Last dataset (H < S < S&H)
    4. Number of stages (single before multi)
    5. Earlier stages
    6. Beta (last)
    """
    stages = x.get('training_stages', [])

    if include_model:
        source_model = x.get('source_model')
        model_key = model_sort_key(source_model) if source_model else ()
    else:
        model_key = ()

    if not stages:
        return (model_key, 0, '', 0, (), '')  # Source model config comes first

    last_dataset = short_dataset_name(stages[-1][0])
    # S&H should sort after S
    last_dataset_sort = last_dataset.replace('S&H', 'S\uffff')
    last_beta = str(stages[-1][1])
    earlier = tuple((short_dataset_name(s[0]), str(s[1])) for s in stages[:-1])

    return (model_key, 1, last_dataset_sort, len(stages), earlier, last_beta)


DEFAULT_COLUMNS = [('Helpfulness', 'helpfulness'), ('Safety', 'safety')]
DELTA_COLUMNS = [('Helpfulness $\\Delta$', 'helpfulness_delta'), ('Safety $\\Delta$', 'safety_delta')]

def generate_results_table(
    results: List[Dict[str, Any]],
    caption: str,
    label: str = "tab:results",
    columns: List[Tuple[str, str]] = None,
    show_sign: bool = False,
) -> str:
    """Generate LaTeX table for results. Auto-detects format based on training configs.

    If no results have training_stages, uses simple format: Model | Helpfulness | Safety
    Otherwise uses full format: Source Model | Training Config | Helpfulness | Safety

    Quantization column is auto-added if multiple quantization levels exist.

    Args:
        results: List of result dicts with training_stages, source_model, and score fields.
        caption: Table caption.
        label: LaTeX label for the table.
        columns: List of (header, field_name) tuples for score columns.
                 Defaults to [('Helpfulness', 'helpfulness'), ('Safety', 'safety')].
        show_sign: If True, format numbers with explicit +/- sign (for deltas).
    """
    if not results:
        return "% No results to display"

    if columns is None:
        columns = DEFAULT_COLUMNS

    results = sorted(results, key=training_sort_key)

    # Detect if any results have training configs (non-empty training_stages)
    has_training_configs = any(r.get('training_stages') for r in results)

    # Detect if we should show quantization column (multiple different values)
    quant_values = set(r.get('quantization', 'none') for r in results)
    show_quantization = len(quant_values) > 1

    if has_training_configs:
        # Full format: Source Model | Training Config | scores
        include_source_model = any(r.get('source_model') is not None for r in results)
        num_text_cols = 2 if include_source_model else 1
    else:
        # Simple format: Model | scores (no Training Config column)
        include_source_model = False
        num_text_cols = 1

    # Add column for quantization if needed
    if show_quantization:
        num_text_cols += 1

    col_spec = 'l' * num_text_cols + 'c' * len(columns)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\centering")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Build header row
    header_parts = []
    if has_training_configs:
        if include_source_model:
            header_parts.append("Source Model")
        header_parts.append("Training Config")
    else:
        header_parts.append("Model")
    if show_quantization:
        header_parts.append("Quantization")
    header_parts.extend(header for header, _ in columns)
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    for result in results:
        row_parts = []

        if has_training_configs:
            # Full format with training config
            if include_source_model:
                source_model = result.get('source_model', '')
                model_name = escape_latex(clean_model_name(source_model, format="long"))
                row_parts.append(model_name)
            config_str = format_label_latex(result)
            row_parts.append(config_str)
        else:
            # Simple format: just model name
            # Use source_model if available, otherwise try to get from other fields
            model = result.get('source_model') or result.get('model_name', '')
            model_name = escape_latex(clean_model_name(model, format="long"))
            row_parts.append(model_name)

        if show_quantization:
            quant = result.get('quantization', 'none')
            row_parts.append(escape_latex(quant))

        for _, field in columns:
            val = result.get(field)
            if val is not None:
                val_str = f"{val:+.2f}" if show_sign else f"{val:.2f}"
            else:
                val_str = "N/A"
            row_parts.append(val_str)

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_all_results_tables(results: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generate separate LaTeX tables for each evaluator.

    Returns dict mapping evaluator name (sanitized for filename) to table content.
    """
    if not results:
        return {}

    # Group by evaluator
    by_evaluator: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        eval_model = result['evaluator_model']
        if eval_model not in by_evaluator:
            by_evaluator[eval_model] = []
        by_evaluator[eval_model].append(result)

    # Generate tables for each evaluator
    tables = {}
    for evaluator in sorted(by_evaluator.keys(), key=model_sort_key):
        eval_name_display = clean_model_name(evaluator, format="long")
        eval_name_short = clean_model_name(evaluator, format="short")
        caption = f"Safety and helpfulness scores on the test set with {eval_name_display} as evaluator."
        label = f"tab:results-{eval_name_short}"
        table = generate_results_table(by_evaluator[evaluator], caption, label)
        tables[eval_name_short] = table

    return tables


# =============================================================================
# Training Metrics Table (Start Loss/Acc from training logs)
# =============================================================================


def generate_training_metrics_table(
    training_data: List[Dict[str, Any]],
) -> str:
    """Generate LaTeX table of training metrics aggregated by training config.

    Averages Start Loss and Start Acc over source models to show one row per training config.
    """
    if not training_data:
        return "% No training data to display"

    # Group by training_stages
    groups = defaultdict(list)
    for d in training_data:
        key = tuple(d.get('training_stages', []))
        groups[key].append(d)

    # Aggregate each group
    aggregated = []
    for stages_tuple, items in groups.items():
        loss_values = [item['start_loss'] for item in items]
        acc_values = [item['start_acc'] for item in items]

        aggregated.append({
            'training_stages': list(stages_tuple),
            'start_loss': statistics.mean(loss_values),
            'start_loss_std': statistics.stdev(loss_values) if len(loss_values) > 1 else 0.0,
            'start_acc': statistics.mean(acc_values),
            'start_acc_std': statistics.stdev(acc_values) if len(acc_values) > 1 else 0.0,
            'n_models': len(items),
        })

    # Sort by training config
    aggregated = sorted(aggregated, key=lambda x: training_sort_key(x, include_model=False))

    # Build LaTeX table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Starting loss and accuracy for DPO training runs, averaged over source models.}",
        r"\label{tab:training-metrics}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Training Config} & \textbf{Start Loss} & \textbf{Start Acc} & \textbf{n} \\",
        r"\midrule",
    ]

    for row in aggregated:
        config_str = format_label_latex(row)

        loss_val = row.get('start_loss')
        acc_val = row.get('start_acc')
        loss_std = row.get('start_loss_std', 0)
        acc_std = row.get('start_acc_std', 0)
        n_models = row.get('n_models', 0)

        # Format with std dev if available
        if loss_std > 0:
            loss_str = f"{loss_val:.3f} $\\pm$ {loss_std:.3f}"
        else:
            loss_str = f"{loss_val:.3f}"

        if acc_std > 0:
            acc_str = f"{acc_val:.3f} $\\pm$ {acc_std:.3f}"
        else:
            acc_str = f"{acc_val:.3f}"

        lines.append(f"{config_str} & {loss_str} & {acc_str} & {n_models} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Persistence Table
# =============================================================================

def load_persistence_stats(results_dir: str) -> Dict[str, Any]:
    """Load persistence_stats.json from results directory if it exists."""
    persistence_path = Path(results_dir) / 'persistence_stats.json'
    if persistence_path.exists():
        with open(persistence_path, 'r') as f:
            return json.load(f)
    return {}


def convert_persistence_stats_to_table_format(
    persistence_stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert persistence_stats.json format to table format for generate_persistence_table.

    Returns data in format expected by generate_persistence_table, adding CI columns.
    Uses persistence section with by_model (averaged over evaluators).
    """
    results = []

    # Use persistence section for the main table
    persistence = persistence_stats.get('persistence', {})
    by_beta = persistence.get('by_beta', {})

    for beta_str in sorted(by_beta.keys(), key=float):
        beta = float(beta_str)
        beta_data = by_beta[beta_str]
        by_model = beta_data.get('by_model', {})

        # Per-model rows (keys are clean model names like "Llama-8B")
        for source_model in sorted(by_model.keys(), key=lambda m: model_sort_key(m) if m else ()):
            model_data = by_model[source_model]
            results.append({
                'beta': beta,
                'source_model': source_model,
                'persist_s': model_data.get('persist_s'),
                'persist_s_ci_lower': model_data.get('persist_s_ci_lower'),
                'persist_s_ci_upper': model_data.get('persist_s_ci_upper'),
                'persist_h': model_data.get('persist_h'),
                'persist_h_ci_lower': model_data.get('persist_h_ci_lower'),
                'persist_h_ci_upper': model_data.get('persist_h_ci_upper'),
            })

        # Average row
        avg_data = beta_data.get('average', {})
        results.append({
            'beta': beta,
            'source_model': None,  # None indicates average row
            'persist_s': avg_data.get('persist_s'),
            'persist_s_ci_lower': avg_data.get('persist_s_ci_lower'),
            'persist_s_ci_upper': avg_data.get('persist_s_ci_upper'),
            'persist_h': avg_data.get('persist_h'),
            'persist_h_ci_lower': avg_data.get('persist_h_ci_lower'),
            'persist_h_ci_upper': avg_data.get('persist_h_ci_upper'),
        })

    return results


def convert_per_evaluator_persistence_stats(
    persistence_stats: Dict[str, Any],
    evaluator: str,
) -> List[Dict[str, Any]]:
    """Convert persistence_stats.json format to table format for a specific evaluator.

    Extracts (model, evaluator) pairs from by_model_and_evaluator that match the evaluator.
    """
    results = []

    persistence = persistence_stats.get('persistence', {})
    by_beta = persistence.get('by_beta', {})

    for beta_str in sorted(by_beta.keys(), key=float):
        beta = float(beta_str)
        beta_data = by_beta[beta_str]
        by_model_and_evaluator = beta_data.get('by_model_and_evaluator', {})

        # Extract models for this evaluator from by_model_and_evaluator keys
        # Keys are formatted as "evaluator|model"
        models_for_evaluator = {}
        for key, data in by_model_and_evaluator.items():
            if '|' in key:
                eval_part, model_part = key.split('|', 1)
                if eval_part == evaluator:
                    models_for_evaluator[model_part] = data

        # Per-model rows
        for model_key in sorted(models_for_evaluator.keys(), key=model_seed_sort_key):
            model_data = models_for_evaluator[model_key]
            results.append({
                'beta': beta,
                'source_model': model_key,
                'persist_s': model_data.get('persist_s'),
                'persist_s_ci_lower': model_data.get('persist_s_ci_lower'),
                'persist_s_ci_upper': model_data.get('persist_s_ci_upper'),
                'persist_h': model_data.get('persist_h'),
                'persist_h_ci_lower': model_data.get('persist_h_ci_lower'),
                'persist_h_ci_upper': model_data.get('persist_h_ci_upper'),
            })

        # Average row (from by_evaluator section)
        by_evaluator_data = beta_data.get('by_evaluator', {})
        avg_data = by_evaluator_data.get(evaluator, {})
        results.append({
            'beta': beta,
            'source_model': None,  # None indicates average row
            'persist_s': avg_data.get('persist_s'),
            'persist_s_ci_lower': avg_data.get('persist_s_ci_lower'),
            'persist_s_ci_upper': avg_data.get('persist_s_ci_upper'),
            'persist_h': avg_data.get('persist_h'),
            'persist_h_ci_lower': avg_data.get('persist_h_ci_lower'),
            'persist_h_ci_upper': avg_data.get('persist_h_ci_upper'),
        })

    return results


def format_persistence_value(
    value: float,
    ci_lower: float = None,
    ci_upper: float = None,
) -> str:
    """Format persistence value, optionally with CI in bracket notation."""
    if value is None:
        return "N/A"
    if ci_lower is not None and ci_upper is not None:
        return f"{value:.2f} \ci{{({ci_lower:.2f}, {ci_upper:.2f})}}"
    return f"{value:.2f}"


def format_model_name_for_table(source_model: str) -> str:
    """Format model name for display in persistence table.

    Strips seed suffix if present (e.g., "Llama-8B_s42" -> "Llama 8B").
    """
    if source_model is None:
        return "Average"

    base_model, _ = parse_model_seed_key(source_model)
    return clean_model_name(base_model, format="long")


def generate_persistence_table(
    persistence_data: List[Dict[str, Any]],
    caption: str = "Persistence of each metric by source model.",
    label: str = "tab:persistence",
    show_ci: bool = True,
) -> str:
    """Generate LaTeX table for persistence metrics by beta and source model.

    Args:
        persistence_data: List of dicts with beta, source_model, persist_s, persist_h,
                          and optionally persist_s_ci_lower, persist_s_ci_upper, etc.
        caption: Table caption.
        label: LaTeX label.
        show_ci: If True and CI data is present, include CIs in bracket notation.
    """
    if not persistence_data:
        return "% No persistence data to display"

    # Check if any row has CI data
    has_ci = any(
        row.get('persist_s_ci_lower') is not None or row.get('persist_h_ci_lower') is not None
        for row in persistence_data
    )
    include_ci = show_ci and has_ci

    lines = [
        r"\begin{table}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\centering",
        r"\begin{tabular}{clcc}",
        r"\toprule",
        r"$\beta$ & Source Model & $\per($S$,\beta)$ & $\per($H$,\beta)$ \\",
        r"\midrule",
    ]

    for row in persistence_data:
        beta = row['beta']
        source_model = row.get('source_model')
        persist_s = row.get('persist_s')
        persist_h = row.get('persist_h')

        # Format source model name (handles seed suffix)
        model_str = format_model_name_for_table(source_model)

        if include_ci:
            ps_str = format_persistence_value(
                persist_s,
                row.get('persist_s_ci_lower'),
                row.get('persist_s_ci_upper'),
            )
            ph_str = format_persistence_value(
                persist_h,
                row.get('persist_h_ci_lower'),
                row.get('persist_h_ci_upper'),
            )
        else:
            ps_str = f"{persist_s:.2f}" if persist_s is not None else "N/A"
            ph_str = f"{persist_h:.2f}" if persist_h is not None else "N/A"

        lines.append(f"{beta} & {model_str} & {ps_str} & {ph_str} \\\\")

        # Add separator after Average row (except for the last one)
        if source_model is None:
            lines.append(r"\midrule")

    # Remove the trailing \midrule if present
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def generate_tables(args) -> None:
    # Determine which tables to generate based on flags
    # If any *-only flag is set, only generate that table
    only_flags = [args.hyperparameters_only, args.results_only, args.training_metrics_only,
                  args.aggregated_results_only, args.aggregated_deltas_only, args.persistence_only]
    any_only = any(only_flags)

    gen_hyperparameters = args.hyperparameters_only if any_only else True
    gen_results = args.results_only if any_only else True
    gen_training_metrics = args.training_metrics_only if any_only else True
    gen_aggregated_results = args.aggregated_results_only if any_only else True
    gen_aggregated_deltas = args.aggregated_deltas_only if any_only else True
    gen_persistence = args.persistence_only if any_only else True

    # Results table requires results_dir
    if gen_results and args.results_dir is None:
        print("Warning: results_dir is required for generating results table, skipping")
        gen_results = False

    # Aggregated results table requires results_dir
    if gen_aggregated_results and args.results_dir is None:
        print("Warning: results_dir is required for generating aggregated results table, skipping")
        gen_aggregated_results = False

    # Aggregated deltas table requires results_dir
    if gen_aggregated_deltas and args.results_dir is None:
        print("Warning: results_dir is required for generating aggregated deltas table, skipping")
        gen_aggregated_deltas = False

    # Persistence table requires results_dir
    if gen_persistence and args.results_dir is None:
        print("Warning: results_dir is required for generating persistence table, skipping")
        gen_persistence = False

    # Training metrics table requires trained_models_dir
    if gen_training_metrics and args.trained_models_dir is None:
        print("Warning: --trained-models-dir is required for generating training metrics table, skipping")
        gen_training_metrics = False

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results data once if any results-based tables are requested
    results = None
    if gen_results or gen_aggregated_results or gen_aggregated_deltas or gen_persistence:
        print(f"Loading results from {args.results_dir}...")
        results = load_results_data(args.results_dir, data_dir=args.data_dir)
        if not results:
            print("No results found")
            return 1
        print(f"Found {len(results)} configurations")

    # Generate hyperparameters table
    if gen_hyperparameters:
        hyperparameters_latex = generate_hyperparameter_table()

        if args.print_only:
            print("=" * 60)
            print("HYPERPARAMETERS TABLE")
            print("=" * 60)
            print(hyperparameters_latex)
            print()
        else:
            save_table(hyperparameters_latex, 'hyperparams.tex', args.output_dir, 'hyperparameters')

    # Generate results table (per source model / evaluator)
    if gen_results:
        # Aggregate by source model, evaluator, training stages, and quantization (averaging over simulators)
        aggregated = aggregate_scores(
            results,
            group_by=['source_model', 'evaluator_model', 'training_stages', 'quantization']
        )
        print(f"Aggregated to {len(aggregated)} rows (agent-evaluator pairs)")

        # Generate LaTeX tables (one per evaluator)
        results_by_evaluator = generate_all_results_tables(aggregated)

        if args.print_only:
            for evaluator, table in results_by_evaluator.items():
                print("=" * 60)
                print(f"RESULTS TABLE ({evaluator})")
                print("=" * 60)
                print(table)
                print()
        else:
            for evaluator, table in results_by_evaluator.items():
                save_table(table, f'{evaluator}_results.tex', args.output_dir, f'results ({evaluator})')

    # Generate aggregated results table (by training stages, averaging over models and evaluators)
    if gen_aggregated_results:
        # Aggregate by training stages only (averaging over models and evaluators)
        aggregated = aggregate_scores(results, group_by=['training_stages'])
        print(f"Aggregated to {len(aggregated)} training configurations")

        caption = "Safety and helpfulness scores aggregated by training configuration, averaged over all source models and evaluators."
        aggregated_latex = generate_results_table(aggregated, caption, label="tab:aggregated-results")

        if args.print_only:
            print("=" * 60)
            print("AGGREGATED RESULTS TABLE")
            print("=" * 60)
            print(aggregated_latex)
        else:
            save_table(aggregated_latex, 'aggregated_results_table.tex', args.output_dir, 'aggregated results')

    # Generate aggregated deltas table (change from source model)
    if gen_aggregated_deltas:
        # Aggregate by training stages only, then compute deltas from base
        aggregated = aggregate_scores(results, group_by=['training_stages'])
        deltas = compute_deltas_from_aggregated(aggregated, group_by=['training_stages'])
        print(f"Computed deltas for {len(deltas)} training configurations")

        caption = "Change in safety and helpfulness from source model, aggregated by training configuration."
        deltas_latex = generate_results_table(
            deltas, caption, label="tab:aggregated-deltas",
            columns=DELTA_COLUMNS, show_sign=True
        )

        if args.print_only:
            print("=" * 60)
            print("AGGREGATED DELTAS TABLE")
            print("=" * 60)
            print(deltas_latex)
        else:
            save_table(deltas_latex, 'aggregated_deltas_table.tex', args.output_dir, 'aggregated deltas')

    # Generate persistence table
    if gen_persistence:
        # Load pre-computed persistence stats with CIs (required)
        persistence_stats = load_persistence_stats(args.results_dir) if args.results_dir else {}
        if not persistence_stats:
            print("Error: persistence_stats.json not found. Run partition_results_by_split.py first.")
            return 1

        print("Using persistence_stats.json with bootstrap CIs")

        # Get unique evaluators from the stats (inside persistence/by_beta/*/by_evaluator)
        persistence = persistence_stats.get('persistence', {})
        by_beta = persistence.get('by_beta', {})
        evaluators = set()
        for beta_data in by_beta.values():
            evaluators.update(beta_data.get('by_evaluator', {}).keys())
        evaluators = sorted(evaluators)

        # Per-evaluator persistence tables
        for evaluator in evaluators:
            eval_suffix = get_evaluator_short_name(evaluator)
            persistence_data = convert_per_evaluator_persistence_stats(persistence_stats, evaluator)
            if persistence_data:
                eval_name_display = clean_model_name(evaluator, format="long")
                caption = f"Persistence of each metric by source model ({eval_name_display} evaluator)."
                label = f"tab:persistence-{eval_suffix}"
                persistence_latex = generate_persistence_table(persistence_data, caption=caption, label=label)

                if args.print_only:
                    print("=" * 60)
                    print(f"PERSISTENCE TABLE ({eval_suffix})")
                    print("=" * 60)
                    print(persistence_latex)
                else:
                    save_table(persistence_latex, f'persist_{eval_suffix}.tex', args.output_dir, f'persistence ({eval_suffix})')

        # Averaged persistence table (over evaluators)
        persistence_data = convert_persistence_stats_to_table_format(persistence_stats)
        print(f"Loaded persistence for {len(persistence_data)} (model, beta) combinations")

        persistence_latex = generate_persistence_table(persistence_data)

        if args.print_only:
            print("=" * 60)
            print("PERSISTENCE TABLE (averaged)")
            print("=" * 60)
            print(persistence_latex)
        else:
            save_table(persistence_latex, 'persist.tex', args.output_dir, 'persistence (averaged)')

    # Generate training metrics table (Start Loss/Acc)
    if gen_training_metrics:
        print(f"Loading training metrics from {args.trained_models_dir}...")
        training_data = load_training_metrics(args.trained_models_dir, data_dir=args.data_dir)

        if not training_data:
            print("No training data found")
            return 1

        print(f"Found {len(training_data)} trained models")

        training_metrics_latex = generate_training_metrics_table(training_data)

        if args.print_only:
            print("=" * 60)
            print("TRAINING METRICS TABLE")
            print("=" * 60)
            print(training_metrics_latex)
        else:
            save_table(training_metrics_latex, 'training_metrics.tex', args.output_dir, 'training metrics')

    if not args.print_only:
        print(f"\nTables saved to {args.output_dir}/")

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from ToolEmu evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'results_dir',
        type=str,
        nargs='?',
        default=None,
        help='Directory containing unified report JSON files (required for results table)'
    )
    parser.add_argument(
        '-t', '--trained-models-dir',
        type=str,
        default=None,
        help='Directory containing trained model subdirectories (required for training metrics table)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='figs',
        help='Output directory for LaTeX files (default: figs)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/dpo_data',
        help='Directory containing DPO dataset files for inferring dataset types (default: data/dpo_data)'
    )
    parser.add_argument(
        '--hyperparameters-only',
        action='store_true',
        help='Only generate hyperparameters table'
    )
    parser.add_argument(
        '--results-only',
        action='store_true',
        help='Only generate results table'
    )
    parser.add_argument(
        '--training-metrics-only',
        action='store_true',
        help='Only generate training metrics table (Start Loss/Acc)'
    )
    parser.add_argument(
        '--aggregated-results-only',
        action='store_true',
        help='Only generate aggregated results table (by training stages)'
    )
    parser.add_argument(
        '--aggregated-deltas-only',
        action='store_true',
        help='Only generate aggregated deltas table (change from source model)'
    )
    parser.add_argument(
        '--persistence-only',
        action='store_true',
        help='Only generate persistence table (training objective durability by beta)'
    )
    parser.add_argument(
        '--print-only',
        action='store_true',
        help='Print tables to stdout instead of writing to files'
    )

    args = parser.parse_args()
    generate_tables(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
