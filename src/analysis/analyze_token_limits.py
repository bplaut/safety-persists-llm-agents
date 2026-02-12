#!/usr/bin/env python3
"""Analyze how often each model hit the token limit in ToolEmu trajectories."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.analysis_utils import calc_col_width
from utils.model_name_utils import clean_model_name
from utils.toolemu_utils import ToolEmuFilePaths


def analyze_token_limits(trajectory_dir: Path) -> dict:
    """Analyze token limit hits per model."""
    stats = defaultdict(lambda: {"total": 0, "token_limit_hit": 0})

    for model_dir in sorted(trajectory_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        for traj_file in model_dir.iterdir():
            if not ToolEmuFilePaths.is_trajectory_file(traj_file):
                continue

            with open(traj_file) as f:
                for line in f:
                    d = json.loads(line)
                    output = d.get("output", "")
                    stats[model_name]["total"] += 1
                    if "Agent stopped: token limit exceeded" in output:
                        stats[model_name]["token_limit_hit"] += 1

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Analyze token limit hits per model")
    parser.add_argument(
        "--trajectory-dir", "-d",
        type=Path,
        default=Path("output/trajectories"),
        help="Directory containing trajectory files (default: output/trajectories)"
    )
    args = parser.parse_args()

    if not args.trajectory_dir.exists():
        raise ValueError(f"Directory not found: {args.trajectory_dir}")

    stats = analyze_token_limits(args.trajectory_dir)

    if not stats:
        print("No trajectory files found.")
        return

    # Sort by model name
    sorted_models = sorted(stats.keys())

    # Prepare display data
    display_names = {m: clean_model_name(m, format="short") for m in sorted_models}
    total_all = sum(s["total"] for s in stats.values())
    hit_all = sum(s["token_limit_hit"] for s in stats.values())

    # Calculate dynamic column widths
    model_w = calc_col_width("Model", list(display_names.values()) + ["TOTAL"])
    total_w = calc_col_width("Total", [stats[m]["total"] for m in sorted_models] + [total_all])
    hit_w = calc_col_width("Token Limit", [stats[m]["token_limit_hit"] for m in sorted_models] + [hit_all])

    # Print header
    header = f"{'Model':<{model_w}}  {'Total':>{total_w}}  {'Token Limit':>{hit_w}}  {'Rate':>7}"
    print(header)
    print("-" * len(header))

    # Print rows
    for model in sorted_models:
        total = stats[model]["total"]
        hit = stats[model]["token_limit_hit"]
        rate = hit / total * 100 if total > 0 else 0
        print(f"{display_names[model]:<{model_w}}  {total:>{total_w}}  {hit:>{hit_w}}  {rate:>6.1f}%")

    # Print summary
    rate_all = hit_all / total_all * 100 if total_all > 0 else 0
    print("-" * len(header))
    print(f"{'TOTAL':<{model_w}}  {total_all:>{total_w}}  {hit_all:>{hit_w}}  {rate_all:>6.1f}%")


if __name__ == "__main__":
    main()
