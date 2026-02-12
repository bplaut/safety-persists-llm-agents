#!/usr/bin/env python3
"""Extract starting loss and accuracy from DPO training SLURM logs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.analysis_utils import calc_col_width, get_latest_slurm_log, extract_first_metrics


def main():
    parser = argparse.ArgumentParser(description="Extract starting loss and accuracy from DPO training SLURM logs.")
    parser.add_argument("input_dir", type=Path, help="Directory containing trained model subdirectories")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    results = []

    for model_dir in sorted(args.input_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        slurm_dir = model_dir / "slurm_output"
        if not slurm_dir.exists():
            results.append((model_dir.name, None, None, "No slurm_output/"))
            continue

        log_path = get_latest_slurm_log(slurm_dir)
        if not log_path:
            results.append((model_dir.name, None, None, "No logs found"))
            continue

        loss, accuracy = extract_first_metrics(log_path)
        results.append((model_dir.name, loss, accuracy, None))

    # Format values for display
    def fmt_loss(loss):
        return f"{loss:.4f}" if loss is not None else "N/A"
    def fmt_acc(acc):
        return f"{acc:.4f}" if acc is not None else "N/A"

    # Calculate dynamic column widths
    col_widths = {
        'model': calc_col_width('Model Name', [r[0] for r in results]),
        'loss': calc_col_width('Start Loss', [fmt_loss(r[1]) for r in results]),
        'acc': calc_col_width('Start Acc', [fmt_acc(r[2]) for r in results]),
    }

    # Print table
    header = f"{'Model Name':<{col_widths['model']}} | {'Start Loss':>{col_widths['loss']}} | {'Start Acc':>{col_widths['acc']}}"
    print(header)
    print("-" * len(header))

    for model_name, loss, accuracy, error in results:
        loss_str = fmt_loss(loss)
        acc_str = fmt_acc(accuracy)
        line = f"{model_name:<{col_widths['model']}} | {loss_str:>{col_widths['loss']}} | {acc_str:>{col_widths['acc']}}"
        if error:
            line += f"  ({error})"
        print(line)

    print(f"\nTotal models: {len(results)}")
    valid = sum(1 for _, l, a, _ in results if l is not None)
    print(f"Models with valid metrics: {valid}")


if __name__ == "__main__":
    main()
