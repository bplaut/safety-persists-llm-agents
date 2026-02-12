#!/usr/bin/env python3
"""
Remove error records from ToolEmu trajectory and evaluation files.

Scans output/trajectories for JSONL files and removes any lines that contain
error records (e.g., {"error": "CUDA out of memory..."}) while preserving
valid trajectory and evaluation data.

Usage:
    python minor_scripts/clean_error_records.py              # Dry run (default)
    python minor_scripts/clean_error_records.py --apply      # Actually modify files
    python minor_scripts/clean_error_records.py --delete     # Delete files entirely instead of cleaning
"""

import argparse
import json
import sys
from pathlib import Path


def find_error_records(filepath: Path) -> tuple[list[dict], list[dict]]:
    """Load a JSONL file and separate valid records from error records."""
    valid_records = []
    error_records = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'error' in data and isinstance(data.get('error'), str):
                    error_records.append({'line': line_num, 'data': data})
                else:
                    valid_records.append(data)
            except json.JSONDecodeError as e:
                # Treat malformed JSON as an error record
                error_records.append({
                    'line': line_num,
                    'data': {'error': f'JSON parse error: {e}', 'raw': line[:200]}
                })

    return valid_records, error_records


def categorize_error(error_msg: str) -> str:
    """Categorize an error message for summary reporting."""
    if 'CUDA out of memory' in error_msg:
        return 'CUDA OOM'
    elif 'APIError' in error_msg:
        return 'API Error'
    elif 'RateLimitError' in error_msg:
        return 'Rate Limit'
    elif 'JSON parse error' in error_msg:
        return 'JSON Parse Error'
    else:
        return 'Other'


def main():
    parser = argparse.ArgumentParser(
        description='Remove error records from ToolEmu output files'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually modify files (default is dry run)'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete files with errors entirely instead of removing just error lines'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output/trajectories'),
        help='Directory to scan for JSONL files (default: output/trajectories)'
    )
    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: Directory {args.output_dir} does not exist")
        sys.exit(1)

    # Find all JSONL files with error records
    files_with_errors = []
    error_categories = {}

    for filepath in sorted(args.output_dir.rglob('*.jsonl')):
        valid_records, error_records = find_error_records(filepath)

        if error_records:
            files_with_errors.append({
                'path': filepath,
                'valid_count': len(valid_records),
                'error_count': len(error_records),
                'valid_records': valid_records,
                'error_records': error_records,
            })

            # Categorize errors for summary
            for err in error_records:
                error_msg = err['data'].get('error', '')
                category = categorize_error(error_msg)
                error_categories[category] = error_categories.get(category, 0) + 1

    if not files_with_errors:
        print("No files with error records found.")
        return

    # Print summary
    print(f"Found {len(files_with_errors)} files with error records:\n")

    total_errors = 0
    total_valid = 0
    for f in files_with_errors:
        rel_path = f['path'].relative_to(args.output_dir)
        print(f"  {rel_path}")
        print(f"    Valid records: {f['valid_count']}, Error records: {f['error_count']}")
        for err in f['error_records']:
            error_preview = err['data'].get('error', '')[:80]
            print(f"    Line {err['line']}: {error_preview}...")
        total_errors += f['error_count']
        total_valid += f['valid_count']

    print(f"\nError categories:")
    for category, count in sorted(error_categories.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")

    print(f"\nSummary: {total_errors} error records across {len(files_with_errors)} files")
    print(f"         {total_valid} valid records will be preserved")

    if not args.apply:
        print("\n[DRY RUN] No files modified. Use --apply to actually modify files.")
        if args.delete:
            print("          With --delete, files would be deleted entirely.")
        else:
            print("          Error lines will be removed, valid content preserved.")
        return

    # Apply changes
    print(f"\nApplying changes...")

    for f in files_with_errors:
        filepath = f['path']

        if args.delete:
            filepath.unlink()
            print(f"  Deleted: {filepath.relative_to(args.output_dir)}")
        else:
            # Rewrite file with only valid records
            with open(filepath, 'w') as fp:
                for record in f['valid_records']:
                    fp.write(json.dumps(record) + '\n')
            print(f"  Cleaned: {filepath.relative_to(args.output_dir)} "
                  f"(removed {f['error_count']} error lines)")

    action = "deleted" if args.delete else "cleaned"
    print(f"\nDone. {len(files_with_errors)} files {action}.")


if __name__ == '__main__':
    main()
