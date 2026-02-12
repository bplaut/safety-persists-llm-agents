#!/usr/bin/env python3
"""Delete evaluation files (*_eval_*.jsonl) from a directory while preserving trajectory files."""

import argparse
import os
import sys
from pathlib import Path


def find_eval_files(directory: Path) -> list[Path]:
    """Find all eval files (containing _eval_ in filename) in directory."""
    eval_files = []
    for file in directory.rglob("*_eval_*.jsonl"):
        if file.is_file():
            eval_files.append(file)
    return sorted(eval_files)


def main():
    parser = argparse.ArgumentParser(
        description="Delete evaluation files (*_eval_*.jsonl) but preserve trajectory files"
    )
    parser.add_argument("directory", type=Path, help="Directory to search for eval files")
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    eval_files = find_eval_files(args.directory)

    if not eval_files:
        print(f"No eval files found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(eval_files)} eval files:")
    for f in eval_files:
        print(f"  {f}")
    print()

    if args.dry_run:
        print("[DRY RUN] No files were deleted.")
        sys.exit(0)

    confirm = input("Are you sure you want to delete these files? (y/N) ")
    if confirm.lower() != "y":
        print("Aborted.")
        sys.exit(0)

    deleted = 0
    for f in eval_files:
        os.remove(f)
        print(f"Deleted: {f}")
        deleted += 1

    print(f"\nDone. Deleted {deleted} eval files.")


if __name__ == "__main__":
    main()
