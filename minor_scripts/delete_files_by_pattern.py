#!/usr/bin/env python3
"""Delete files recursively that contain a specified string in their filename."""

import argparse
import os
import sys
from pathlib import Path


def find_matching_files(directory: Path, pattern: str) -> list[Path]:
    """Find all files containing the pattern string in their filename."""
    matching_files = []
    for file in directory.rglob("*"):
        if file.is_file() and pattern in file.name:
            matching_files.append(file)
    return sorted(matching_files)


def main():
    parser = argparse.ArgumentParser(
        description="Delete files recursively that contain a specified string in their filename"
    )
    parser.add_argument("directory", type=Path, help="Directory to search recursively")
    parser.add_argument("pattern", type=str, help="String pattern to match in filenames")
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    if not args.pattern:
        print("Error: pattern cannot be empty", file=sys.stderr)
        sys.exit(1)

    matching_files = find_matching_files(args.directory, args.pattern)

    if not matching_files:
        print(f"No files matching '{args.pattern}' found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(matching_files)} files matching '{args.pattern}':")
    for f in matching_files:
        print(f"  {f}")
    print()

    if args.dry_run:
        print("[DRY RUN] No files were deleted.")
        sys.exit(0)

    confirm = input(f"Are you sure you want to delete these {len(matching_files)} files? (y/N) ")
    if confirm.lower() != "y":
        print("Aborted.")
        sys.exit(0)

    deleted = 0
    for f in matching_files:
        os.remove(f)
        print(f"Deleted: {f}")
        deleted += 1

    print(f"\nDone. Deleted {deleted} files.")


if __name__ == "__main__":
    main()
