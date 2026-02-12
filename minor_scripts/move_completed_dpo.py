#!/usr/bin/env python3
"""Move dpo_output folders that have a 'final' subdirectory to a target directory."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", "-s", default="dpo_output",
                        help="Source directory containing training outputs (default: dpo_output)")
    parser.add_argument("--target-dir", "-t", required=True,
                        help="Target directory to move completed training runs to")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    if not source_dir.exists():
        raise ValueError(f"Source directory '{source_dir}' does not exist")

    # Find directories with 'final' subdirectory
    dirs_to_move = []
    for subdir in sorted(source_dir.iterdir()):
        if subdir.is_dir() and (subdir / "final").is_dir():
            dirs_to_move.append(subdir)

    if not dirs_to_move:
        print(f"No directories with 'final' subdirectory found in {source_dir}")
        return

    # Print what will be moved
    print("=" * 60)
    print(f"The following {len(dirs_to_move)} directories will be moved to '{target_dir}/':")
    print("=" * 60)
    for d in dirs_to_move:
        print(f"  {d.name}")
    print("=" * 60)
    print()

    # Ask for confirmation
    response = input("Proceed with move? [y/N] ").strip().lower()
    if response not in ("y", "yes"):
        print("Aborted.")
        return

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Move each directory
    for d in dirs_to_move:
        dest = target_dir / d.name
        if dest.exists():
            print(f"WARNING: Skipping {d.name} - already exists in target")
            continue
        print(f"Moving {d.name}...")
        shutil.move(str(d), str(dest))

    print()
    print(f"Done! Moved {len(dirs_to_move)} directories to {target_dir}/")


if __name__ == "__main__":
    main()
