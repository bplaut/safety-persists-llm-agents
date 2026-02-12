#!/usr/bin/env python3
"""
Rename files (and optionally directories) by replacing a target string with a replacement string.

Usage:
    python rename_files.py <directory> <target_string> <replace_string> [--dry-run] [--recursive] [--include-dirs]

Example:
    python rename_files.py output/trajectories "1A6000" "" --dry-run
    python rename_files.py output/trajectories "dpo_merged" "dpo" --include-dirs --recursive
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple


def find_items_to_rename(
    directory: Path,
    target_string: str,
    replace_string: str,
    recursive: bool,
    include_dirs: bool
) -> List[Tuple[Path, str, str]]:
    """
    Find all files (and optionally directories) containing the target string in their name.
    Returns list of tuples: (path, old_name, new_name). Directories are sorted deepest-first
    to avoid path issues during renaming.
    """
    file_matches = []
    dir_matches = []

    if recursive:
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            for filename in files:
                if target_string in filename:
                    old_path = root_path / filename
                    new_filename = filename.replace(target_string, replace_string)
                    file_matches.append((old_path, filename, new_filename))
            if include_dirs:
                for dirname in dirs:
                    if target_string in dirname:
                        old_path = root_path / dirname
                        new_dirname = dirname.replace(target_string, replace_string)
                        dir_matches.append((old_path, dirname, new_dirname))
    else:
        for item in directory.iterdir():
            if target_string in item.name:
                new_name = item.name.replace(target_string, replace_string)
                if item.is_file():
                    file_matches.append((item, item.name, new_name))
                elif item.is_dir() and include_dirs:
                    dir_matches.append((item, item.name, new_name))

    # Sort directories by depth (deepest first) to avoid path issues when renaming
    dir_matches.sort(key=lambda x: len(x[0].parts), reverse=True)

    # Rename files first, then directories (deepest first)
    return file_matches + dir_matches


def rename_items(
    directory: Path,
    target_string: str,
    replace_string: str,
    dry_run: bool = True,
    recursive: bool = False,
    include_dirs: bool = False
) -> None:
    """Rename files (and optionally directories) by replacing target_string with replace_string."""
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    item_type = "item" if include_dirs else "file"

    # Find all items to rename
    items_to_rename = find_items_to_rename(directory, target_string, replace_string, recursive, include_dirs)

    if not items_to_rename:
        print(f"No {item_type}s found containing '{target_string}' in their name.")
        return

    print(f"Found {len(items_to_rename)} {item_type}(s) to rename:")
    print()

    # Display what will be renamed
    for old_path, old_name, new_name in items_to_rename:
        parent = old_path.parent
        suffix = "/" if old_path.is_dir() else ""
        print(f"  {parent}/")
        print(f"    {old_name}{suffix}")
        print(f"    -> {new_name}{suffix}")
        print()

    if dry_run:
        print(f"DRY RUN: No {item_type}s were actually renamed.")
        print("Run without --dry-run to perform the renaming.")
        return

    # Confirm before renaming
    response = input(f"Rename {len(items_to_rename)} {item_type}(s)? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Perform the renaming
    renamed_count = 0
    for old_path, old_name, new_name in items_to_rename:
        new_path = old_path.parent / new_name

        # Check if target already exists
        if new_path.exists():
            print(f"ERROR: Target already exists: {new_path}")
            print(f"       Skipping: {old_path}")
            continue

        try:
            old_path.rename(new_path)
            renamed_count += 1
            print(f"Renamed: {old_name} -> {new_name}")
        except Exception as e:
            print(f"ERROR renaming {old_path}: {e}")

    print()
    print(f"Successfully renamed {renamed_count} {item_type}(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Rename files (and optionally directories) by replacing a target string with a replacement string.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be renamed
  python rename_files.py output/trajectories "_1A6000" "" --dry-run

  # Actually rename the files
  python rename_files.py output/trajectories "_1A6000" ""

  # Recursively rename in all subdirectories
  python rename_files.py output/trajectories "_1A6000" "" --recursive

  # Rename directories as well as files
  python rename_files.py output/ "dpo_merged" "dpo" --include-dirs --recursive
        """
    )

    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing files to rename"
    )

    parser.add_argument(
        "target_string",
        type=str,
        help="String to find in filenames"
    )

    parser.add_argument(
        "replace_string",
        type=str,
        help="String to replace with (use empty string '' to remove)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming"
    )

    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search subdirectories recursively"
    )

    parser.add_argument(
        "--include-dirs",
        "-d",
        action="store_true",
        help="Also rename directories (not just files)"
    )

    args = parser.parse_args()

    directory = Path(args.directory)

    rename_items(
        directory=directory,
        target_string=args.target_string,
        replace_string=args.replace_string,
        dry_run=args.dry_run,
        recursive=args.recursive,
        include_dirs=args.include_dirs
    )


if __name__ == "__main__":
    main()
