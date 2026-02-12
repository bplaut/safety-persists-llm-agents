#!/usr/bin/env python3
"""Transform multi-replicate eval files into single-replicate format by randomly sampling.

For each case in a multi-replicate eval file, randomly chooses one non-null (non-error)
replicate and converts to single-replicate format.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Any

# Add src to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.toolemu_utils import load_jsonl, write_jsonl, ToolEmuFilePaths


def find_eval_files(directory: Path) -> list[Path]:
    """Find all eval files (*_eval_*.jsonl) recursively in directory."""
    return sorted(directory.rglob("*_eval_*.jsonl"))


def is_multi_replicate(data: list[dict[str, Any]]) -> bool:
    """Check if eval data is in multi-replicate format (has 'replicates' key)."""
    if not data:
        return False
    return "replicates" in data[0]


def get_score_key_from_filepath(filepath: Path) -> str:
    """Determine the expected score key based on eval file type."""
    eval_type = ToolEmuFilePaths.extract_eval_type(str(filepath))
    if eval_type is None:
        raise ValueError(f"Cannot determine eval type from filename: {filepath.name}")
    return "ToolCallRisk" if eval_type == "agent_safe" else "Helpfulness"


def replicate_has_valid_score(replicate: dict[str, Any], score_key: str) -> bool:
    """Check if a replicate has a valid (non-null) score for the expected metric."""
    if "error" in replicate:
        return False
    eval_scores = replicate.get("eval_scores")
    if not eval_scores or not isinstance(eval_scores, dict):
        return False
    score = eval_scores.get(score_key)
    # Check score is a valid number in expected range [0, 3]
    if score is None:
        return False
    if not isinstance(score, (int, float)):
        return False
    if score < 0 or score > 3:
        return False
    return True


def get_valid_replicates(entry: dict[str, Any], score_key: str) -> list[dict[str, Any]]:
    """Get replicates with valid (non-null) scores for the expected metric."""
    replicates = entry.get("replicates", [])
    return [r for r in replicates if replicate_has_valid_score(r, score_key)]


def convert_to_single_replicate(
    entry: dict[str, Any], score_key: str, rng: random.Random
) -> dict[str, Any]:
    """Convert a multi-replicate entry to single-replicate format by randomly sampling.

    Returns entry unchanged if not multi-replicate or no valid replicates found.
    """
    if "replicates" not in entry:
        return entry

    valid_replicates = get_valid_replicates(entry, score_key)
    if not valid_replicates:
        # No valid replicates - keep eval_id but mark as error
        return {
            "eval_id": entry.get("eval_id"),
            "error": "No valid replicates to sample from",
            "original_replicates": len(entry.get("replicates", [])),
        }

    # Randomly choose one valid replicate
    chosen = rng.choice(valid_replicates)

    # Build single-replicate format: copy chosen fields, then ensure correct eval_id
    result = dict(chosen)
    result["eval_id"] = entry.get("eval_id")
    return result


def process_eval_file(filepath: Path, rng: random.Random, dry_run: bool) -> tuple[bool, int, int]:
    """Process a single eval file. Returns (was_converted, num_entries, num_no_valid)."""
    data = load_jsonl(str(filepath), description=f"eval file {filepath.name}")

    if not is_multi_replicate(data):
        return False, len(data), 0

    score_key = get_score_key_from_filepath(filepath)

    # Convert all entries
    converted = [convert_to_single_replicate(entry, score_key, rng) for entry in data]

    # Count entries with no valid replicates
    num_no_valid = sum(1 for c in converted if "error" in c and "No valid replicates" in str(c.get("error", "")))

    if not dry_run:
        write_jsonl(converted, str(filepath))

    return True, len(data), num_no_valid


def main():
    parser = argparse.ArgumentParser(
        description="Transform multi-replicate eval files into single-replicate format"
    )
    parser.add_argument("directory", type=Path, help="Directory to search for eval files")
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be converted without actually converting"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    eval_files = find_eval_files(args.directory)
    if not eval_files:
        print(f"No eval files found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(eval_files)} eval files")
    print(f"Random seed: {args.seed}")
    if args.dry_run:
        print("[DRY RUN] No files will be modified.\n")

    converted_count = 0
    skipped_count = 0
    total_no_valid = 0

    for filepath in eval_files:
        try:
            was_converted, num_entries, num_no_valid = process_eval_file(filepath, rng, args.dry_run)
            if was_converted:
                converted_count += 1
                total_no_valid += num_no_valid
                action = "Would convert" if args.dry_run else "Converted"
                warning = f" ({num_no_valid} entries with no valid replicates)" if num_no_valid > 0 else ""
                print(f"  {action}: {filepath} ({num_entries} entries){warning}")
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  Error processing {filepath}: {e}", file=sys.stderr)

    print(f"\nSummary:")
    print(f"  Converted: {converted_count}")
    print(f"  Skipped (already single-replicate): {skipped_count}")
    if total_no_valid > 0:
        print(f"  Entries with no valid replicates: {total_no_valid}")


if __name__ == "__main__":
    main()
