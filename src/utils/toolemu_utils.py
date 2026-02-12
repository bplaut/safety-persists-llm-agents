"""
ToolEmu file path utilities, data loading, and validation.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.model_name_utils import sanitize_model_name, clean_model_name


# Global constants
TOOLEMU_FULL_DATASET_SIZE = 144  # Total number of test cases in ToolEmu dataset
LOG_TAIL_BYTES = 50000  # Bytes to read from end of log files for progress monitoring


class IncompleteTrajectoryError(Exception):
    """
    Raised when a trajectory file should be skipped due to incompleteness.

    This includes:
    - Count mismatch: trajectory count doesn't match filename range
    - Missing required eval files

    Callers should catch this, log a warning, and skip the trajectory.
    """
    pass


class ToolEmuFilePaths:
    """Centralized utilities for ToolEmu file path conversions and constants."""

    # All expected eval types
    EVAL_TYPES = [
        'agent_safe',
        'agent_help',
        'agent_help_ignore_safety',
    ]

    # Mapping from metric names (used in scripts) to eval types
    METRIC_TO_EVAL_TYPE = {
        'safe': 'agent_safe',
        'help': 'agent_help',
        'help_ignore_safety': 'agent_help_ignore_safety',
    }

    # Suffixes to skip when identifying trajectory files
    SKIP_SUFFIXES = (
        '_eval_agent_help.jsonl',
        '_eval_agent_help_ignore_safety.jsonl',
        '_eval_agent_safe.jsonl',
        '_unified_report.json'
    )

    # Mapping from help safety level to filename suffix
    HELP_SAFETY_SUFFIX_MAP = {
        'normal': '',
        'ignore_safety': '_ignore_safety'
    }

    @staticmethod
    def trajectory_to_eval(traj_path: str, eval_type: str) -> str:
        """Convert trajectory path to eval path (e.g., "model.jsonl" -> "model_eval_agent_safe.jsonl")."""
        if not traj_path.endswith('.jsonl'):
            raise ValueError(f"Trajectory path must end with .jsonl: {traj_path}")
        return traj_path.replace('.jsonl', f'_eval_{eval_type}.jsonl')

    @staticmethod
    def eval_to_trajectory(eval_path: str) -> str:
        """Convert eval path to trajectory path by stripping the eval suffix."""
        return re.sub(r'_eval_agent_\w+\.jsonl$', '.jsonl', eval_path)

    @staticmethod
    def extract_eval_type(eval_path: str) -> Optional[str]:
        """Extract eval type from eval file path (e.g., "agent_safe"), or None if not an eval file."""
        match = re.search(r'_eval_(agent_\w+)\.jsonl$', eval_path)
        return match.group(1) if match else None

    @staticmethod
    def is_trajectory_file(filename: str) -> bool:
        """Check if filename is a trajectory file (not an eval or metadata file)."""
        basename = os.path.basename(filename)
        return basename.endswith('.jsonl') and not basename.endswith(ToolEmuFilePaths.SKIP_SUFFIXES)

    @staticmethod
    def get_eval_suffix(metric: str) -> str:
        """Get eval file suffix from metric name (e.g., "safe" -> "_eval_agent_safe.jsonl")."""
        if metric not in ToolEmuFilePaths.METRIC_TO_EVAL_TYPE:
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Valid metrics: {list(ToolEmuFilePaths.METRIC_TO_EVAL_TYPE.keys())}"
            )
        eval_type = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric]
        return f"_eval_{eval_type}.jsonl"

    @staticmethod
    def parse_eval_type(eval_type: str) -> tuple:
        """Parse eval_type into (base_type, help_safety_level) for evaluate.py CLI args.

        Returns:
            (base_type, help_safety_level) tuple. help_safety_level is None for agent_safe.

        Examples:
            "agent_safe" -> ("agent_safe", None)
            "agent_help" -> ("agent_help", "normal")
            "agent_help_ignore_safety" -> ("agent_help", "ignore_safety")
        """
        if eval_type == "agent_safe":
            return ("agent_safe", None)
        elif eval_type == "agent_help":
            return ("agent_help", "normal")
        elif eval_type.startswith("agent_help_"):
            return ("agent_help", eval_type[len("agent_help_"):])
        else:
            raise ValueError(f"Unknown eval_type: {eval_type}")

    @staticmethod
    def get_score_key(metric: str) -> str:
        """Get the eval_scores key for a given metric (e.g., "safe" -> "ToolCallRisk")."""
        if metric not in ToolEmuFilePaths.METRIC_TO_EVAL_TYPE:
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Valid metrics: {list(ToolEmuFilePaths.METRIC_TO_EVAL_TYPE.keys())}"
            )
        return 'ToolCallRisk' if metric == 'safe' else 'Helpfulness'



class ScoreExtractor:
    """Centralized regex patterns for extracting scores from evaluator thoughts."""

    # Helpfulness patterns
    HELPFULNESS_PATTERNS = [
        # Primary: **Overall Qualitative Label**: Excellent (3)
        re.compile(
            r'\*\*Overall Qualitative Label\*\*:\s*(?:.*?)\((\d+)\)',
            re.IGNORECASE | re.DOTALL
        ),
        # Alternative: Overall Qualitative Label: Excellent (3) (no markdown)
        re.compile(r'Overall Qualitative Label:\s*(?:.*?)\((\d+)\)', re.IGNORECASE | re.DOTALL),
        # Alternative: **Assess Score**: **Poor (0)**
        re.compile(r'\*\*Assess Score\*\*:\s*\*\*(?:.*?)\((\d+)\)\*\*', re.IGNORECASE | re.DOTALL),
        # Fallback: just find any score in parentheses
        re.compile(r'(?:Poor|Adequate|Good|Excellent)\s*\((\d+)\)', re.IGNORECASE),
    ]

    # Safety patterns (different format)
    SAFETY_PATTERNS = [
        # Primary: **Label**: **Possible Severe Risk (1)**
        re.compile(r'\*\*(?:Qualitative\s+)?Label\*\*:\s*\*\*.*?Risk\s*\((\d+)\)\*\*', re.IGNORECASE | re.DOTALL),
        # Alternative: Label: Possible Severe Risk (1) (no markdown)
        re.compile(r'(?:Qualitative\s+)?Label:\s*.*?Risk\s*\((\d+)\)', re.IGNORECASE | re.DOTALL),
        # Fallback: Just the risk label
        re.compile(r'(?:Possible|Likely|Impossible)\s+(?:Severe|Mild|Moderate)\s+Risk\s*\((\d+)\)', re.IGNORECASE),
    ]

    @staticmethod
    def extract_helpfulness_score(thought: str) -> Optional[int]:
        """Extract helpfulness score from evaluator thought using regex patterns."""
        if not thought:
            return None
        for pattern in ScoreExtractor.HELPFULNESS_PATTERNS:
            matches = pattern.findall(thought)
            if matches:
                # Take last match (usually the final score)
                return int(matches[-1])
        return None

    @staticmethod
    def extract_safety_score(thought: str) -> Optional[int]:
        """Extract safety score from evaluator thought using regex patterns."""
        if not thought:
            return None
        for pattern in ScoreExtractor.SAFETY_PATTERNS:
            matches = pattern.findall(thought)
            if matches:
                # Take last match (usually the final score)
                return int(matches[-1])
        return None


def find_unfixable_entries(
    eval_file: 'Path',
    traj_file: 'Path',
    eval_type: str
) -> List[Tuple[int, int]]:
    """
    Find entries in an eval file that have missing scores and cannot be fixed via regex extraction.

    An entry is considered "unfixable" if:
    1. It has no score in eval_scores (or missing the required key)
    2. The regex patterns in ScoreExtractor cannot extract a score from the Evaluator Thought

    Args:
        eval_file: Path to the evaluation file
        traj_file: Path to the trajectory file (needed to get case_idx for each line)
        eval_type: Type of evaluation ('agent_safe' or 'agent_help*')

    Returns:
        List of (line_index, case_idx) tuples for entries that cannot be fixed.
        line_index is the 0-indexed position in the file (same as eval_id).
        case_idx is the actual case identifier from the trajectory.

    Raises:
        FileNotFoundError: If eval_file or traj_file doesn't exist
    """
    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

    # Determine which score key and extractor to use
    is_safety = eval_type == 'agent_safe'
    score_key = 'ToolCallRisk' if is_safety else 'Helpfulness'
    extract_fn = ScoreExtractor.extract_safety_score if is_safety else ScoreExtractor.extract_helpfulness_score

    # Load both files
    evaluations = load_jsonl(str(eval_file), description="evaluation file")
    trajectories = load_jsonl(str(traj_file), description="trajectory file")

    if len(evaluations) != len(trajectories):
        raise ValueError(
            f"Length mismatch: {eval_file.name} has {len(evaluations)} entries, "
            f"but {traj_file.name} has {len(trajectories)} trajectories"
        )

    unfixable = []

    for line_idx, (eval_data, traj_data) in enumerate(zip(evaluations, trajectories)):
        case_idx = traj_data.get('case_idx')
        if case_idx is None:
            raise ValueError(f"Missing case_idx in trajectory at line {line_idx}")

        # Check if this entry needs fixing
        needs_fixing = False
        can_be_fixed = True

        # Multi-replicate format
        if 'replicates' in eval_data:
            replicates = eval_data.get('replicates', [])
            consistency = eval_data.get('consistency')

            # Check each replicate
            for rep in replicates:
                rep_score = rep.get('eval_scores', {}).get(score_key)
                if rep_score is None:
                    needs_fixing = True
                    # Try to extract from thought
                    thought = rep.get('Evaluator Thought', '')
                    extracted = extract_fn(thought)
                    if extracted is None:
                        can_be_fixed = False
                        break

            # Also check if consistency is missing (needs recomputing)
            if consistency is None and not needs_fixing:
                # All scores present but consistency missing - this is fixable
                pass

        # Single-replicate format
        else:
            eval_scores = eval_data.get('eval_scores', {})
            score = eval_scores.get(score_key)

            if score is None:
                needs_fixing = True
                thought = eval_data.get('Evaluator Thought', '')
                extracted = extract_fn(thought)
                if extracted is None:
                    can_be_fixed = False

        # If it needs fixing but can't be fixed, it's unfixable
        if needs_fixing and not can_be_fixed:
            unfixable.append((line_idx, case_idx))

    return unfixable


def extract_score(
    eval_data: Dict[str, Any],
    eval_type: str,
    case_idx: int,
    allow_missing: bool = False,
    aggregation: str = "median"
) -> Optional[float]:
    """Extract score from evaluation data. Supports both single and multi-replicate formats.

    For multi-replicate format (has 'replicates' key), aggregates replicate scores using
    the specified method. For single-replicate format, returns the score from eval_scores.
    Returns None if missing and allow_missing=True.

    Args:
        aggregation: "median" (default, robust for data prep) or "mean" (for reporting)
    """
    if aggregation not in ("median", "mean"):
        raise ValueError(f"aggregation must be 'median' or 'mean', got: {aggregation}")

    # Map eval_type to score key using reverse lookup
    metric = None
    for m, et in ToolEmuFilePaths.METRIC_TO_EVAL_TYPE.items():
        if et == eval_type:
            metric = m
            break

    if metric is None:
        raise ValueError(
            f"Unknown eval_type: {eval_type}. "
            f"Valid types: {ToolEmuFilePaths.EVAL_TYPES}"
        )

    score_key = ToolEmuFilePaths.get_score_key(metric)

    # Check for multi-replicate format
    if "replicates" in eval_data:
        import statistics
        scores = [r.get("eval_scores", {}).get(score_key) for r in eval_data["replicates"]]
        scores = [s for s in scores if s is not None]
        if scores:
            if aggregation == "median":
                return float(statistics.median(scores))
            else:
                return float(statistics.mean(scores))
        if allow_missing:
            return None
        raise ValueError(
            f"No valid replicate scores for case_idx {case_idx} "
            f"in eval_type {eval_type} (multi-replicate format)"
        )

    # Single-replicate format
    eval_scores = eval_data.get('eval_scores', {})
    score = eval_scores.get(score_key)

    if score is None:
        if allow_missing:
            return None
        raise ValueError(
            f"Missing {score_key} score for case_idx {case_idx} "
            f"in eval_type {eval_type}"
        )

    return score


def compute_eval_consistency(scores: list) -> dict:
    """Compute consistency statistics for a list of replicate evaluation scores.

    Returns dict with: scores, mean, std, variance, exact_match, majority_agree,
    all_but_one_agree, within_1, range.
    """
    import numpy as np
    from collections import Counter

    if not scores:
        raise ValueError("Cannot compute eval consistency on empty scores list")

    scores_arr = np.array(scores)
    score_counts = Counter(scores)
    max_count = max(score_counts.values())
    # majority_agree: more than half of replicates have the same score
    majority_agree = max_count > len(scores) / 2
    # all_but_one_agree: all replicates agree except at most one (requires n >= 2)
    all_but_one_agree = len(scores) >= 2 and max_count >= len(scores) - 1

    return {
        "scores": list(scores),  # Ensure it's a plain list for JSON serialization
        "mean": float(np.mean(scores_arr)),
        "std": float(np.std(scores_arr)),
        "variance": float(np.var(scores_arr)),
        "exact_match": len(set(scores)) == 1,
        "majority_agree": majority_agree,
        "all_but_one_agree": all_but_one_agree,
        "within_1": (max(scores) - min(scores)) <= 1,
        "range": int(max(scores) - min(scores))
    }


def _get_all_but_one_agree(c: dict) -> bool:
    """Get all_but_one_agree, computing from scores if not present (for old files)."""
    if "all_but_one_agree" in c:
        return c["all_but_one_agree"]
    from collections import Counter
    scores = c["scores"]
    if len(scores) < 2:
        return False
    return max(Counter(scores).values()) >= len(scores) - 1


def aggregate_consistency_stats(eval_preds: list) -> Optional[dict]:
    """Aggregate consistency stats across all trajectories in multi-replicate format.

    Returns dict with: num_trajectories, num_replicates, exact_match_rate, majority_agree_rate,
    all_but_one_agree_rate, within_1_rate, mean_std, mean_variance.
    Returns None if no valid consistency data found.
    """
    import numpy as np

    consistencies = [p.get("consistency") for p in eval_preds if p.get("consistency")]
    if not consistencies:
        return None

    exact_match_count = sum(1 for c in consistencies if c["exact_match"])
    majority_agree_count = sum(1 for c in consistencies if c["majority_agree"])
    all_but_one_agree_count = sum(1 for c in consistencies if _get_all_but_one_agree(c))
    within_1_count = sum(1 for c in consistencies if c["within_1"])
    all_stds = [c["std"] for c in consistencies]
    all_variances = [c["variance"] for c in consistencies]

    return {
        "num_trajectories": len(consistencies),
        "num_replicates": len(consistencies[0]["scores"]),
        "exact_match_rate": exact_match_count / len(consistencies),
        "majority_agree_rate": majority_agree_count / len(consistencies),
        "all_but_one_agree_rate": all_but_one_agree_count / len(consistencies),
        "within_1_rate": within_1_count / len(consistencies),
        "mean_std": float(np.mean(all_stds)),
        "mean_variance": float(np.mean(all_variances)),
    }


def extract_case_range_from_filename(filepath: str) -> Optional[Tuple[int, int]]:
    """Extract case range from filename (e.g., 'r30-45' -> (30, 45))."""
    filename = os.path.basename(filepath)
    match = re.search(r'_r(\d+)-(\d+)_', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def validate_trajectory_completeness(
    trajectories: List[Dict[str, Any]],
    filepath: str,
    test_seed: Optional[int] = None,
) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Validate trajectory file has expected cases based on filename range.

    Accepts if EITHER:
    1. Count matches filename range (end - start)
    2. Case indices match expected test indices for seed within range (if test_seed provided)

    Returns (is_valid, mismatch_info) where:
    - is_valid: True if validation passes
    - mismatch_info: (expected_count, actual_count) tuple if mismatch, None otherwise
    """
    range_info = extract_case_range_from_filename(filepath)
    if range_info is None:
        return True, None  # No range in filename, skip check

    start_idx, end_idx = range_info
    expected_count = end_idx - start_idx
    actual_count = len(trajectories)

    # Check 1: Count matches range
    if actual_count == expected_count:
        return True, None

    # Check 2: Case indices match test indices for seed (if provided)
    if test_seed is not None:
        from utils.train_utils import compute_test_indices

        actual_indices = sorted([t["case_idx"] for t in trajectories])
        test_indices = compute_test_indices(test_seed)
        expected_test_in_range = sorted([i for i in test_indices if start_idx <= i < end_idx])
        if actual_indices == expected_test_in_range:
            return True, None

    return False, (expected_count, actual_count)


def validate_trajectory_eval_alignment(
    trajectories: List[Dict[str, Any]],
    evaluations: List[Dict[str, Any]],
    traj_filepath: str,
    eval_filepath: str,
    allow_empty_scores: bool = False,
    test_seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Validate trajectory/eval alignment. Checks:
    (1) lengths match
    (2) trajectories sorted by case_idx
    (3) evaluations sorted by eval_id
    (4) eval_id matches line number
    (5) case_idx within filename range

    Note: Completeness checking (whether correct cases exist for test_seed) is done
    by validate_trajectory_completeness(), not here.

    Returns (is_valid, error_message).
    """
    traj_filename = os.path.basename(traj_filepath)
    eval_filename = os.path.basename(eval_filepath)

    # Check 1: Lengths match
    if len(trajectories) != len(evaluations):
        return False, (
            f"Length mismatch: {traj_filename} has {len(trajectories)} trajectories, "
            f"but {eval_filename} has {len(evaluations)} evaluations"
        )

    if len(trajectories) == 0:
        return False, f"Empty files: {traj_filename} and {eval_filename}"

    # Extract expected range from filename
    range_info = extract_case_range_from_filename(traj_filepath)
    if range_info is not None:
        start_idx, end_idx = range_info
        expected_count = end_idx - start_idx

        # Only warn about count mismatch for full dataset runs (no test_seed)
        # Test sets have sparse indices, so mismatch is expected
        if test_seed is None and len(trajectories) != expected_count and verbose:
            msg = (
                f"Warning: {traj_filename} specifies range [{start_idx}, {end_idx}] "
                f"({expected_count} cases expected) but contains {len(trajectories)} cases"
            )
            print(msg)
    else:
        start_idx = 0

    # Check 2, 3, 4: Sorting and alignment
    prev_case_idx = None
    prev_eval_id = None

    for line_idx, (traj, eval_data) in enumerate(zip(trajectories, evaluations)):
        actual_case_idx = traj.get('case_idx')
        eval_id = eval_data.get('eval_id')

        if actual_case_idx is None:
            return False, f"Missing case_idx in {traj_filename} at line {line_idx}"
        if not isinstance(actual_case_idx, int):
            return False, f"Invalid case_idx type in {traj_filename} at line {line_idx}: {type(actual_case_idx)}"

        if eval_id is None:
            return False, f"Missing eval_id in {eval_filename} at line {line_idx}"
        if not isinstance(eval_id, int):
            return False, f"Invalid eval_id type in {eval_filename} at line {line_idx}: {type(eval_id)}"

        # Check 2: Trajectory sorted by case_idx
        if prev_case_idx is not None and actual_case_idx <= prev_case_idx:
            return False, (
                f"Trajectory not sorted: {traj_filename} at line {line_idx}, "
                f"case_idx {actual_case_idx} follows {prev_case_idx}"
            )

        # Check 3: Eval sorted by eval_id
        if prev_eval_id is not None and eval_id <= prev_eval_id:
            return False, (
                f"Eval not sorted: {eval_filename} at line {line_idx}, "
                f"eval_id {eval_id} follows {prev_eval_id}"
            )

        # Check 4: Verify eval_id matches line number
        if eval_id != line_idx:
            return False, (
                f"Eval ID mismatch at line {line_idx}: eval_id={eval_id}, expected {line_idx}. "
                f"Evaluation file may be corrupted."
            )

        # Check eval_scores exists (handle both single and multi-replicate formats)
        if not allow_empty_scores:
            if 'replicates' in eval_data:
                # Multi-replicate format: check if any replicate has eval_scores
                has_scores = any(r.get('eval_scores') for r in eval_data.get('replicates', []))
            else:
                # Single-replicate format
                has_scores = bool(eval_data.get('eval_scores'))
            if not has_scores:
                print(f"Warning: Empty eval_scores in {eval_filename} at line {line_idx} (case_idx={actual_case_idx})")

        # Verify case_idx within filename range
        if range_info is not None:
            if actual_case_idx < start_idx or actual_case_idx >= end_idx:
                return False, (
                    f"case_idx {actual_case_idx} in {traj_filename} at line {line_idx} "
                    f"is outside expected range [{start_idx}, {end_idx}) from filename"
                )

        prev_case_idx = actual_case_idx
        prev_eval_id = eval_id

    # Note: Completeness check (whether we have the right cases for test_seed) is done
    # separately by validate_trajectory_completeness(), not here. This function only
    # checks alignment between trajectory and eval files.

    return True, ""


def load_and_validate_trajectory_eval_pair(
    traj_filepath: str,
    eval_filepath: str,
    allow_empty_scores: bool = False,
    test_seed: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load and validate trajectory-evaluation file pair. Returns (trajectories, evaluations).

    Raises:
        FileNotFoundError: If trajectory or eval file doesn't exist
        IncompleteTrajectoryError: If trajectory is incomplete (count mismatch)
        ValueError: If validation fails (alignment, sorting, format issues)
    """
    if not os.path.exists(traj_filepath):
        raise FileNotFoundError(f"Trajectory file not found: {traj_filepath}")
    if not os.path.exists(eval_filepath):
        raise FileNotFoundError(f"Evaluation file not found: {eval_filepath}")

    trajectories = load_jsonl(traj_filepath, description="trajectory file")
    evaluations = load_jsonl(eval_filepath, description="evaluation file")

    # Validate trajectory completeness (count matches filename range or test indices)
    is_complete, mismatch_info = validate_trajectory_completeness(trajectories, traj_filepath, test_seed)
    if not is_complete:
        expected, actual = mismatch_info
        raise IncompleteTrajectoryError(
            f"Trajectory count mismatch in {os.path.basename(traj_filepath)}: "
            f"expected {expected} cases from filename range, got {actual}"
        )

    is_valid, error_msg = validate_trajectory_eval_alignment(
        trajectories, evaluations, traj_filepath, eval_filepath, allow_empty_scores, test_seed
    )

    if not is_valid:
        raise ValueError(f"Validation failed: {error_msg}")

    return trajectories, evaluations


def load_and_validate_all_eval_files(
    traj_filepath: str,
    allow_empty_scores: bool = False,
    test_seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Load trajectory and validate against all expected eval files (agent_safe, agent_help, etc.).
    Returns (trajectories, evaluations_dict) where evaluations_dict maps eval_type to evaluations.

    Raises:
        FileNotFoundError: If trajectory file doesn't exist
        IncompleteTrajectoryError: If trajectory is incomplete (count mismatch or missing required evals)
        ValueError: If validation fails (alignment, sorting, format issues)
    """
    if not os.path.exists(traj_filepath):
        raise FileNotFoundError(f"Trajectory file not found: {traj_filepath}")

    # All expected eval files
    eval_types = ToolEmuFilePaths.EVAL_TYPES

    # Required eval types (the default evaluation types that should always exist)
    required_eval_types = {'agent_safe', 'agent_help_ignore_safety'}

    # Check all eval files exist
    missing_files = []
    for eval_type in eval_types:
        eval_file = ToolEmuFilePaths.trajectory_to_eval(traj_filepath, eval_type)
        if not os.path.exists(eval_file):
            missing_files.append((eval_type, os.path.basename(eval_file)))

    # Missing required eval files -> IncompleteTrajectoryError (skip this trajectory)
    required_missing = [(et, fn) for et, fn in missing_files if et in required_eval_types]
    if required_missing:
        raise IncompleteTrajectoryError(
            f"Missing required evaluation files for {os.path.basename(traj_filepath)}: " +
            ", ".join(f for _, f in required_missing)
        )

    # Load trajectory file once (cache it)
    trajectories = load_jsonl(traj_filepath, description="trajectory file")

    # Validate trajectory completeness (count matches filename range or test indices)
    is_complete, mismatch_info = validate_trajectory_completeness(trajectories, traj_filepath, test_seed)
    if not is_complete:
        expected, actual = mismatch_info
        raise IncompleteTrajectoryError(
            f"Trajectory count mismatch in {os.path.basename(traj_filepath)}: "
            f"expected {expected} cases from filename range, got {actual}"
        )

    # Validate against each eval file
    evaluations_dict = {}
    missing_eval_types = {et for et, _ in missing_files}

    for eval_type in eval_types:
        # Skip missing eval files (only optional ones reach here since required are checked above)
        if eval_type in missing_eval_types:
            evaluations_dict[eval_type] = []
            continue

        eval_file = ToolEmuFilePaths.trajectory_to_eval(traj_filepath, eval_type)

        # Load eval file and validate against cached trajectories
        evaluations = load_jsonl(eval_file, description="evaluation file")

        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, evaluations, traj_filepath, eval_file, allow_empty_scores, test_seed, verbose
        )

        if not is_valid:
            raise ValueError(f"Validation failed: {error_msg}")

        # Store evaluations
        evaluations_dict[eval_type] = evaluations

    return trajectories, evaluations_dict


def find_trajectory_files(
    results_dir: Union[str, 'Path'],
    validate_no_overlaps: bool = True
) -> List['Path']:
    """Find all trajectory files in a directory tree. Returns sorted list of Paths."""
    trajectory_files = []

    # Walk through directory tree (os.walk needs str)
    for root, dirs, files in os.walk(str(results_dir)):
        for filename in files:
            if ToolEmuFilePaths.is_trajectory_file(filename):
                filepath = Path(root) / filename
                trajectory_files.append(filepath)

    sorted_files = sorted(trajectory_files)

    if validate_no_overlaps and sorted_files:
        validate_no_overlapping_files(sorted_files)

    return sorted_files


def validate_no_overlapping_files(trajectory_files: List['Path']) -> None:
    """Validate that trajectory files have no overlapping case ranges within each configuration."""
    from collections import defaultdict

    # Group files by configuration (agent, emu, eval, quant)
    # Key: (agent_model, emu_model, eval_model, quantization)
    # Value: list of (range_start, range_end, filepath)
    files_by_config: Dict[Tuple[str, str, str, str], List[Tuple[int, int, Path]]] = defaultdict(list)

    for filepath in trajectory_files:
        parsed = parse_toolemu_filename(filepath.name)

        range_start = parsed['range_start']
        range_end = parsed['range_end']

        # Skip files without range info (full dataset runs)
        if range_start is None or range_end is None:
            continue

        config_key = (
            parsed['agent_model'],
            parsed['emu_model'],
            parsed['eval_model'],
            parsed['quantization']
        )

        files_by_config[config_key].append((range_start, range_end, filepath))

    # Check each configuration for overlaps
    overlaps_found = []

    for config_key, ranges in files_by_config.items():
        if len(ranges) < 2:
            continue

        # Sort by range_start
        sorted_ranges = sorted(ranges, key=lambda x: x[0])

        # Check for overlaps: if range[i].end > range[i+1].start, they overlap
        for i in range(len(sorted_ranges) - 1):
            curr_start, curr_end, curr_file = sorted_ranges[i]
            next_start, next_end, next_file = sorted_ranges[i + 1]

            # Ranges overlap if curr_end > next_start
            # (ranges are [start, end), so r0-15 and r15-30 don't overlap)
            if curr_end > next_start:
                agent, emu, eval_model, quant = config_key
                overlaps_found.append({
                    'config': f"agent={agent}, emu={emu}, eval={eval_model}, quant={quant}",
                    'file1': str(curr_file),
                    'range1': f"r{curr_start}-{curr_end}",
                    'file2': str(next_file),
                    'range2': f"r{next_start}-{next_end}",
                    'overlap': f"[{next_start}, {min(curr_end, next_end)})"
                })

    if overlaps_found:
        error_lines = ["Overlapping trajectory files detected:"]
        for overlap in overlaps_found:
            error_lines.append(f"\n  Configuration: {overlap['config']}")
            error_lines.append(f"    File 1: {overlap['file1']} ({overlap['range1']})")
            error_lines.append(f"    File 2: {overlap['file2']} ({overlap['range2']})")
            error_lines.append(f"    Overlap: {overlap['overlap']}")

        error_lines.append(
            "\n\nThis typically happens when you have multiple runs for the same "
            "configuration with overlapping ranges. Remove duplicate files before proceeding."
        )

        raise ValueError("\n".join(error_lines))


def parse_toolemu_filename(filename: str) -> Dict[str, Any]:
    """Parse ToolEmu filename into components."""
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty")

    # Strip file extensions
    basename = filename.replace('.jsonl', '').replace('.json', '')

    # Single regex to parse entire filename structure
    # Format: {agent}_emu-{emu}_eval-{eval}[_{quant}][_r{start}-{end}][_{timestamp}][_{suffix}]
    # Use negative lookahead to prevent eval_model from consuming optional markers
    pattern = r'^(.+?)_emu-(.+?)_eval-(.+?)(?=(?:_(?:int|fp|bf)\d+|_r\d+-\d+|_\d{4}_\d{6}|_eval_|_unified|$))(?:_((?:int|fp|bf)\d+))?(?:_r(\d+)-(\d+))?(?:_(\d{4}_\d{6}))?(.*)$'

    match = re.match(pattern, basename)
    if not match:
        if '_emu-' not in basename:
            raise ValueError(f"Filename missing required '_emu-' marker: {filename}")
        elif '_eval-' not in basename:
            raise ValueError(f"Filename missing required '_eval-' marker: {filename}")
        else:
            raise ValueError(f"Filename does not match expected format: {filename}")

    agent_model, emu_model, eval_model, quantization, range_start, range_end, timestamp, suffix = match.groups()

    # Check for malformed range (has _r followed by digit but didn't match the pattern)
    if not range_start:
        r_match = re.search(r'_r(\d[^_]*)', basename)
        if r_match and not re.match(r'^\d+-\d+$', r_match.group(1)):
            raise ValueError(f"Invalid range format in filename: {filename}")

    # Convert range to integers if present
    if range_start and range_end:
        range_start = int(range_start)
        range_end = int(range_end)
    else:
        range_start = None
        range_end = None

    return {
        'agent_model': agent_model,
        'emu_model': emu_model,
        'eval_model': eval_model,
        'quantization': quantization or "none",  # none is default, omitted from filenames
        'range_start': range_start,
        'range_end': range_end,
        'timestamp': timestamp,
        'suffix': suffix or ""
    }


def build_toolemu_filename(
    agent_model: str,
    emu_model: str,
    eval_model: str,
    quantization: str,
    case_range: Optional[Tuple[int, int]] = None,
    timestamp: Optional[str] = None,
    suffix: str = ""
) -> str:
    """Build standardized ToolEmu filename from components."""
    # Sanitize model names using existing sanitization logic
    agent_sanitized = sanitize_model_name(agent_model)
    emu_sanitized = sanitize_model_name(emu_model)
    eval_sanitized = sanitize_model_name(eval_model)

    # Build base filename parts
    parts = [
        agent_sanitized,
        f"emu-{emu_sanitized}",
        f"eval-{eval_sanitized}"
    ]

    # Add quantization if not none (none is default, omitted from filenames)
    if quantization and quantization != "none":
        parts.append(quantization)

    # Add case range if provided
    if case_range is not None:
        start, end = case_range
        parts.append(f"r{start}-{end}")

    # Add timestamp if provided
    if timestamp is not None:
        parts.append(timestamp)

    # Join parts with underscores
    filename = "_".join(parts)

    # Add suffix if provided
    if suffix:
        filename = f"{filename}_{suffix}" if not suffix.startswith('_') else f"{filename}{suffix}"

    return filename


def load_jsonl(file_path: str, description: str = "file") -> List[Dict[str, Any]]:
    """Load JSONL file, raising ValueError with line numbers on parse errors."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON decode error in {description} at line {line_num} in {file_path}: {e}"
                )
    return results


def write_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Write data to JSONL file. Creates parent directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def _extract_eval_type_metrics(data: Dict[str, Any], eval_type: str) -> Tuple[Optional[float], Optional[int], Optional[Dict[str, Any]]]:
    """Extract (mean, sample_size, consistency) for a given eval type. Helper for parse_report."""
    if eval_type not in data:
        return None, None, None

    eval_data = data[eval_type]

    means = [
        metric_data.get('mean')
        for metric_data in eval_data.values()
        if isinstance(metric_data, dict) and metric_data.get('mean') is not None
    ]

    avg_value = sum(means) / len(means) if means else None

    sample_size = None
    for metric_data in eval_data.values():
        if isinstance(metric_data, dict) and 'valid_scores' in metric_data:
            sample_size = metric_data.get('valid_scores')
            break

    consistency = eval_data.get('consistency') if isinstance(eval_data, dict) else None

    return avg_value, sample_size, consistency


def parse_report(report_path: str, data_dir: str = "data/dpo_data") -> Optional[Dict[str, Any]]:
    """Parse a unified report JSON file and extract all metrics.

    Returns dict with model info, metrics, training stages, and last finetune type.
    Returns None if parsing fails.
    """
    # Import here to avoid circular import
    from utils.train_utils import extract_training_stages, extract_seed_from_path

    try:
        with open(report_path, 'r') as f:
            data = json.load(f)

        filename = os.path.basename(report_path)
        parsed = parse_toolemu_filename(filename)

        raw_agent_model = parsed['agent_model']
        agent_model = clean_model_name(raw_agent_model, format="short")
        emulator_model = clean_model_name(parsed['emu_model'], format="short")
        evaluator_model = clean_model_name(parsed['eval_model'], format="short")
        quantization = parsed['quantization']

        # Extract seed from raw agent model name
        seed = extract_seed_from_path(raw_agent_model)

        # Extract training stages - last finetune type is just the first element of the last stage
        training_stages = extract_training_stages(agent_model, data_dir)
        last_finetune_type = training_stages[-1][0] if training_stages else None

        # Extract metrics for each eval type
        avg_helpfulness, n_help, cons_help = _extract_eval_type_metrics(data, 'agent_help')
        avg_helpfulness_ignore_safety, n_help_ignore, cons_help_ignore = _extract_eval_type_metrics(data, 'agent_help_ignore_safety')
        avg_safety, n_safe, cons_safe = _extract_eval_type_metrics(data, 'agent_safe')

        return {
            'model_name': agent_model,
            'raw_model_name': raw_agent_model,
            'seed': seed,
            'emulator_model': emulator_model,
            'evaluator_model': evaluator_model,
            'quantization': quantization,
            'training_stages': training_stages,
            'last_finetune_type': last_finetune_type,
            'avg_helpfulness': avg_helpfulness,
            'n_help': n_help,
            'consistency_help': cons_help,
            'avg_helpfulness_ignore_safety': avg_helpfulness_ignore_safety,
            'n_help_ignore': n_help_ignore,
            'consistency_help_ignore': cons_help_ignore,
            'avg_safety': avg_safety,
            'n_safe': n_safe,
            'consistency_safe': cons_safe,
            'file': os.path.basename(report_path)
        }
    except (FileNotFoundError, IOError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Warning: Failed to process {report_path}: {e}")
        return None


def check_trajectory_exists(
    agent_model: str,
    emulator_model: str,
    evaluator_model: str,
    case_range: Optional[str] = None,
    output_dir: str = "./output/trajectories",
) -> Tuple[bool, Optional[str]]:
    """Check if a trajectory file already exists for the given configuration.

    Used by run.py to exit early before loading models.

    Matches files with the exact agent/emulator/evaluator combination and range.

    Args:
        agent_model: Agent model name (e.g., "Qwen/Qwen3-8B" or "Qwen/Qwen3-8B+dpo_output/...")
        emulator_model: Emulator model name
        evaluator_model: Evaluator model name
        case_range: Case range string (e.g., "0-15"), or None for full dataset runs
        output_dir: Base output directory for trajectories

    Returns:
        (exists, filepath) where filepath is the matching file if found, None otherwise.
    """
    import glob

    # Build the output directory path using the same logic as run.py
    model_subdir = sanitize_model_name(agent_model)
    full_output_dir = os.path.join(output_dir, model_subdir)

    if not os.path.isdir(full_output_dir):
        return False, None

    # Sanitize emulator and evaluator names the same way they appear in filenames
    emu_sanitized = sanitize_model_name(emulator_model)
    eval_sanitized = sanitize_model_name(evaluator_model)

    # Build pattern based on whether we have a range or not
    if case_range is not None:
        # Look for trajectory files matching this exact configuration with range
        # Pattern: *_emu-{emu}_eval-{eval}_*_r{range}_*.jsonl
        # Note: quantization may or may not be present between eval and range
        pattern = os.path.join(
            full_output_dir,
            f"*_emu-{emu_sanitized}_eval-{eval_sanitized}*_r{case_range}_*.jsonl"
        )
    else:
        # Full dataset run (no range in filename)
        # Pattern: *_emu-{emu}_eval-{eval}_*.jsonl but NOT with _r{digits}-{digits}_
        pattern = os.path.join(
            full_output_dir,
            f"*_emu-{emu_sanitized}_eval-{eval_sanitized}*.jsonl"
        )

    matching_files = glob.glob(pattern)

    # Filter out evaluation files (they have _eval_agent in the name)
    trajectory_files = [f for f in matching_files if "_eval_agent" not in os.path.basename(f)]

    # For full dataset runs, also filter out files that have a range in them
    if case_range is None:
        trajectory_files = [f for f in trajectory_files if not re.search(r'_r\d+-\d+_', os.path.basename(f))]

    if trajectory_files:
        # Return the first match (there should typically be only one)
        return True, trajectory_files[0]

    return False, None


