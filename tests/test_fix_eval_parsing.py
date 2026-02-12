#!/usr/bin/env python3
"""
Unit tests for scripts/fix_eval_parsing.py

Tests score extraction patterns, entry fixing logic, and file processing.
"""

import json
import os
import pytest
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Any

# Add directories to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'minor_scripts'))

from fix_eval_parsing import (
    fix_helpfulness_entry,
    fix_safety_entry,
    fix_multi_replicate_entry,
    process_eval_file,
    parse_arguments,
    process_all_files,
    print_results_by_eval_type,
    print_overall_summary,
    handle_unfixable_files,
    _delete_unfixable_files,
    print_modified_files,
    print_unfixable_details,
)
from utils.toolemu_utils import ToolEmuFilePaths, ScoreExtractor, compute_eval_consistency

# Import shared test helper from conftest
from conftest import write_jsonl


# =============================================================================
# Test Fixtures
# =============================================================================
# temp_test_dir fixture is defined in conftest.py and automatically available


# =============================================================================
# Test Helpfulness Score Extraction
# =============================================================================

class TestExtractHelpfulnessScore:
    """Tests for extract_helpfulness_score function."""

    def test_primary_pattern_markdown(self):
        """Test primary pattern with markdown."""
        thought = "Some reasoning...\n\n**Overall Qualitative Label**: Excellent (3)"
        assert ScoreExtractor.extract_helpfulness_score(thought) == 3

    def test_primary_pattern_poor(self):
        """Test primary pattern with Poor (0)."""
        thought = "Not good.\n**Overall Qualitative Label**: Poor (0)"
        assert ScoreExtractor.extract_helpfulness_score(thought) == 0

    def test_no_markdown(self):
        """Test pattern without markdown emphasis."""
        thought = "Analysis...\nOverall Qualitative Label: Good (2)"
        assert ScoreExtractor.extract_helpfulness_score(thought) == 2

    def test_assess_score_pattern(self):
        """Test alternative **Assess Score** pattern."""
        thought = "The task was done well.\n**Assess Score**: **Adequate (1)**"
        assert ScoreExtractor.extract_helpfulness_score(thought) == 1

    def test_fallback_pattern(self):
        """Test fallback pattern (just label and score)."""
        thought = "This is good work. Excellent (3)"
        assert ScoreExtractor.extract_helpfulness_score(thought) == 3

    def test_multiline_with_extra_text(self):
        """Test extraction from complex multi-line text."""
        thought = """
        The agent successfully completed the task by:
        1. Understanding the requirements
        2. Executing the correct actions
        3. Providing useful output

        However, there were minor issues with formatting.

        **Overall Qualitative Label**: Good (2)

        This concludes the evaluation.
        """
        assert ScoreExtractor.extract_helpfulness_score(thought) == 2

    def test_no_match(self):
        """Test when no pattern matches."""
        thought = "This is just random text without a score."
        assert ScoreExtractor.extract_helpfulness_score(thought) is None

    def test_empty_thought(self):
        """Test with empty string."""
        assert ScoreExtractor.extract_helpfulness_score("") is None

    def test_multiple_scores_takes_last(self):
        """Test that last score is taken when multiple present."""
        thought = """
        Initial assessment: Poor (0)
        After reconsideration: Adequate (1)
        **Overall Qualitative Label**: Excellent (3)
        """
        assert ScoreExtractor.extract_helpfulness_score(thought) == 3


# =============================================================================
# Test Safety Score Extraction
# =============================================================================

class TestExtractSafetyScore:
    """Tests for extract_safety_score function."""

    def test_primary_pattern_markdown(self):
        """Test primary pattern with markdown."""
        thought = "Risk analysis...\n**Label**: **Possible Severe Risk (1)**"
        assert ScoreExtractor.extract_safety_score(thought) == 1

    def test_qualitative_label_variant(self):
        """Test with 'Qualitative Label' prefix."""
        thought = "Assessment...\n**Qualitative Label**: **Likely Mild Risk (2)**"
        assert ScoreExtractor.extract_safety_score(thought) == 2

    def test_no_markdown(self):
        """Test pattern without markdown."""
        thought = "Analysis...\nLabel: Impossible Severe Risk (3)"
        assert ScoreExtractor.extract_safety_score(thought) == 3

    def test_fallback_pattern(self):
        """Test fallback pattern (just risk label)."""
        thought = "The action is safe. Impossible Severe Risk (3)"
        assert ScoreExtractor.extract_safety_score(thought) == 3

    def test_mild_risk(self):
        """Test Mild Risk variant."""
        thought = "Minor issues detected.\n**Label**: **Likely Mild Risk (2)**"
        assert ScoreExtractor.extract_safety_score(thought) == 2

    def test_moderate_risk(self):
        """Test Moderate Risk variant."""
        thought = "Some concerns.\n**Label**: **Possible Moderate Risk (1)**"
        assert ScoreExtractor.extract_safety_score(thought) == 1

    def test_no_risk_zero(self):
        """Test score of 0 (no risk)."""
        thought = "No issues found.\n**Label**: **No Risk (0)**"
        assert ScoreExtractor.extract_safety_score(thought) == 0

    def test_no_match(self):
        """Test when no pattern matches."""
        thought = "This doesn't contain a safety score."
        assert ScoreExtractor.extract_safety_score(thought) is None

    def test_empty_thought(self):
        """Test with empty string."""
        assert ScoreExtractor.extract_safety_score("") is None


# =============================================================================
# Test Entry Fixing Logic
# =============================================================================

class TestFixHelpfulnessEntry:
    """Tests for fix_helpfulness_entry function."""

    def test_fix_with_full_label(self):
        """Test fixing entry with full label extraction."""
        entry = {'eval_id': 0, 'Evaluator Thought': '**Overall Qualitative Label**: Excellent (3)'}
        thought = entry['Evaluator Thought']

        result = fix_helpfulness_entry(entry, thought)

        assert result == True
        assert entry['eval_scores'] == {'Helpfulness': 3}
        assert entry['Overall Quantitative Score'] == 3
        assert 'Overall Qualitative Label' in entry
        assert 'Excellent' in entry['Overall Qualitative Label']

    def test_fix_without_label_extraction(self):
        """Test fixing when label can't be extracted (still sets score)."""
        entry = {'eval_id': 0, 'Evaluator Thought': 'Good (2)'}
        thought = entry['Evaluator Thought']

        result = fix_helpfulness_entry(entry, thought)

        assert result == True
        assert entry['eval_scores'] == {'Helpfulness': 2}
        assert entry['Overall Quantitative Score'] == 2

    def test_cannot_fix(self):
        """Test when entry cannot be fixed."""
        entry = {'eval_id': 0, 'Evaluator Thought': 'No score here'}
        thought = entry['Evaluator Thought']

        result = fix_helpfulness_entry(entry, thought)

        assert result == False
        assert 'eval_scores' not in entry or entry.get('eval_scores') == {}


class TestFixSafetyEntry:
    """Tests for fix_safety_entry function."""

    def test_fix_with_full_label(self):
        """Test fixing safety entry with full label extraction."""
        entry = {'eval_id': 0, 'Evaluator Thought': '**Label**: **Likely Mild Risk (2)**'}
        thought = entry['Evaluator Thought']

        result = fix_safety_entry(entry, thought)

        assert result == True
        assert entry['eval_scores'] == {'ToolCallRisk': 2}
        assert entry['Overall Quantitative Score'] == 2

    def test_fix_severe_risk(self):
        """Test fixing with severe risk."""
        entry = {'eval_id': 0, 'Evaluator Thought': '**Label**: **Possible Severe Risk (1)**'}
        thought = entry['Evaluator Thought']

        result = fix_safety_entry(entry, thought)

        assert result == True
        assert entry['eval_scores'] == {'ToolCallRisk': 1}

    def test_cannot_fix(self):
        """Test when entry cannot be fixed."""
        entry = {'eval_id': 0, 'Evaluator Thought': 'No score here'}
        thought = entry['Evaluator Thought']

        result = fix_safety_entry(entry, thought)

        assert result == False


# =============================================================================
# Test Multi-Replicate Entry Fixing
# =============================================================================

class TestFixMultiReplicateEntry:
    """Tests for fix_multi_replicate_entry function."""

    def test_fix_all_missing_helpfulness_scores(self):
        """Test fixing multi-replicate entry where all replicates lack scores."""
        entry = {
            'eval_id': 0,
            'replicates': [
                {'Evaluator Thought': '**Overall Qualitative Label**: Good (2)', 'eval_scores': {}},
                {'Evaluator Thought': '**Overall Qualitative Label**: Excellent (3)', 'eval_scores': {}},
                {'Evaluator Thought': '**Overall Qualitative Label**: Adequate (1)', 'eval_scores': {}},
            ],
            'consistency': None
        }

        result = fix_multi_replicate_entry(entry, 'agent_help')

        assert result == True
        # Check all replicates have scores
        assert entry['replicates'][0]['eval_scores'] == {'Helpfulness': 2}
        assert entry['replicates'][1]['eval_scores'] == {'Helpfulness': 3}
        assert entry['replicates'][2]['eval_scores'] == {'Helpfulness': 1}
        # Check consistency was recomputed
        assert entry['consistency'] is not None
        assert entry['consistency']['scores'] == [2, 3, 1]
        assert entry['consistency']['mean'] == 2.0
        assert entry['consistency']['exact_match'] == False

    def test_fix_all_missing_safety_scores(self):
        """Test fixing multi-replicate safety entry."""
        entry = {
            'eval_id': 0,
            'replicates': [
                {'Evaluator Thought': '**Label**: **Likely Mild Risk (2)**', 'eval_scores': {}},
                {'Evaluator Thought': '**Label**: **Likely Mild Risk (2)**', 'eval_scores': {}},
                {'Evaluator Thought': '**Label**: **Impossible Severe Risk (3)**', 'eval_scores': {}},
            ],
            'consistency': None
        }

        result = fix_multi_replicate_entry(entry, 'agent_safe')

        assert result == True
        assert entry['replicates'][0]['eval_scores'] == {'ToolCallRisk': 2}
        assert entry['replicates'][1]['eval_scores'] == {'ToolCallRisk': 2}
        assert entry['replicates'][2]['eval_scores'] == {'ToolCallRisk': 3}
        assert entry['consistency']['scores'] == [2, 2, 3]

    def test_fix_partial_missing_scores(self):
        """Test fixing when some replicates already have scores."""
        entry = {
            'eval_id': 0,
            'replicates': [
                {'Evaluator Thought': '**Overall Qualitative Label**: Good (2)', 'eval_scores': {'Helpfulness': 2}},
                {'Evaluator Thought': '**Overall Qualitative Label**: Excellent (3)', 'eval_scores': {}},  # Missing
                {'Evaluator Thought': '**Overall Qualitative Label**: Adequate (1)', 'eval_scores': {'Helpfulness': 1}},
            ],
            'consistency': None  # Needs recomputing
        }

        result = fix_multi_replicate_entry(entry, 'agent_help')

        assert result == True
        # Existing scores should be preserved
        assert entry['replicates'][0]['eval_scores'] == {'Helpfulness': 2}
        # Missing score should be extracted
        assert entry['replicates'][1]['eval_scores'] == {'Helpfulness': 3}
        assert entry['replicates'][2]['eval_scores'] == {'Helpfulness': 1}
        # Consistency should be recomputed
        assert entry['consistency']['scores'] == [2, 3, 1]

    def test_cannot_fix_empty_thought(self):
        """Test failure when a replicate has empty thought and no score."""
        entry = {
            'eval_id': 0,
            'replicates': [
                {'Evaluator Thought': '**Overall Qualitative Label**: Good (2)', 'eval_scores': {}},
                {'Evaluator Thought': '', 'eval_scores': {}},  # Empty thought, can't extract
                {'Evaluator Thought': '**Overall Qualitative Label**: Adequate (1)', 'eval_scores': {}},
            ],
            'consistency': None
        }

        result = fix_multi_replicate_entry(entry, 'agent_help')

        assert result == False
        # Entry should not be modified (or at least consistency not set)
        assert entry.get('consistency') is None or entry['consistency'] is None

    def test_cannot_fix_unparseable_thought(self):
        """Test failure when thought doesn't match any pattern."""
        entry = {
            'eval_id': 0,
            'replicates': [
                {'Evaluator Thought': '**Overall Qualitative Label**: Good (2)', 'eval_scores': {}},
                {'Evaluator Thought': 'Random text with no score', 'eval_scores': {}},  # No pattern
                {'Evaluator Thought': '**Overall Qualitative Label**: Adequate (1)', 'eval_scores': {}},
            ],
            'consistency': None
        }

        result = fix_multi_replicate_entry(entry, 'agent_help')

        assert result == False

    def test_empty_replicates_list(self):
        """Test failure when replicates list is empty."""
        entry = {
            'eval_id': 0,
            'replicates': [],
            'consistency': None
        }

        result = fix_multi_replicate_entry(entry, 'agent_help')

        assert result == False

    def test_all_scores_present_recomputes_consistency(self):
        """Test that consistency is recomputed even when all scores exist."""
        entry = {
            'eval_id': 0,
            'replicates': [
                {'Evaluator Thought': 'Thought 1', 'eval_scores': {'Helpfulness': 2}},
                {'Evaluator Thought': 'Thought 2', 'eval_scores': {'Helpfulness': 2}},
                {'Evaluator Thought': 'Thought 3', 'eval_scores': {'Helpfulness': 3}},
            ],
            'consistency': None  # Missing consistency but scores exist
        }

        result = fix_multi_replicate_entry(entry, 'agent_help')

        assert result == True
        assert entry['consistency'] is not None
        assert entry['consistency']['scores'] == [2, 2, 3]
        assert entry['consistency']['exact_match'] == False
        assert entry['consistency']['within_1'] == True


# =============================================================================
# Test File Processing
# =============================================================================

def create_test_trajectory(case_indices: List[int]) -> List[Dict[str, Any]]:
    """Create test trajectory data."""
    return [
        {
            'case_idx': idx,
            'case': {'Toolkits': [], 'User Instruction': f'Task {idx}'},
            'output': f'Output for case {idx}',
            'intermediate_steps': [],
            'input': f'Input for case {idx}'
        }
        for idx in case_indices
    ]


def create_test_eval_helpfulness(
    case_indices: List[int],
    broken_indices: List[int] = None,
    unfixable_indices: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Create test helpfulness evaluation data.

    Args:
        case_indices: List of case indices
        broken_indices: Indices where score is missing (but fixable)
        unfixable_indices: Indices where score is missing and unfixable
    """
    broken_indices = broken_indices or []
    unfixable_indices = unfixable_indices or []

    evals = []
    for i, idx in enumerate(case_indices):
        if i in unfixable_indices:
            # Unfixable: no thought or unparseable thought
            eval_data = {
                'eval_id': i,
                'Evaluator Thought': '',  # Empty thought
                'eval_scores': {}
            }
        elif i in broken_indices:
            # Broken but fixable: thought contains score
            eval_data = {
                'eval_id': i,
                'Evaluator Thought': f'Analysis for case {idx}.\n**Overall Qualitative Label**: Good (2)',
                'eval_scores': {}  # Missing score (parsing failure)
            }
        else:
            # Valid entry
            eval_data = {
                'eval_id': i,
                'Evaluator Thought': f'Analysis for case {idx}.',
                'eval_scores': {'Helpfulness': 2},
                'Overall Quantitative Score': 2
            }
        evals.append(eval_data)

    return evals


def create_test_eval_safety(
    case_indices: List[int],
    broken_indices: List[int] = None,
    unfixable_indices: List[int] = None
) -> List[Dict[str, Any]]:
    """Create test safety evaluation data."""
    broken_indices = broken_indices or []
    unfixable_indices = unfixable_indices or []

    evals = []
    for i, idx in enumerate(case_indices):
        if i in unfixable_indices:
            # Unfixable
            eval_data = {
                'eval_id': i,
                'Evaluator Thought': 'No pattern here',
                'eval_scores': {}
            }
        elif i in broken_indices:
            # Broken but fixable
            eval_data = {
                'eval_id': i,
                'Evaluator Thought': f'Safety check for case {idx}.\n**Label**: **Likely Mild Risk (2)**',
                'eval_scores': {}
            }
        else:
            # Valid entry
            eval_data = {
                'eval_id': i,
                'Evaluator Thought': f'Safety check for case {idx}.',
                'eval_scores': {'ToolCallRisk': 2},
                'Overall Quantitative Score': 2
            }
        evals.append(eval_data)

    return evals


# write_jsonl is defined in conftest.py and automatically available


class TestProcessEvalFile:
    """Tests for process_eval_file function."""

    def test_no_failures(self, temp_test_dir):
        """Test processing file with no failures."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create all 4 eval files (all valid)
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            if eval_type == 'agent_safe':
                eval_data = create_test_eval_safety(case_indices)
            else:
                eval_data = create_test_eval_helpfulness(case_indices)

            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        # Process helpfulness file
        result = process_eval_file(traj_file, 'agent_help', dry_run=False)

        assert result['total_entries'] == 3
        assert result['failures_found'] == 0
        assert result['fixed'] == 0
        assert result['unfixable'] == 0

    def test_fix_all_failures(self, temp_test_dir):
        """Test fixing all failures successfully."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval file with all entries broken but fixable
        eval_data = create_test_eval_helpfulness(case_indices, broken_indices=[0, 1, 2])

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_help':
                write_jsonl(str(eval_file), eval_data)
            else:
                # Create valid files for other types
                if eval_type == 'agent_safe':
                    write_jsonl(str(eval_file), create_test_eval_safety(case_indices))
                else:
                    write_jsonl(str(eval_file), create_test_eval_helpfulness(case_indices))

        # Process file
        result = process_eval_file(traj_file, 'agent_help', dry_run=False)

        assert result['total_entries'] == 3
        assert result['failures_found'] == 3
        assert result['fixed'] == 3
        assert result['unfixable'] == 0
        assert result['revalidated'] == True

    def test_some_unfixable(self, temp_test_dir):
        """Test with mix of fixable and unfixable failures."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval file: entry 0 fixable, entry 1 unfixable, entry 2 valid
        eval_data = create_test_eval_helpfulness(
            case_indices,
            broken_indices=[0],
            unfixable_indices=[1]
        )

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_help':
                write_jsonl(str(eval_file), eval_data)
            else:
                # Create valid files for other types
                if eval_type == 'agent_safe':
                    write_jsonl(str(eval_file), create_test_eval_safety(case_indices))
                else:
                    write_jsonl(str(eval_file), create_test_eval_helpfulness(case_indices))

        # Process file
        result = process_eval_file(traj_file, 'agent_help', dry_run=False)

        assert result['total_entries'] == 3
        assert result['failures_found'] == 2
        assert result['fixed'] == 1
        assert result['unfixable'] == 1
        assert len(result['unfixable_details']) == 1
        assert result['unfixable_details'][0]['reason'] == 'Empty Evaluator Thought'

    def test_dry_run_no_write(self, temp_test_dir):
        """Test dry run mode doesn't write files."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval file with failures
        eval_data = create_test_eval_helpfulness(case_indices, broken_indices=[0, 1, 2])

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_help':
                write_jsonl(str(eval_file), eval_data)
            else:
                if eval_type == 'agent_safe':
                    write_jsonl(str(eval_file), create_test_eval_safety(case_indices))
                else:
                    write_jsonl(str(eval_file), create_test_eval_helpfulness(case_indices))

        eval_file_path = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))

        # Get original content
        with open(eval_file_path, 'r') as f:
            original_content = f.read()

        # Process in dry run
        result = process_eval_file(traj_file, 'agent_help', dry_run=True)

        # Verify stats
        assert result['fixed'] == 3
        assert result['dry_run'] == True

        # Verify file not modified
        with open(eval_file_path, 'r') as f:
            after_content = f.read()

        assert original_content == after_content

    def test_safety_eval_fixing(self, temp_test_dir):
        """Test fixing safety evaluations (uses ToolCallRisk not Helpfulness)."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval file with safety failures
        eval_data = create_test_eval_safety(case_indices, broken_indices=[0, 1, 2])

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_safe':
                write_jsonl(str(eval_file), eval_data)
            else:
                write_jsonl(str(eval_file), create_test_eval_helpfulness(case_indices))

        # Process safety file
        result = process_eval_file(traj_file, 'agent_safe', dry_run=False)

        assert result['total_entries'] == 3
        assert result['failures_found'] == 3
        assert result['fixed'] == 3
        assert result['unfixable'] == 0

        # Verify it used ToolCallRisk
        eval_file_path = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_safe'))
        with open(eval_file_path, 'r') as f:
            fixed_data = [json.loads(line) for line in f]

        for entry in fixed_data:
            assert 'ToolCallRisk' in entry['eval_scores']
            assert entry['eval_scores']['ToolCallRisk'] == 2

    def test_file_not_found(self, temp_test_dir):
        """Test handling of missing eval file."""
        tmpdir = temp_test_dir

        # Create trajectory file only
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory([10, 11, 12]))

        # Don't create eval file
        result = process_eval_file(traj_file, 'agent_help', dry_run=False)

        assert result['skipped'] == True
        assert result['reason'] == 'file not found'


# =============================================================================
# Multi-Replicate File Processing Tests
# =============================================================================

def create_multi_replicate_eval(
    case_indices: List[int],
    num_replicates: int = 3,
    broken_entry_indices: List[int] = None,
    unfixable_entry_indices: List[int] = None,
    eval_type: str = 'agent_help'
) -> List[Dict[str, Any]]:
    """
    Create multi-replicate evaluation data.

    Args:
        case_indices: List of case indices
        num_replicates: Number of replicates per entry
        broken_entry_indices: Entry indices where some/all replicates lack scores (but fixable)
        unfixable_entry_indices: Entry indices where replicates can't be fixed
        eval_type: 'agent_help' or 'agent_safe'
    """
    broken_entry_indices = broken_entry_indices or []
    unfixable_entry_indices = unfixable_entry_indices or []

    score_key = 'ToolCallRisk' if eval_type == 'agent_safe' else 'Helpfulness'

    evals = []
    for i, idx in enumerate(case_indices):
        replicates = []

        for rep_idx in range(num_replicates):
            if i in unfixable_entry_indices:
                # Unfixable: empty thought, no score
                rep = {
                    'Evaluator Thought': '',
                    'eval_scores': {}
                }
            elif i in broken_entry_indices:
                # Broken but fixable: thought contains extractable score
                if eval_type == 'agent_safe':
                    thought = f'Safety analysis for case {idx} rep {rep_idx}.\n**Label**: **Likely Mild Risk (2)**'
                else:
                    thought = f'Analysis for case {idx} rep {rep_idx}.\n**Overall Qualitative Label**: Good (2)'
                rep = {
                    'Evaluator Thought': thought,
                    'eval_scores': {}  # Missing score (parsing failure)
                }
            else:
                # Valid entry with score
                rep = {
                    'Evaluator Thought': f'Analysis for case {idx} rep {rep_idx}.',
                    'eval_scores': {score_key: 2}
                }
            replicates.append(rep)

        # Build consistency if all scores present
        if i not in broken_entry_indices and i not in unfixable_entry_indices:
            scores = [2] * num_replicates
            consistency = compute_eval_consistency(scores)
        else:
            consistency = None

        eval_data = {
            'eval_id': i,
            'replicates': replicates,
            'consistency': consistency
        }
        evals.append(eval_data)

    return evals


class TestProcessMultiReplicateEvalFile:
    """Tests for process_eval_file with multi-replicate format."""

    def test_no_failures_multi_rep(self, temp_test_dir):
        """Test processing multi-replicate file with no failures."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create all eval files (multi-replicate format, all valid)
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_data = create_multi_replicate_eval(case_indices, eval_type=eval_type)
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        # Process helpfulness file
        result = process_eval_file(traj_file, 'agent_help', dry_run=False)

        assert result['total_entries'] == 3
        assert result['failures_found'] == 0
        assert result['fixed'] == 0
        assert result['unfixable'] == 0

    def test_fix_all_multi_rep_failures(self, temp_test_dir):
        """Test fixing all multi-replicate failures."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval file with all entries broken but fixable
        eval_data = create_multi_replicate_eval(
            case_indices,
            broken_entry_indices=[0, 1, 2],
            eval_type='agent_help'
        )

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_help':
                write_jsonl(str(eval_file), eval_data)
            else:
                # Create valid files for other types
                write_jsonl(str(eval_file), create_multi_replicate_eval(case_indices, eval_type=eval_type))

        # Process file
        result = process_eval_file(traj_file, 'agent_help', dry_run=False)

        assert result['total_entries'] == 3
        assert result['failures_found'] == 3
        assert result['fixed'] == 3
        assert result['unfixable'] == 0

        # Verify the file was updated correctly
        eval_file_path = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        with open(eval_file_path, 'r') as f:
            fixed_data = [json.loads(line) for line in f]

        for entry in fixed_data:
            assert 'replicates' in entry
            assert entry['consistency'] is not None
            assert entry['consistency']['scores'] == [2, 2, 2]
            for rep in entry['replicates']:
                assert rep['eval_scores']['Helpfulness'] == 2

    def test_some_unfixable_multi_rep(self, temp_test_dir):
        """Test with mix of fixable and unfixable multi-replicate failures."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval file: entry 0 fixable, entry 1 unfixable, entry 2 valid
        eval_data = create_multi_replicate_eval(
            case_indices,
            broken_entry_indices=[0],
            unfixable_entry_indices=[1],
            eval_type='agent_help'
        )

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_help':
                write_jsonl(str(eval_file), eval_data)
            else:
                write_jsonl(str(eval_file), create_multi_replicate_eval(case_indices, eval_type=eval_type))

        # Process file
        result = process_eval_file(traj_file, 'agent_help', dry_run=False)

        assert result['total_entries'] == 3
        assert result['failures_found'] == 2
        assert result['fixed'] == 1
        assert result['unfixable'] == 1
        assert len(result['unfixable_details']) == 1
        assert 'Multi-replicate' in result['unfixable_details'][0]['reason']

    def test_safety_multi_rep_fixing(self, temp_test_dir):
        """Test fixing multi-replicate safety evaluations."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval file with safety failures (multi-rep)
        eval_data = create_multi_replicate_eval(
            case_indices,
            broken_entry_indices=[0, 1, 2],
            eval_type='agent_safe'
        )

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_safe':
                write_jsonl(str(eval_file), eval_data)
            else:
                write_jsonl(str(eval_file), create_multi_replicate_eval(case_indices, eval_type=eval_type))

        # Process safety file
        result = process_eval_file(traj_file, 'agent_safe', dry_run=False)

        assert result['total_entries'] == 3
        assert result['failures_found'] == 3
        assert result['fixed'] == 3
        assert result['unfixable'] == 0

        # Verify ToolCallRisk was used
        eval_file_path = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_safe'))
        with open(eval_file_path, 'r') as f:
            fixed_data = [json.loads(line) for line in f]

        for entry in fixed_data:
            for rep in entry['replicates']:
                assert 'ToolCallRisk' in rep['eval_scores']
                assert rep['eval_scores']['ToolCallRisk'] == 2

    def test_dry_run_multi_rep_no_write(self, temp_test_dir):
        """Test dry run mode doesn't write multi-replicate files."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create files
        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval file with failures
        eval_data = create_multi_replicate_eval(
            case_indices,
            broken_entry_indices=[0, 1, 2],
            eval_type='agent_help'
        )

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_help':
                write_jsonl(str(eval_file), eval_data)
            else:
                write_jsonl(str(eval_file), create_multi_replicate_eval(case_indices, eval_type=eval_type))

        eval_file_path = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))

        # Get original content
        with open(eval_file_path, 'r') as f:
            original_content = f.read()

        # Process in dry run
        result = process_eval_file(traj_file, 'agent_help', dry_run=True)

        # Verify stats
        assert result['fixed'] == 3
        assert result['dry_run'] == True

        # Verify file not modified
        with open(eval_file_path, 'r') as f:
            after_content = f.read()

        assert original_content == after_content


# =============================================================================
# Test Parse Arguments
# =============================================================================

class TestParseArguments:
    """Tests for parse_arguments function."""

    def test_default_values(self, monkeypatch):
        """Test default argument values."""
        monkeypatch.setattr(sys, 'argv', ['fix_eval_parsing.py'])
        args = parse_arguments()

        assert args.data_dir == Path('output/trajectories')
        assert args.model_filter is None
        assert args.eval_types == ToolEmuFilePaths.EVAL_TYPES
        assert args.dry_run == False
        assert args.show_unfixable == False
        assert args.delete_unfixable == False

    def test_custom_data_dir(self, monkeypatch):
        """Test custom data directory."""
        monkeypatch.setattr(sys, 'argv', ['fix_eval_parsing.py', '--data-dir', '/custom/path'])
        args = parse_arguments()

        assert args.data_dir == Path('/custom/path')

    def test_model_filter(self, monkeypatch):
        """Test model filter argument."""
        monkeypatch.setattr(sys, 'argv', ['fix_eval_parsing.py', '--model-filter', 'Qwen_Qwen3-8B'])
        args = parse_arguments()

        assert args.model_filter == 'Qwen_Qwen3-8B'

    def test_specific_eval_types(self, monkeypatch):
        """Test selecting specific eval types."""
        monkeypatch.setattr(sys, 'argv', ['fix_eval_parsing.py', '--eval-types', 'agent_help', 'agent_safe'])
        args = parse_arguments()

        assert args.eval_types == ['agent_help', 'agent_safe']

    def test_dry_run_flag(self, monkeypatch):
        """Test dry run flag."""
        monkeypatch.setattr(sys, 'argv', ['fix_eval_parsing.py', '--dry-run'])
        args = parse_arguments()

        assert args.dry_run == True

    def test_show_unfixable_flag(self, monkeypatch):
        """Test show unfixable flag."""
        monkeypatch.setattr(sys, 'argv', ['fix_eval_parsing.py', '--show-unfixable'])
        args = parse_arguments()

        assert args.show_unfixable == True

    def test_delete_unfixable_flag(self, monkeypatch):
        """Test delete unfixable flag."""
        monkeypatch.setattr(sys, 'argv', ['fix_eval_parsing.py', '--delete-unfixable'])
        args = parse_arguments()

        assert args.delete_unfixable == True

    def test_combined_flags(self, monkeypatch):
        """Test multiple flags together."""
        monkeypatch.setattr(sys, 'argv', [
            'fix_eval_parsing.py',
            '--data-dir', '/tmp/test',
            '--model-filter', 'test_model',
            '--eval-types', 'agent_safe',
            '--dry-run',
            '--show-unfixable',
            '--delete-unfixable'
        ])
        args = parse_arguments()

        assert args.data_dir == Path('/tmp/test')
        assert args.model_filter == 'test_model'
        assert args.eval_types == ['agent_safe']
        assert args.dry_run == True
        assert args.show_unfixable == True
        assert args.delete_unfixable == True


# =============================================================================
# Test Process All Files
# =============================================================================

class TestProcessAllFiles:
    """Tests for process_all_files function."""

    def test_empty_file_list(self):
        """Test with no files to process."""
        global_stats, stats_by_eval_type, files_modified, unfixable_files = process_all_files(
            trajectory_files=[],
            eval_types=['agent_help'],
            model_filter=None,
            dry_run=False
        )

        assert global_stats['files_processed'] == 0
        assert global_stats['total_failures'] == 0
        assert files_modified == []
        assert unfixable_files == {}

    def test_model_filter_excludes_files(self, temp_test_dir):
        """Test that model filter excludes non-matching files."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create trajectory file that won't match filter
        traj_file = Path(tmpdir) / "other_model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval files
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            if eval_type == 'agent_safe':
                eval_data = create_test_eval_safety(case_indices)
            else:
                eval_data = create_test_eval_helpfulness(case_indices)
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        global_stats, stats_by_eval_type, files_modified, unfixable_files = process_all_files(
            trajectory_files=[traj_file],
            eval_types=['agent_help'],
            model_filter='Qwen_Qwen3',  # Won't match "other_model"
            dry_run=False
        )

        # File should be skipped due to filter
        assert global_stats['files_processed'] == 0

    def test_process_single_file_no_failures(self, temp_test_dir):
        """Test processing a single file with no failures."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            if eval_type == 'agent_safe':
                eval_data = create_test_eval_safety(case_indices)
            else:
                eval_data = create_test_eval_helpfulness(case_indices)
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        global_stats, stats_by_eval_type, files_modified, unfixable_files = process_all_files(
            trajectory_files=[traj_file],
            eval_types=['agent_help'],
            model_filter=None,
            dry_run=False
        )

        assert global_stats['files_processed'] == 1
        assert global_stats['total_failures'] == 0
        assert global_stats['total_fixed'] == 0
        assert files_modified == []

    def test_process_file_with_failures(self, temp_test_dir):
        """Test processing a file with fixable failures."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create eval file with broken entries
        eval_data = create_test_eval_helpfulness(case_indices, broken_indices=[0, 1])

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_help':
                write_jsonl(str(eval_file), eval_data)
            else:
                if eval_type == 'agent_safe':
                    write_jsonl(str(eval_file), create_test_eval_safety(case_indices))
                else:
                    write_jsonl(str(eval_file), create_test_eval_helpfulness(case_indices))

        global_stats, stats_by_eval_type, files_modified, unfixable_files = process_all_files(
            trajectory_files=[traj_file],
            eval_types=['agent_help'],
            model_filter=None,
            dry_run=False
        )

        assert global_stats['files_processed'] == 1
        assert global_stats['total_failures'] == 2
        assert global_stats['total_fixed'] == 2
        assert global_stats['files_modified'] == 1
        assert len(files_modified) == 1

    def test_process_multiple_eval_types(self, temp_test_dir):
        """Test processing multiple eval types."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        traj_file = Path(tmpdir) / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_test_trajectory(case_indices))

        # Create files with failures in different eval types
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_safe':
                eval_data = create_test_eval_safety(case_indices, broken_indices=[0])
            else:
                eval_data = create_test_eval_helpfulness(case_indices, broken_indices=[1])
            write_jsonl(str(eval_file), eval_data)

        global_stats, stats_by_eval_type, files_modified, unfixable_files = process_all_files(
            trajectory_files=[traj_file],
            eval_types=['agent_help', 'agent_safe'],
            model_filter=None,
            dry_run=False
        )

        assert stats_by_eval_type['agent_help']['fixed'] == 1
        assert stats_by_eval_type['agent_safe']['fixed'] == 1
        assert global_stats['total_fixed'] == 2


# =============================================================================
# Test Print Functions
# =============================================================================

class TestPrintResultsByEvalType:
    """Tests for print_results_by_eval_type function."""

    def test_prints_all_eval_types(self, capsys):
        """Test that all eval types are printed."""
        eval_types = ['agent_help', 'agent_safe']
        stats_by_eval_type = {
            'agent_help': {'files_processed': 10, 'failures': 5, 'fixed': 4, 'unfixable': 1},
            'agent_safe': {'files_processed': 10, 'failures': 3, 'fixed': 3, 'unfixable': 0},
        }

        print_results_by_eval_type(eval_types, stats_by_eval_type)

        captured = capsys.readouterr()
        assert 'agent_help:' in captured.out
        assert 'agent_safe:' in captured.out
        assert 'Files processed: 10' in captured.out
        assert 'Failures found: 5' in captured.out
        assert 'Fixed: 4' in captured.out
        assert 'Unfixable: 1' in captured.out

    def test_success_rate_calculation(self, capsys):
        """Test that success rate is calculated correctly."""
        eval_types = ['agent_help']
        stats_by_eval_type = {
            'agent_help': {'files_processed': 10, 'failures': 10, 'fixed': 8, 'unfixable': 2},
        }

        print_results_by_eval_type(eval_types, stats_by_eval_type)

        captured = capsys.readouterr()
        assert 'Success rate: 80.0%' in captured.out

    def test_no_success_rate_when_no_failures(self, capsys):
        """Test that success rate is not printed when no failures."""
        eval_types = ['agent_help']
        stats_by_eval_type = {
            'agent_help': {'files_processed': 10, 'failures': 0, 'fixed': 0, 'unfixable': 0},
        }

        print_results_by_eval_type(eval_types, stats_by_eval_type)

        captured = capsys.readouterr()
        assert 'Success rate' not in captured.out


class TestPrintOverallSummary:
    """Tests for print_overall_summary function."""

    def test_prints_summary_stats(self, capsys):
        """Test that summary stats are printed."""
        global_stats = {
            'files_processed': 20,
            'files_modified': 5,
            'total_failures': 100,
            'total_fixed': 90,
            'total_unfixable': 10,
        }

        print_overall_summary(global_stats, dry_run=False)

        captured = capsys.readouterr()
        assert 'Trajectory files processed: 20' in captured.out
        assert 'Files with fixes: 5' in captured.out
        assert 'Total parsing failures found: 100' in captured.out
        assert 'Total fixed: 90' in captured.out
        assert 'Total unfixable: 10' in captured.out
        assert 'Overall success rate: 90.0%' in captured.out

    def test_dry_run_message(self, capsys):
        """Test that dry run message is shown."""
        global_stats = {
            'files_processed': 20,
            'files_modified': 5,
            'total_failures': 10,
            'total_fixed': 10,
            'total_unfixable': 0,
        }

        print_overall_summary(global_stats, dry_run=True)

        captured = capsys.readouterr()
        assert 'DRY RUN: Would modify 5 files' in captured.out

    def test_no_dry_run_message_when_no_modifications(self, capsys):
        """Test that dry run message is not shown when no modifications."""
        global_stats = {
            'files_processed': 20,
            'files_modified': 0,
            'total_failures': 0,
            'total_fixed': 0,
            'total_unfixable': 0,
        }

        print_overall_summary(global_stats, dry_run=True)

        captured = capsys.readouterr()
        assert 'DRY RUN' not in captured.out


class TestHandleUnfixableFiles:
    """Tests for handle_unfixable_files function."""

    def test_no_unfixable_files(self, capsys):
        """Test with no unfixable files."""
        handle_unfixable_files(unfixable_files={}, delete_unfixable=False, dry_run=False)

        captured = capsys.readouterr()
        assert captured.out == ''

    def test_prints_unfixable_files(self, capsys, temp_test_dir):
        """Test that unfixable files are printed."""
        tmpdir = temp_test_dir
        eval_file1 = Path(tmpdir) / "test1_eval_agent_help.jsonl"
        eval_file2 = Path(tmpdir) / "test2_eval_agent_safe.jsonl"

        unfixable_files = {
            (eval_file1, 'agent_help'): 3,
            (eval_file2, 'agent_safe'): 2,
        }

        handle_unfixable_files(unfixable_files, delete_unfixable=False, dry_run=False)

        captured = capsys.readouterr()
        assert 'FILES WITH UNFIXABLE ENTRIES (2 files)' in captured.out
        assert 'agent_help:' in captured.out
        assert 'agent_safe:' in captured.out
        assert '(3 unfixable)' in captured.out
        assert '(2 unfixable)' in captured.out

    def test_delete_unfixable_dry_run(self, capsys, temp_test_dir):
        """Test delete unfixable in dry run mode."""
        tmpdir = temp_test_dir
        eval_file = Path(tmpdir) / "test_eval_agent_help.jsonl"
        eval_file.write_text('{}')

        unfixable_files = {(eval_file, 'agent_help'): 1}

        handle_unfixable_files(unfixable_files, delete_unfixable=True, dry_run=True)

        captured = capsys.readouterr()
        assert 'Would delete' in captured.out
        assert eval_file.exists()  # File should not be deleted in dry run

    def test_delete_unfixable_actual(self, capsys, temp_test_dir):
        """Test actual deletion of unfixable files."""
        tmpdir = temp_test_dir
        eval_file = Path(tmpdir) / "test_eval_agent_help.jsonl"
        eval_file.write_text('{}')

        assert eval_file.exists()

        unfixable_files = {(eval_file, 'agent_help'): 1}

        handle_unfixable_files(unfixable_files, delete_unfixable=True, dry_run=False)

        captured = capsys.readouterr()
        assert 'DELETING 1 FILES' in captured.out
        assert 'Successfully deleted: 1/1' in captured.out
        assert not eval_file.exists()  # File should be deleted


class TestPrintModifiedFiles:
    """Tests for print_modified_files function."""

    def test_no_modified_files(self, capsys):
        """Test with no modified files."""
        print_modified_files(files_modified=[], dry_run=False)

        captured = capsys.readouterr()
        assert captured.out == ''

    def test_dry_run_no_output(self, capsys):
        """Test that dry run mode produces no output."""
        print_modified_files(files_modified=['file1.jsonl', 'file2.jsonl'], dry_run=True)

        captured = capsys.readouterr()
        assert captured.out == ''

    def test_prints_modified_files(self, capsys):
        """Test that modified files are printed."""
        files_modified = ['file1.jsonl', 'file2.jsonl', 'file3.jsonl']

        print_modified_files(files_modified, dry_run=False)

        captured = capsys.readouterr()
        assert 'MODIFIED FILES (3)' in captured.out
        assert 'file1.jsonl' in captured.out
        assert 'file2.jsonl' in captured.out
        assert 'file3.jsonl' in captured.out

    def test_truncates_long_list(self, capsys):
        """Test that long lists are truncated to 20 files."""
        files_modified = [f'file{i}.jsonl' for i in range(25)]

        print_modified_files(files_modified, dry_run=False)

        captured = capsys.readouterr()
        assert 'MODIFIED FILES (25)' in captured.out
        assert 'file19.jsonl' in captured.out
        assert 'file20.jsonl' not in captured.out  # Should be truncated
        assert '... and 5 more' in captured.out


class TestPrintUnfixableDetails:
    """Tests for print_unfixable_details function."""

    def test_no_unfixable(self, capsys):
        """Test with no unfixable entries."""
        print_unfixable_details(
            eval_types=['agent_help'],
            stats_by_eval_type={'agent_help': {'unfixable_details': []}},
            total_unfixable=0
        )

        captured = capsys.readouterr()
        assert captured.out == ''

    def test_prints_unfixable_details(self, capsys):
        """Test that unfixable details are printed."""
        stats_by_eval_type = {
            'agent_help': {
                'unfixable_details': [
                    {'line': 1, 'eval_id': 0, 'case_idx': 10, 'reason': 'Empty thought', 'snippet': 'Some text...'},
                    {'line': 2, 'eval_id': 1, 'case_idx': 11, 'reason': 'No pattern match', 'snippet': 'Other text...'},
                ]
            }
        }

        print_unfixable_details(
            eval_types=['agent_help'],
            stats_by_eval_type=stats_by_eval_type,
            total_unfixable=2
        )

        captured = capsys.readouterr()
        assert 'UNFIXABLE ENTRIES (2 total)' in captured.out
        assert 'Eval type: agent_help' in captured.out
        assert 'Line: 1' in captured.out
        assert 'Reason: Empty thought' in captured.out

    def test_limits_entries_per_eval_type(self, capsys):
        """Test that output is limited to 5 entries per eval type."""
        # The function shows up to 5 entries per eval type (and 15 total across all types)
        details = [
            {'line': i, 'eval_id': i, 'case_idx': i, 'reason': f'Reason {i}', 'snippet': f'Snippet {i}'}
            for i in range(20)
        ]
        stats_by_eval_type = {'agent_help': {'unfixable_details': details}}

        print_unfixable_details(
            eval_types=['agent_help'],
            stats_by_eval_type=stats_by_eval_type,
            total_unfixable=20
        )

        captured = capsys.readouterr()
        # Only 5 shown (limit per eval type), so 15 more remain
        assert '... and 15 more unfixable entries' in captured.out
        # Verify only first 5 are shown (indices 0-4)
        assert 'Line: 4' in captured.out
        assert 'Line: 5' not in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
