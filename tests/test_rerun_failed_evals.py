#!/usr/bin/env python3
"""
Unit tests for slurm_scripts/rerun_failed_evals.py

Tests the find_unfixable_entries(), merge_results_in_place(), and retry logic.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any

# Add src and slurm_scripts directories to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'slurm_scripts'))

from utils.toolemu_utils import (
    ToolEmuFilePaths,
    find_unfixable_entries,
    compute_eval_consistency,
)

# Import shared test helpers from conftest
from conftest import write_jsonl, create_trajectory_data


# =============================================================================
# Test Data Helpers
# =============================================================================

def create_eval_data_with_failures(
    num_entries: int,
    eval_type: str,
    missing_score_indices: List[int] = None,
    empty_thought_indices: List[int] = None,
) -> List[Dict[str, Any]]:
    """
    Create eval data with some entries missing scores.

    Args:
        num_entries: Number of eval entries to create
        eval_type: 'agent_safe' or 'agent_help*'
        missing_score_indices: Line indices where score is missing but thought has extractable score
        empty_thought_indices: Line indices where thought is empty (truly unfixable)
    """
    missing_score_indices = missing_score_indices or []
    empty_thought_indices = empty_thought_indices or []

    score_key = 'ToolCallRisk' if eval_type == 'agent_safe' else 'Helpfulness'

    evals = []
    for i in range(num_entries):
        if i in empty_thought_indices:
            # Truly unfixable: empty thought, no score
            entry = {
                'eval_id': i,
                'Evaluator Thought': '',
                'eval_scores': {}
            }
        elif i in missing_score_indices:
            # Missing score but thought has extractable pattern
            if eval_type == 'agent_safe':
                thought = f'Analysis for entry {i}.\n**Label**: **Likely Mild Risk (2)**'
            else:
                thought = f'Analysis for entry {i}.\n**Overall Qualitative Label**: Good (2)'
            entry = {
                'eval_id': i,
                'Evaluator Thought': thought,
                'eval_scores': {}  # Missing score
            }
        else:
            # Valid entry with score
            entry = {
                'eval_id': i,
                'Evaluator Thought': f'Valid analysis for entry {i}.',
                'eval_scores': {score_key: 2},
                'Overall Quantitative Score': 2
            }
        evals.append(entry)

    return evals


def create_multi_rep_eval_data_with_failures(
    num_entries: int,
    num_replicates: int,
    eval_type: str,
    missing_score_indices: List[int] = None,
    empty_thought_indices: List[int] = None,
) -> List[Dict[str, Any]]:
    """Create multi-replicate eval data with some entries missing scores."""
    missing_score_indices = missing_score_indices or []
    empty_thought_indices = empty_thought_indices or []

    score_key = 'ToolCallRisk' if eval_type == 'agent_safe' else 'Helpfulness'

    evals = []
    for i in range(num_entries):
        replicates = []

        for rep in range(num_replicates):
            if i in empty_thought_indices:
                # Truly unfixable
                rep_data = {'Evaluator Thought': '', 'eval_scores': {}}
            elif i in missing_score_indices:
                # Missing score but extractable
                if eval_type == 'agent_safe':
                    thought = f'Rep {rep} analysis.\n**Label**: **Likely Mild Risk (2)**'
                else:
                    thought = f'Rep {rep} analysis.\n**Overall Qualitative Label**: Good (2)'
                rep_data = {'Evaluator Thought': thought, 'eval_scores': {}}
            else:
                # Valid
                rep_data = {
                    'Evaluator Thought': f'Valid rep {rep}.',
                    'eval_scores': {score_key: 2}
                }
            replicates.append(rep_data)

        # Compute consistency for valid entries
        if i not in missing_score_indices and i not in empty_thought_indices:
            consistency = compute_eval_consistency([2] * num_replicates)
        else:
            consistency = None

        evals.append({
            'eval_id': i,
            'replicates': replicates,
            'consistency': consistency
        })

    return evals


# =============================================================================
# Test find_unfixable_entries - Single Replicate Format
# =============================================================================

class TestFindUnfixableEntriesSingleRep:
    """Tests for find_unfixable_entries with single-replicate eval files."""

    def test_all_valid_entries(self, temp_test_dir):
        """Test with all valid entries - should return empty list."""
        case_indices = [10, 11, 12]

        # Create trajectory file
        traj_file = temp_test_dir / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # Create valid eval file
        eval_data = create_eval_data_with_failures(3, 'agent_help')
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        assert result == []

    def test_all_unfixable_entries(self, temp_test_dir):
        """Test with all entries unfixable (empty thoughts)."""
        case_indices = [10, 11, 12]

        traj_file = temp_test_dir / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # All entries have empty thoughts
        eval_data = create_eval_data_with_failures(3, 'agent_help', empty_thought_indices=[0, 1, 2])
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        # Should return all 3 as unfixable: (line_index, case_idx) pairs
        assert len(result) == 3
        assert result[0] == (0, 10)
        assert result[1] == (1, 11)
        assert result[2] == (2, 12)

    def test_mixed_entries(self, temp_test_dir):
        """Test with mix of valid, fixable, and unfixable entries."""
        case_indices = [10, 11, 12, 13]

        traj_file = temp_test_dir / "model_r10-14_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # Entry 0: valid, Entry 1: fixable (has pattern), Entry 2: unfixable, Entry 3: valid
        eval_data = create_eval_data_with_failures(
            4, 'agent_help',
            missing_score_indices=[1],  # Fixable - has extractable pattern
            empty_thought_indices=[2]   # Unfixable - empty thought
        )
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        # Only entry 2 is truly unfixable (entry 1 is fixable via regex)
        assert len(result) == 1
        assert result[0] == (2, 12)

    def test_safety_eval_type(self, temp_test_dir):
        """Test with safety evaluation type (uses ToolCallRisk key)."""
        case_indices = [5, 6, 7]

        traj_file = temp_test_dir / "model_r5-8_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # Entry 1 is unfixable
        eval_data = create_eval_data_with_failures(
            3, 'agent_safe',
            empty_thought_indices=[1]
        )
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            if eval_type == 'agent_safe':
                write_jsonl(str(eval_file), eval_data)
            else:
                write_jsonl(str(eval_file), create_eval_data_with_failures(3, eval_type))

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_safe'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_safe')

        assert len(result) == 1
        assert result[0] == (1, 6)

    def test_non_contiguous_case_indices(self, temp_test_dir):
        """Test with non-contiguous case indices (e.g., from a split)."""
        case_indices = [0, 5, 10, 15]  # Non-contiguous

        traj_file = temp_test_dir / "model_r0-16_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # Entries 1 and 3 are unfixable
        eval_data = create_eval_data_with_failures(
            4, 'agent_help',
            empty_thought_indices=[1, 3]
        )
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        # Should return correct case_idx values (not line indices)
        assert len(result) == 2
        assert result[0] == (1, 5)   # line 1, case_idx 5
        assert result[1] == (3, 15)  # line 3, case_idx 15


# =============================================================================
# Test find_unfixable_entries - Multi-Replicate Format
# =============================================================================

class TestFindUnfixableEntriesMultiRep:
    """Tests for find_unfixable_entries with multi-replicate eval files."""

    def test_all_valid_multi_rep(self, temp_test_dir):
        """Test with all valid multi-replicate entries."""
        case_indices = [10, 11, 12]

        traj_file = temp_test_dir / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        eval_data = create_multi_rep_eval_data_with_failures(3, 3, 'agent_help')
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        assert result == []

    def test_unfixable_multi_rep(self, temp_test_dir):
        """Test with unfixable multi-replicate entries (empty thoughts in all reps)."""
        case_indices = [10, 11, 12]

        traj_file = temp_test_dir / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # Entry 1 has empty thoughts in all replicates
        eval_data = create_multi_rep_eval_data_with_failures(
            3, 3, 'agent_help',
            empty_thought_indices=[1]
        )
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        assert len(result) == 1
        assert result[0] == (1, 11)

    def test_missing_consistency_but_fixable(self, temp_test_dir):
        """Test entry with missing consistency but extractable scores (fixable)."""
        case_indices = [10, 11]

        traj_file = temp_test_dir / "model_r10-12_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # Entry 1 has missing scores but extractable pattern
        eval_data = create_multi_rep_eval_data_with_failures(
            2, 3, 'agent_help',
            missing_score_indices=[1]  # Fixable
        )
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        # Entry 1 is fixable via regex, so should not be in unfixable list
        assert result == []


# =============================================================================
# Test find_unfixable_entries - Edge Cases
# =============================================================================

class TestFindUnfixableEntriesEdgeCases:
    """Edge case tests for find_unfixable_entries."""

    def test_empty_files(self, temp_test_dir):
        """Test with empty trajectory and eval files."""
        traj_file = temp_test_dir / "model_r0-0_test.jsonl"
        write_jsonl(str(traj_file), [])

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), [])

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        assert result == []

    def test_single_entry_valid(self, temp_test_dir):
        """Test with single valid entry."""
        case_indices = [42]

        traj_file = temp_test_dir / "model_r42-43_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        eval_data = create_eval_data_with_failures(1, 'agent_help')
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        assert result == []

    def test_single_entry_unfixable(self, temp_test_dir):
        """Test with single unfixable entry."""
        case_indices = [42]

        traj_file = temp_test_dir / "model_r42-43_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        eval_data = create_eval_data_with_failures(1, 'agent_help', empty_thought_indices=[0])
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))
        result = find_unfixable_entries(eval_file, traj_file, 'agent_help')

        assert len(result) == 1
        assert result[0] == (0, 42)

    def test_eval_file_not_found(self, temp_test_dir):
        """Test with missing eval file."""
        case_indices = [10, 11]

        traj_file = temp_test_dir / "model_r10-12_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # Don't create eval file
        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))

        with pytest.raises(FileNotFoundError):
            find_unfixable_entries(eval_file, traj_file, 'agent_help')

    def test_trajectory_file_not_found(self, temp_test_dir):
        """Test with missing trajectory file."""
        traj_file = temp_test_dir / "nonexistent.jsonl"
        eval_file = temp_test_dir / "nonexistent_eval_agent_help.jsonl"

        with pytest.raises(FileNotFoundError):
            find_unfixable_entries(eval_file, traj_file, 'agent_help')


# =============================================================================
# Test merge_results_in_place
# =============================================================================

class TestMergeResultsInPlace:
    """Tests for merge_results_in_place function."""

    def test_replace_single_entry(self, temp_test_dir):
        """Test replacing a single entry in the file."""
        # Import the function (will be added later)
        from rerun_failed_evals import merge_results_in_place

        # Create original eval file with 3 entries
        eval_file = temp_test_dir / "test_eval.jsonl"
        original_data = [
            {'eval_id': 0, 'eval_scores': {'Helpfulness': 2}},
            {'eval_id': 1, 'eval_scores': {}},  # Missing score
            {'eval_id': 2, 'eval_scores': {'Helpfulness': 3}},
        ]
        write_jsonl(str(eval_file), original_data)

        # New result for line 1
        new_results = {
            1: {'eval_id': 1, 'eval_scores': {'Helpfulness': 2}, 'fixed': True}
        }

        merge_results_in_place(eval_file, new_results)

        # Verify the file was updated
        with open(eval_file, 'r') as f:
            updated_data = [json.loads(line) for line in f]

        assert len(updated_data) == 3
        assert updated_data[0] == original_data[0]  # Unchanged
        assert updated_data[1]['eval_scores'] == {'Helpfulness': 2}
        assert updated_data[1]['fixed'] == True
        assert updated_data[2] == original_data[2]  # Unchanged

    def test_replace_multiple_entries(self, temp_test_dir):
        """Test replacing multiple entries."""
        from rerun_failed_evals import merge_results_in_place

        eval_file = temp_test_dir / "test_eval.jsonl"
        original_data = [
            {'eval_id': 0, 'eval_scores': {}},
            {'eval_id': 1, 'eval_scores': {'Helpfulness': 2}},
            {'eval_id': 2, 'eval_scores': {}},
            {'eval_id': 3, 'eval_scores': {'Helpfulness': 1}},
        ]
        write_jsonl(str(eval_file), original_data)

        # Replace entries 0 and 2
        new_results = {
            0: {'eval_id': 0, 'eval_scores': {'Helpfulness': 3}},
            2: {'eval_id': 2, 'eval_scores': {'Helpfulness': 1}},
        }

        merge_results_in_place(eval_file, new_results)

        with open(eval_file, 'r') as f:
            updated_data = [json.loads(line) for line in f]

        assert updated_data[0]['eval_scores'] == {'Helpfulness': 3}
        assert updated_data[1] == original_data[1]  # Unchanged
        assert updated_data[2]['eval_scores'] == {'Helpfulness': 1}
        assert updated_data[3] == original_data[3]  # Unchanged

    def test_empty_new_results(self, temp_test_dir):
        """Test with no entries to replace."""
        from rerun_failed_evals import merge_results_in_place

        eval_file = temp_test_dir / "test_eval.jsonl"
        original_data = [
            {'eval_id': 0, 'eval_scores': {'Helpfulness': 2}},
            {'eval_id': 1, 'eval_scores': {'Helpfulness': 3}},
        ]
        write_jsonl(str(eval_file), original_data)

        # Get original file content
        with open(eval_file, 'r') as f:
            original_content = f.read()

        merge_results_in_place(eval_file, {})

        with open(eval_file, 'r') as f:
            after_content = f.read()

        assert original_content == after_content

    def test_atomic_write_safety(self, temp_test_dir):
        """Test that merge uses atomic write (temp file + rename)."""
        from rerun_failed_evals import merge_results_in_place

        eval_file = temp_test_dir / "test_eval.jsonl"
        original_data = [{'eval_id': 0, 'eval_scores': {}}]
        write_jsonl(str(eval_file), original_data)

        new_results = {0: {'eval_id': 0, 'eval_scores': {'Helpfulness': 2}}}

        # After merge, there should be no .tmp file left
        merge_results_in_place(eval_file, new_results)

        temp_file = eval_file.with_suffix('.tmp')
        assert not temp_file.exists()
        assert eval_file.exists()

    def test_invalid_line_index_raises(self, temp_test_dir):
        """Test that invalid line index raises error."""
        from rerun_failed_evals import merge_results_in_place

        eval_file = temp_test_dir / "test_eval.jsonl"
        original_data = [
            {'eval_id': 0, 'eval_scores': {'Helpfulness': 2}},
            {'eval_id': 1, 'eval_scores': {'Helpfulness': 3}},
        ]
        write_jsonl(str(eval_file), original_data)

        # Try to replace line 5 which doesn't exist
        new_results = {5: {'eval_id': 5, 'eval_scores': {'Helpfulness': 1}}}

        with pytest.raises(IndexError):
            merge_results_in_place(eval_file, new_results)


# =============================================================================
# Test retry loop behavior
# =============================================================================

class TestRerunUntilSuccess:
    """Tests for rerun_until_success function - uses mocked subprocess calls."""

    def test_success_on_first_retry(self, temp_test_dir):
        """Test that loop exits when all entries are fixed on first retry."""
        from rerun_failed_evals import rerun_until_success

        # Create trajectory and eval files
        case_indices = [10, 11, 12]
        traj_file = temp_test_dir / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # Entry 1 is unfixable initially
        eval_data = create_eval_data_with_failures(3, 'agent_help', empty_thought_indices=[1])
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))

        # Mock run_targeted_eval to return a valid result
        def mock_run_eval(traj_file, case_idx, eval_type, temperature, evaluator_model, quantization, seed=42):
            return {'eval_id': 1, 'eval_scores': {'Helpfulness': 2}, 'Evaluator Thought': 'Fixed'}

        with patch('rerun_failed_evals.run_targeted_eval', side_effect=mock_run_eval):
            result = rerun_until_success(
                eval_file=eval_file,
                traj_file=traj_file,
                eval_type='agent_help',
                evaluator_model='test-model',
                quantization='int4',
                temperature=0.3,
                max_retries=20
            )

        assert result['success'] == True
        assert result['retries'] == 1
        assert result['fixed_count'] == 1

    def test_gives_up_after_max_retries(self, temp_test_dir):
        """Test that loop gives up after max retries."""
        from rerun_failed_evals import rerun_until_success

        case_indices = [10, 11]
        traj_file = temp_test_dir / "model_r10-12_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        eval_data = create_eval_data_with_failures(2, 'agent_help', empty_thought_indices=[0])
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))

        # Mock run_targeted_eval to always return unfixable result
        def mock_run_eval_still_broken(*args, **kwargs):
            return {'eval_id': 0, 'eval_scores': {}, 'Evaluator Thought': ''}

        with patch('rerun_failed_evals.run_targeted_eval', side_effect=mock_run_eval_still_broken):
            result = rerun_until_success(
                eval_file=eval_file,
                traj_file=traj_file,
                eval_type='agent_help',
                evaluator_model='test-model',
                quantization='int4',
                temperature=0.3,
                max_retries=3
            )

        assert result['success'] == False
        assert result['retries'] == 3
        assert len(result['permanently_unfixable']) == 1

    def test_no_unfixable_entries_skips(self, temp_test_dir):
        """Test that files with no unfixable entries are skipped."""
        from rerun_failed_evals import rerun_until_success

        case_indices = [10, 11, 12]
        traj_file = temp_test_dir / "model_r10-13_test.jsonl"
        write_jsonl(str(traj_file), create_trajectory_data(case_indices))

        # All entries are valid
        eval_data = create_eval_data_with_failures(3, 'agent_help')
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), eval_type))
            write_jsonl(str(eval_file), eval_data)

        eval_file = Path(ToolEmuFilePaths.trajectory_to_eval(str(traj_file), 'agent_help'))

        result = rerun_until_success(
            eval_file=eval_file,
            traj_file=traj_file,
            eval_type='agent_help',
            evaluator_model='test-model',
            quantization='int4',
            temperature=0.3,
            max_retries=20
        )

        assert result['success'] == True
        assert result['retries'] == 0
        assert result['skipped'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
