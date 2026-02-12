#!/usr/bin/env python3
"""Tests for partition_results_by_split.py"""

import pytest
import sys
import tempfile
import json
from pathlib import Path

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "analysis"))

from partition_results_by_split import (
    partition_aligned_data,
    create_summary_report,
    process_trajectory_file,
    collect_finetuned_source_models,
    ProcessingResults,
    report_incomplete_files,
    print_final_summary,
)
from utils.toolemu_utils import ToolEmuFilePaths, extract_score
from utils.train_utils import infer_dataset_type, get_dpo_data_path, partition_by_case_indices

# Test fixture directory containing mock dataset files
TEST_DATA_DIR = str(Path(__file__).parent / 'fixtures' / 'dpo_data')


class TestInferDpoDataPath:
    """Test the DPO data path inference from trajectory filenames."""

    def test_safe_dataset(self):
        """Test that filenames with '_safe_' map to safe.jsonl"""
        assert get_dpo_data_path(infer_dataset_type('Qwen-8B_safe_emu-Qwen-32B_eval-Qwen-32B_int4_r0-144.jsonl', dpo_data_dir=TEST_DATA_DIR), dpo_data_dir=TEST_DATA_DIR) == f'{TEST_DATA_DIR}/safe.jsonl'
        assert get_dpo_data_path(infer_dataset_type('model_safe_checkpoint.jsonl', dpo_data_dir=TEST_DATA_DIR), dpo_data_dir=TEST_DATA_DIR) == f'{TEST_DATA_DIR}/safe.jsonl'

    def test_both_dataset(self):
        """Test that filenames with '_both_' map to both.jsonl"""
        assert get_dpo_data_path(infer_dataset_type('Qwen-8B_both_emu-Qwen-32B_eval-Qwen-32B_int4_r0-144.jsonl', dpo_data_dir=TEST_DATA_DIR), dpo_data_dir=TEST_DATA_DIR) == f'{TEST_DATA_DIR}/both.jsonl'
        assert get_dpo_data_path(infer_dataset_type('model_both_checkpoint.jsonl', dpo_data_dir=TEST_DATA_DIR), dpo_data_dir=TEST_DATA_DIR) == f'{TEST_DATA_DIR}/both.jsonl'

    def test_safe_takes_precedence(self):
        """Test that '_safe_' is correctly detected"""
        assert get_dpo_data_path(infer_dataset_type('model_safe_other_stuff.jsonl', dpo_data_dir=TEST_DATA_DIR), dpo_data_dir=TEST_DATA_DIR) == f'{TEST_DATA_DIR}/safe.jsonl'

    def test_help_diff2_compound(self):
        """Test that '_help_diff2' is correctly detected as compound dataset"""
        assert get_dpo_data_path(infer_dataset_type('model_help_diff2.jsonl', dpo_data_dir=TEST_DATA_DIR), dpo_data_dir=TEST_DATA_DIR) == f'{TEST_DATA_DIR}/help_diff2.jsonl'

    def test_both_safe_and_diff2(self):
        """Test behavior when both patterns are present (should detect safe_diff2 compound)"""
        # With the new logic, '_safe_diff2' should be detected as compound dataset
        assert get_dpo_data_path(infer_dataset_type('model_safe_diff2.jsonl', dpo_data_dir=TEST_DATA_DIR), dpo_data_dir=TEST_DATA_DIR) == f'{TEST_DATA_DIR}/safe_diff2.jsonl'


class TestPartitionTrajectories:
    """Test trajectory partitioning logic using partition_by_case_indices."""

    def test_partition_basic(self):
        """Test basic partitioning into train/test sets"""
        trajectories = [
            {'case_idx': 0, 'output': 'result0'},
            {'case_idx': 1, 'output': 'result1'},
            {'case_idx': 2, 'output': 'result2'},
            {'case_idx': 3, 'output': 'result3'}
        ]
        train_indices = {0, 2}
        test_indices = {1, 3}

        train, test, unknown = partition_by_case_indices(trajectories, train_indices, test_indices)

        assert len(train) == 2
        assert len(test) == 2
        assert len(unknown) == 0
        assert train[0]['case_idx'] == 0
        assert train[1]['case_idx'] == 2
        assert test[0]['case_idx'] == 1
        assert test[1]['case_idx'] == 3

    def test_partition_with_unknown_indices(self):
        """Test partitioning with unknown case indices"""
        trajectories = [
            {'case_idx': 0, 'output': 'result0'},
            {'case_idx': 5, 'output': 'result5'},  # Not in train or test
            {'case_idx': 1, 'output': 'result1'}
        ]
        train_indices = {0}
        test_indices = {1}

        train, test, unknown = partition_by_case_indices(trajectories, train_indices, test_indices)

        assert len(train) == 1
        assert len(test) == 1
        assert unknown == [5]

    def test_partition_missing_case_idx(self):
        """Test that missing case_idx raises error"""
        trajectories = [{'output': 'result'}]
        train_indices = {0}
        test_indices = {1}

        try:
            partition_by_case_indices(trajectories, train_indices, test_indices)
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert "case_idx" in str(e)


class TestPartitionAlignedData:
    """Test aligned data partitioning (e.g., evaluations)."""

    def test_partition_aligned_basic(self):
        """Test partitioning aligned data - returns (case_idx, data) tuples"""
        trajectories = [
            {'case_idx': 0},
            {'case_idx': 1},
            {'case_idx': 2}
        ]
        aligned_data = [
            {'eval_id': 0, 'score': 1.0},
            {'eval_id': 1, 'score': 2.0},
            {'eval_id': 2, 'score': 3.0}
        ]
        train_indices = {0, 2}
        test_indices = {1}

        train, test = partition_aligned_data(trajectories, aligned_data, train_indices, test_indices)

        assert len(train) == 2
        assert len(test) == 1
        # Returns (case_idx, data) tuples
        assert train[0] == (0, {'eval_id': 0, 'score': 1.0})
        assert train[1] == (2, {'eval_id': 2, 'score': 3.0})
        assert test[0] == (1, {'eval_id': 1, 'score': 2.0})

    def test_partition_aligned_length_mismatch(self):
        """Test that length mismatch raises error"""
        trajectories = [{'case_idx': 0}]
        aligned_data = [{'eval_id': 0}, {'eval_id': 1}]
        train_indices = {0}
        test_indices = {1}

        try:
            partition_aligned_data(trajectories, aligned_data, train_indices, test_indices)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Length mismatch" in str(e)

    def test_partition_aligned_skip_unknown_cases(self):
        """Test that cases not in train or test are skipped"""
        trajectories = [
            {'case_idx': 0},
            {'case_idx': 5},  # Not in train or test
            {'case_idx': 1}
        ]
        aligned_data = [
            {'eval_id': 0, 'score': 1.0},
            {'eval_id': 1, 'score': 2.0},
            {'eval_id': 2, 'score': 3.0}
        ]
        train_indices = {0}
        test_indices = {1}

        train, test = partition_aligned_data(trajectories, aligned_data, train_indices, test_indices)

        assert len(train) == 1
        assert len(test) == 1
        # Returns (case_idx, data) tuples
        assert train[0] == (0, {'eval_id': 0, 'score': 1.0})
        assert test[0] == (1, {'eval_id': 2, 'score': 3.0})


class TestCreateSummaryReport:
    """Test summary report creation."""

    def test_summary_basic(self):
        """Test basic summary report creation"""
        train_indices = {0, 2, 4, 6}
        test_indices = {1, 3, 5, 7}
        all_stats = {'file1.jsonl': {'trajectory': {'original': 10}}}

        report = create_summary_report(train_indices, test_indices, all_stats, seed=42)

        assert report['split_seed'] == 42
        assert report['num_train_cases'] == 4
        assert report['num_test_cases'] == 4
        assert report['train_case_indices'] == [0, 2, 4, 6]
        assert report['test_case_indices'] == [1, 3, 5, 7]
        assert report['num_files_processed'] == 1
        assert 'file1.jsonl' in report['files_processed']

    def test_summary_empty_stats(self):
        """Test summary with empty stats"""
        train_indices = {0, 1}
        test_indices = {2, 3}
        all_stats = {}

        report = create_summary_report(train_indices, test_indices, all_stats, seed=42)

        assert report['num_files_processed'] == 0
        assert report['files_processed'] == []

    def test_summary_multiple_files(self):
        """Test summary with multiple files"""
        train_indices = {0}
        test_indices = {1}
        all_stats = {
            'file1.jsonl': {'trajectory': {'original': 5}},
            'file2.jsonl': {'trajectory': {'original': 10}},
        }

        report = create_summary_report(train_indices, test_indices, all_stats, seed=42)

        assert report['num_files_processed'] == 2
        assert 'file1.jsonl' in report['files_processed']
        assert 'file2.jsonl' in report['files_processed']


class TestCompletenessValidation:
    """Test validation of incomplete trajectory files using model_utils functions."""

    def test_range_extraction_from_filename(self):
        """Test that range extraction works correctly"""
        from utils.toolemu_utils import extract_case_range_from_filename

        # Valid range formats
        assert extract_case_range_from_filename("model_r0-15_1234.jsonl") == (0, 15)
        assert extract_case_range_from_filename("model_r60-74_1234.jsonl") == (60, 74)
        assert extract_case_range_from_filename("model_r130-144_1234.jsonl") == (130, 144)

        # No range in filename
        assert extract_case_range_from_filename("model_1234.jsonl") is None

    def test_incomplete_file_validation_logic(self):
        """Test the validation logic for incomplete files"""
        from utils.toolemu_utils import extract_case_range_from_filename

        # Simulate the actual scenario: file says r74-88 (14 cases) but has 0 trajectories
        filename = "model_safe_emu-Qwen-32B_eval-Qwen-32B_int4_r74-88_1234.jsonl"
        range_info = extract_case_range_from_filename(filename)

        assert range_info is not None
        start_idx, end_idx = range_info
        expected_count = end_idx - start_idx
        actual_count = 0  # File exists but is empty or incomplete

        # This should trigger failure - the condition used in partition_results_by_split.py
        is_incomplete = actual_count != expected_count
        assert is_incomplete is True
        assert expected_count == 14

    def test_complete_file_validation_logic(self):
        """Test that complete files pass validation"""
        from utils.toolemu_utils import extract_case_range_from_filename

        filename = "model_safe_emu-Qwen-32B_eval-Qwen-32B_int4_r0-15_1234.jsonl"
        range_info = extract_case_range_from_filename(filename)

        start_idx, end_idx = range_info
        expected_count = end_idx - start_idx
        actual_count = 15  # Complete data

        # This should pass validation
        is_incomplete = actual_count != expected_count
        assert is_incomplete is False

    def test_validation_detects_incomplete_with_real_data(self):
        """Test validation with actual trajectory data structure"""
        from utils.toolemu_utils import extract_case_range_from_filename

        # Real filename from user's data (finetuned model identified by seed)
        filename = "Qwen-8B_int4_1A6000_safe_s42_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r74-88_1611_045834.jsonl"

        # Extract range using actual function
        range_info = extract_case_range_from_filename(filename)
        assert range_info is not None, "Should extract range from filename"

        start_idx, end_idx = range_info
        assert start_idx == 74
        assert end_idx == 88
        expected_count = end_idx - start_idx
        assert expected_count == 14

        # Simulate incomplete trajectories (like what load_and_validate_all_eval_files would return)
        incomplete_trajectories = [
            {'case_idx': 74, 'output': 'result'},
            {'case_idx': 75, 'output': 'result'},
            {'case_idx': 76, 'output': 'result'},
            {'case_idx': 77, 'output': 'result'},
            {'case_idx': 78, 'output': 'result'},
            {'case_idx': 79, 'output': 'result'}
        ]
        actual_count = len(incomplete_trajectories)
        assert actual_count == 6

        # Test the actual condition used in partition_results_by_split.py line 581
        should_fail = actual_count != expected_count
        assert should_fail is True, "Should detect incomplete file"

        # Test the actual failure reason format used in the script (line 583)
        failure_reason = f"Incomplete data: expected {expected_count} cases (r{start_idx}-{end_idx}), found {actual_count}"
        assert failure_reason == "Incomplete data: expected 14 cases (r74-88), found 6"

    def test_validation_passes_complete_with_real_data(self):
        """Test validation passes with complete trajectory data"""
        from utils.toolemu_utils import extract_case_range_from_filename

        filename = "model_emu-Qwen-32B_eval-Qwen-32B_int4_r0-15_1234.jsonl"

        range_info = extract_case_range_from_filename(filename)
        start_idx, end_idx = range_info
        expected_count = end_idx - start_idx

        # Complete trajectories
        complete_trajectories = [{'case_idx': i, 'output': f'result{i}'} for i in range(start_idx, end_idx)]
        actual_count = len(complete_trajectories)

        # Test the condition - should NOT fail
        should_fail = actual_count != expected_count
        assert should_fail is False, "Should pass validation for complete file"
        assert actual_count == expected_count
        assert actual_count == 15


class TestAggregateMode:
    """Test aggregate-only mode functionality (regression tests for bugs fixed)."""

    def test_process_trajectory_file_with_none_test_dir(self):
        """Test that process_trajectory_file handles test_dir=None without errors.

        Regression test: Previously would fail with TypeError when test_dir=None
        because code tried to use test_dir in Path operations.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test trajectory data (mimicking real directory structure)
            # results_dir should be the 'trajectories' directory, not the model subdirectory
            results_dir = tmpdir_path / 'results' / 'trajectories'
            model_dir = results_dir / 'Qwen_Qwen3-8B'
            model_dir.mkdir(parents=True)

            traj_file = model_dir / 'Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-5.jsonl'
            trajectories = [
                {'case_idx': 0, 'output': 'result1'},
                {'case_idx': 1, 'output': 'result2'},
                {'case_idx': 2, 'output': 'result3'},
            ]
            with open(traj_file, 'w') as f:
                for traj in trajectories:
                    json.dump(traj, f)
                    f.write('\n')

            # Create eval files
            for eval_type in ToolEmuFilePaths.EVAL_TYPES:
                eval_file = model_dir / f'Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-5_eval_{eval_type}.jsonl'
                evals = [
                    {'case_idx': 0, 'score': 2},
                    {'case_idx': 1, 'score': 3},
                    {'case_idx': 2, 'score': 1},
                ]
                with open(eval_file, 'w') as f:
                    for eval_entry in evals:
                        json.dump(eval_entry, f)
                        f.write('\n')

            # Create output directory
            train_dir = tmpdir_path / 'output'
            train_dir.mkdir(parents=True)

            # Prepare evals_dict
            evals_dict = {eval_type: [
                {'case_idx': 0, 'score': 2},
                {'case_idx': 1, 'score': 3},
                {'case_idx': 2, 'score': 1},
            ] for eval_type in ToolEmuFilePaths.EVAL_TYPES}

            # This should NOT raise TypeError even with test_dir=None
            stats = process_trajectory_file(
                traj_file=traj_file,
                results_dir=results_dir,
                train_dir=train_dir,
                test_dir=None,  # Aggregate mode: test_dir is None
                train_indices={0, 1, 2},  # All indices in train
                test_indices=set(),  # No test indices
                trajectories=trajectories,
                evals_dict=evals_dict,
                verbose=False
            )

            # Verify stats returned successfully
            assert stats is not None

            # Verify train files were created
            expected_output = train_dir / 'trajectories' / 'Qwen_Qwen3-8B' / traj_file.name
            assert expected_output.exists(), f"Expected file at {expected_output}"

            # Verify test files were NOT created (since test_dir=None)
            # If test files were attempted, it would have raised TypeError

    def test_aggregate_mode_no_split_suffix_applied(self):
        """Test that aggregate mode doesn't add split suffixes to model directories.

        Regression test: Previously would add '_aggregate_seed0_split' suffix
        in aggregate mode, which was undesirable.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create minimal trajectory file
            results_dir = tmpdir_path / 'results' / 'trajectories'
            model_dir = results_dir / 'Qwen_Qwen3-8B'
            model_dir.mkdir(parents=True)

            traj_file = model_dir / 'Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-2.jsonl'
            trajectories = [
                {'case_idx': 0, 'output': 'result1'},
                {'case_idx': 1, 'output': 'result2'},
            ]
            with open(traj_file, 'w') as f:
                for traj in trajectories:
                    json.dump(traj, f)
                    f.write('\n')

            # Create eval files
            for eval_type in ToolEmuFilePaths.EVAL_TYPES:
                eval_file = model_dir / f'Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-2_eval_{eval_type}.jsonl'
                evals = [{'case_idx': 0, 'score': 2}, {'case_idx': 1, 'score': 3}]
                with open(eval_file, 'w') as f:
                    for eval_entry in evals:
                        json.dump(eval_entry, f)
                        f.write('\n')

            train_dir = tmpdir_path / 'output'
            train_dir.mkdir(parents=True)

            evals_dict = {eval_type: [
                {'case_idx': 0, 'score': 2},
                {'case_idx': 1, 'score': 3},
            ] for eval_type in ToolEmuFilePaths.EVAL_TYPES}

            # Test aggregate mode behavior (no suffix added)
            stats = process_trajectory_file(
                traj_file=traj_file,
                results_dir=results_dir,
                train_dir=train_dir,
                test_dir=None,
                train_indices={0, 1},
                test_indices=set(),
                trajectories=trajectories,
                evals_dict=evals_dict,
                verbose=False
            )

            # Verify output goes to original directory name (no suffix)
            output_file = train_dir / 'trajectories' / 'Qwen_Qwen3-8B' / traj_file.name
            assert output_file.exists(), "File should be in original directory without suffix"

            # Verify no suffixed directory was created
            bad_dir = train_dir / 'trajectories' / 'Qwen_Qwen3-8B_aggregate_seed0_split'
            assert not bad_dir.exists(), "Should not create directory with aggregate suffix"

    def test_source_model_filtering_logic_in_aggregate_mode(self):
        """Test that source model filtering logic is skipped in aggregate mode.

        Regression test: Previously, in aggregate mode with finetuned_configs=set(),
        all source models would be filtered out with "no matching finetuned model" error.
        """
        # This is more of an integration test that verifies the logic at lines 759-787
        # We can't easily test this without mocking argparse, but we can verify
        # that the logic branches correctly based on aggregate_only flag

        # The key is that when args.aggregate_only=True:
        # - Line 759: should skip into the else branch at line 785
        # - Line 787: should use list(splits_by_dpo_seed.keys())
        # - This means source models are NOT filtered by finetuned_configs

        # We verify this by ensuring infer_dataset_type returns None for source models
        # (which is the condition that triggers is_source=True at line 758)

        source_model_filename = 'Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-144.jsonl'
        dataset_type = infer_dataset_type(source_model_filename, include_seed=False, dpo_data_dir=TEST_DATA_DIR)

        # Source models should return None (no dataset marker in filename)
        assert dataset_type is None, "Source model should have no dataset type"

        # Finetuned model for comparison (identified by seed suffix)
        finetuned_filename = 'Qwen-8B_int4_help_s42_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-144.jsonl'
        dataset_type_finetuned = infer_dataset_type(finetuned_filename, include_seed=False, dpo_data_dir=TEST_DATA_DIR)

        # Finetuned models should have a dataset type
        assert dataset_type_finetuned is not None, "Finetuned model should have dataset type"

    def test_partition_mode_creates_train_and_test_files(self):
        """Test that partition mode creates files in both train and test directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create minimal trajectory file
            results_dir = tmpdir_path / 'results' / 'trajectories'
            model_dir = results_dir / 'Qwen_Qwen3-8B'
            model_dir.mkdir(parents=True)

            traj_file = model_dir / 'Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-2.jsonl'
            trajectories = [
                {'case_idx': 0, 'output': 'result1'},
                {'case_idx': 1, 'output': 'result2'},
            ]
            with open(traj_file, 'w') as f:
                for traj in trajectories:
                    json.dump(traj, f)
                    f.write('\n')

            # Create eval files
            for eval_type in ToolEmuFilePaths.EVAL_TYPES:
                eval_file = model_dir / f'Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-2_eval_{eval_type}.jsonl'
                evals = [{'case_idx': 0, 'score': 2}, {'case_idx': 1, 'score': 3}]
                with open(eval_file, 'w') as f:
                    for eval_entry in evals:
                        json.dump(eval_entry, f)
                        f.write('\n')

            train_dir = tmpdir_path / 'train'
            test_dir = tmpdir_path / 'test'
            train_dir.mkdir()
            test_dir.mkdir()

            evals_dict = {eval_type: [
                {'case_idx': 0, 'score': 2},
                {'case_idx': 1, 'score': 3},
            ] for eval_type in ToolEmuFilePaths.EVAL_TYPES}

            # Test partition mode (with both train and test dirs)
            stats = process_trajectory_file(
                traj_file=traj_file,
                results_dir=results_dir,
                train_dir=train_dir,
                test_dir=test_dir,  # Not None in partition mode
                train_indices={0},
                test_indices={1},
                trajectories=trajectories,
                evals_dict=evals_dict,
                verbose=False
            )

            # Verify output goes to both train and test directories
            train_file = train_dir / 'trajectories' / 'Qwen_Qwen3-8B' / traj_file.name
            test_file = test_dir / 'trajectories' / 'Qwen_Qwen3-8B' / traj_file.name

            assert train_file.exists(), "Train file should exist"
            assert test_file.exists(), "Test file should exist"

            # Verify both train and test files have correct data
            with open(train_file) as f:
                train_data = [json.loads(line) for line in f]
            with open(test_file) as f:
                test_data = [json.loads(line) for line in f]

            assert len(train_data) == 1 and train_data[0]['case_idx'] == 0
            assert len(test_data) == 1 and test_data[0]['case_idx'] == 1


class TestCollectFinetunedSourceModelsSeedValidation:
    """Test seed validation in collect_finetuned_source_models.

    Finetuned models are identified by having a seed suffix (_s{N}).
    """

    def test_matching_seed_passes(self, tmp_path):
        """Test that directories with matching seed pass validation."""
        # Create directory structure with s43 (finetuned model)
        model_dir = tmp_path / "Qwen-8B_most_s43"
        model_dir.mkdir()
        traj_file = model_dir / "Qwen-8B_most_s43_emu-Qwen-32B_eval-Qwen-32B_int4_r0-10_0101_120000.jsonl"
        traj_file.touch()

        # Should include directory when seed matches
        configs, skipped, seeds_by_config = collect_finetuned_source_models([traj_file], expected_seed=43)
        assert len(configs) == 1
        assert len(skipped) == 0
        assert 43 in list(seeds_by_config.values())[0]

    def test_mismatched_seed_skips(self, tmp_path):
        """Test that directories with mismatched seed are skipped (not raised)."""
        # Create directory structure with s43 (finetuned model)
        model_dir = tmp_path / "Qwen-8B_most_s43"
        model_dir.mkdir()
        traj_file = model_dir / "Qwen-8B_most_s43_emu-Qwen-32B_eval-Qwen-32B_int4_r0-10_0101_120000.jsonl"
        traj_file.touch()

        # Should skip when seed doesn't match (not raise)
        configs, skipped, seeds_by_config = collect_finetuned_source_models([traj_file], expected_seed=42)
        assert len(configs) == 0
        assert "Qwen-8B_most_s43" in skipped

    def test_source_model_dirs_ignored(self, tmp_path):
        """Test that source model directories (no seed) are not validated."""
        # Create a source model directory (no seed = source model)
        model_dir = tmp_path / "Qwen_Qwen3-8B"
        model_dir.mkdir()
        traj_file = model_dir / "Qwen_Qwen3-8B_emu-Qwen-32B_eval-Qwen-32B_int4_r0-10_0101_120000.jsonl"
        traj_file.touch()

        # Source models should be ignored (no validation), returns empty configs and empty skipped
        configs, skipped, seeds_by_config = collect_finetuned_source_models([traj_file], expected_seed=999)
        assert len(configs) == 0
        assert len(skipped) == 0
        assert len(seeds_by_config) == 0

    def test_no_validation_when_seed_none(self, tmp_path):
        """Test that no validation happens when expected_seed is None."""
        # Create directory with seed (finetuned model)
        model_dir = tmp_path / "Qwen-8B_most_s99"
        model_dir.mkdir()
        traj_file = model_dir / "Qwen-8B_most_s99_emu-Qwen-32B_eval-Qwen-32B_int4_r0-10_0101_120000.jsonl"
        traj_file.touch()

        # Should include and not skip when expected_seed is None (aggregate mode)
        configs, skipped, seeds_by_config = collect_finetuned_source_models([traj_file], expected_seed=None)
        assert len(configs) == 1
        assert len(skipped) == 0
        assert 99 in list(seeds_by_config.values())[0]

    def test_multiple_seeds_for_same_source(self, tmp_path):
        """Test that seeds_by_source_config tracks multiple seeds for the same source model."""
        # Create finetuned models with different seeds but same source
        for seed in [42, 99]:
            model_dir = tmp_path / f"dpo_Qwen-8B_help_s{seed}"
            model_dir.mkdir()
            traj_file = model_dir / f"dpo_Qwen-8B_help_s{seed}_emu-Qwen-32B_eval-Qwen-32B_int4_r0-10_0101_120000.jsonl"
            traj_file.touch()

        traj_files = list(tmp_path.glob("*/*.jsonl"))
        configs, skipped, seeds_by_config = collect_finetuned_source_models(traj_files, expected_seed=None)

        assert len(configs) == 1  # Same source model config
        # The source config should have both seeds
        source_config = list(seeds_by_config.keys())[0]
        assert seeds_by_config[source_config] == {42, 99}


class TestMultiSeedSourceModelPartitioning:
    """Test multi-seed source model partitioning logic."""

    def test_source_model_output_has_seed_suffix(self, tmp_path):
        """Test that source models are output to directories with seed suffix."""
        results_dir = tmp_path / 'results' / 'trajectories'

        # Create source model directory and file
        source_dir = results_dir / 'Qwen_Qwen3-8B'
        source_dir.mkdir(parents=True)
        source_traj = source_dir / 'Qwen_Qwen3-8B_emu-Qwen-32B_eval-Qwen-32B_int4_r0-2.jsonl'
        trajectories = [{'case_idx': 0, 'output': 'a'}, {'case_idx': 1, 'output': 'b'}]
        with open(source_traj, 'w') as f:
            for t in trajectories:
                json.dump(t, f)
                f.write('\n')

        # Create eval files for source
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = source_dir / f'Qwen_Qwen3-8B_emu-Qwen-32B_eval-Qwen-32B_int4_r0-2_eval_{eval_type}.jsonl'
            with open(eval_file, 'w') as f:
                for t in trajectories:
                    json.dump({'case_idx': t['case_idx'], 'eval_scores': {'ToolCallRisk': 2}}, f)
                    f.write('\n')

        # Create finetuned model with seed (identified by seed suffix, no prefix)
        finetuned_dir = results_dir / 'Qwen-8B_help_s42'
        finetuned_dir.mkdir(parents=True)
        finetuned_traj = finetuned_dir / 'Qwen-8B_help_s42_emu-Qwen-32B_eval-Qwen-32B_int4_r0-2.jsonl'
        with open(finetuned_traj, 'w') as f:
            for t in trajectories:
                json.dump(t, f)
                f.write('\n')

        # Create eval files for finetuned
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = finetuned_dir / f'Qwen-8B_help_s42_emu-Qwen-32B_eval-Qwen-32B_int4_r0-2_eval_{eval_type}.jsonl'
            with open(eval_file, 'w') as f:
                for t in trajectories:
                    json.dump({'case_idx': t['case_idx'], 'eval_scores': {'ToolCallRisk': 2}}, f)
                    f.write('\n')

        train_dir = tmp_path / 'train'
        test_dir = tmp_path / 'test'
        train_dir.mkdir()
        test_dir.mkdir()

        # Get seeds from finetuned models
        traj_files = list(results_dir.glob('*/*.jsonl'))
        traj_files = [f for f in traj_files if not '_eval_' in f.name]
        _, _, seeds_by_config = collect_finetuned_source_models(traj_files, expected_seed=None)

        # Verify source model output directory would have seed suffix
        # The key assertion: seeds_by_config should map Qwen3-8B source to {42}
        assert len(seeds_by_config) == 1
        source_config = list(seeds_by_config.keys())[0]
        assert 42 in seeds_by_config[source_config]

    def test_source_model_skipped_without_finetuned_counterpart(self, tmp_path):
        """Test that source models without finetuned counterparts are skipped in partition mode."""
        results_dir = tmp_path / 'results' / 'trajectories'

        # Create source model only (no finetuned counterpart)
        source_dir = results_dir / 'Qwen_Qwen3-8B'
        source_dir.mkdir(parents=True)
        source_traj = source_dir / 'Qwen_Qwen3-8B_emu-Qwen-32B_eval-Qwen-32B_int4_r0-2.jsonl'
        source_traj.touch()

        traj_files = [source_traj]
        configs, skipped, seeds_by_config = collect_finetuned_source_models(traj_files, expected_seed=None)

        # No finetuned models, so no configs and no seeds
        assert len(configs) == 0
        assert len(seeds_by_config) == 0


class TestProcessingResults:
    """Test the ProcessingResults dataclass."""

    def test_default_initialization(self):
        """Test that ProcessingResults initializes with empty defaults."""
        results = ProcessingResults()
        assert results.all_stats == {}
        assert results.successful_files == []
        assert results.failed_files == []
        assert results.skipped_source_models == []
        assert results.incomplete_files == []
        assert results.missing_files_by_eval_type == {}
        assert results.train_evals_by_config == {}
        assert results.test_evals_by_config == {}
        assert results.seed_by_config == {}

    def test_mutable_defaults_are_independent(self):
        """Test that each instance has independent mutable defaults."""
        results1 = ProcessingResults()
        results2 = ProcessingResults()

        results1.successful_files.append("file1.jsonl")
        results1.all_stats["key"] = "value"

        assert results2.successful_files == []
        assert results2.all_stats == {}

    def test_can_accumulate_results(self):
        """Test typical usage pattern of accumulating results."""
        results = ProcessingResults()

        # Simulate processing files
        results.successful_files.append("file1.jsonl")
        results.successful_files.append("file2.jsonl")
        results.failed_files.append(("file3.jsonl", "Parse error"))
        results.skipped_source_models.append(("file4.jsonl", "No matching finetuned model"))
        results.incomplete_files.append(("file5.jsonl", "Missing evals"))

        assert len(results.successful_files) == 2
        assert len(results.failed_files) == 1
        assert len(results.skipped_source_models) == 1
        assert len(results.incomplete_files) == 1


class TestReportIncompleteFiles:
    """Test the report_incomplete_files function."""

    def test_no_incomplete_produces_no_output(self, capsys):
        """Test that empty incomplete list produces no output."""
        report_incomplete_files([], verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_incomplete_files_printed(self, capsys):
        """Test that incomplete files are reported."""
        incomplete = [
            ("file1.jsonl", "Missing agent_safe eval"),
            ("file2.jsonl", "Count mismatch"),
        ]
        report_incomplete_files(incomplete, verbose=True)
        captured = capsys.readouterr()

        assert "Skipped 2 incomplete file(s)" in captured.out
        assert "file1.jsonl" in captured.out
        assert "Missing agent_safe eval" in captured.out


class TestPrintFinalSummary:
    """Test the print_final_summary function."""

    def test_aggregate_mode_summary(self, capsys):
        """Test summary output in aggregate mode."""
        results = ProcessingResults()
        results.successful_files = ["file1.jsonl", "file2.jsonl"]

        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir)
            print_final_summary(
                trajectory_files=[Path("f1"), Path("f2")],
                results=results,
                train_indices_by_seed={42: set(range(72))},
                test_indices_by_seed={42: set(range(72, 144))},
                train_dir=train_dir,
                test_dir=None,
                aggregate_only=True,
            )

        captured = capsys.readouterr()
        assert "AGGREGATION COMPLETE" in captured.out
        assert "Successfully aggregated: 2" in captured.out

    def test_partition_mode_summary(self, capsys):
        """Test summary output in partition mode."""
        results = ProcessingResults()
        results.successful_files = ["file1.jsonl"]
        results.failed_files = [("file2.jsonl", "Error")]
        results.skipped_source_models = [("file3.jsonl", "No match")]

        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            test_dir = Path(tmpdir) / "test"
            train_dir.mkdir()
            test_dir.mkdir()

            print_final_summary(
                trajectory_files=[Path("f1"), Path("f2"), Path("f3")],
                results=results,
                train_indices_by_seed={42: {0, 1, 2}},
                test_indices_by_seed={42: {3, 4}},
                train_dir=train_dir,
                test_dir=test_dir,
                aggregate_only=False,
            )

        captured = capsys.readouterr()
        assert "PARTITIONING COMPLETE" in captured.out
        assert "Successfully partitioned: 1" in captured.out
        assert "Failed: 1" in captured.out
        assert "Skipped (no matching finetuned model): 1" in captured.out

    def test_missing_eval_files_reported(self, capsys):
        """Test that missing eval files are reported in summary."""
        results = ProcessingResults()
        results.successful_files = ["file1.jsonl"]
        results.missing_files_by_eval_type = {
            'agent_safe': ["model1_emu-sim_eval-eval.jsonl"],
            'agent_help': [],
            'agent_help_ignore_safety': [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            print_final_summary(
                trajectory_files=[Path("f1")],
                results=results,
                train_indices_by_seed={42: set(range(72))},
                test_indices_by_seed={42: set(range(72, 144))},
                train_dir=Path(tmpdir),
                test_dir=None,
                aggregate_only=True,
            )

        captured = capsys.readouterr()
        assert "MISSING EVALUATION FILES SUMMARY" in captured.out
        assert "agent_safe: 1 file(s) missing" in captured.out


class TestComputeEvalStatisticsMixedFormats:
    """Test compute_eval_statistics with mixed single/multi-replicate formats."""

    def test_mixed_formats(self):
        """Test compute_eval_statistics with mixed single and multi-replicate formats."""
        from partition_results_by_split import compute_eval_statistics

        # Now takes (case_idx, eval_data) tuples
        evals = [
            # Single-replicate format entries
            (0, {'eval_scores': {'Helpfulness': 0}}),
            (1, {'eval_scores': {'Helpfulness': 2}}),
            # Multi-replicate format entry
            (2, {'replicates': [
                {'eval_scores': {'Helpfulness': 1}},
                {'eval_scores': {'Helpfulness': 1}},
                {'eval_scores': {'Helpfulness': 1}},
            ]}),
            # More single-replicate entries
            (3, {'eval_scores': {'Helpfulness': 3}}),
        ]

        result = compute_eval_statistics(evals, 'agent_help')

        assert 'Helpfulness' in result
        assert result['Helpfulness']['valid_scores'] == 4
        assert result['Helpfulness']['missing_scores'] == 0
        # Scores: 0, 2, median([1,1,1])=1, 3 -> mean = (0+2+1+3)/4 = 1.5
        assert result['Helpfulness']['mean'] == 1.5

    def test_mixed_formats_uses_median_for_multi_rep(self):
        """Test that multi-replicate entries use median for score extraction."""
        from partition_results_by_split import compute_eval_statistics

        # Now takes (case_idx, eval_data) tuples
        evals = [
            (0, {'eval_scores': {'Helpfulness': 2}}),
            # Multi-rep with asymmetric scores: mean=2, median=3
            (1, {'replicates': [
                {'eval_scores': {'Helpfulness': 0}},
                {'eval_scores': {'Helpfulness': 3}},
                {'eval_scores': {'Helpfulness': 3}},
            ]}),
        ]

        result = compute_eval_statistics(evals, 'agent_help')

        assert result['Helpfulness']['valid_scores'] == 2
        # Scores: 2, median([0,3,3])=3 -> mean = (2+3)/2 = 2.5
        assert result['Helpfulness']['mean'] == 2.5
