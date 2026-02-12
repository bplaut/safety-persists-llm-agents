#!/usr/bin/env python3
"""
Unit tests for src/utils/train_utils.py

Tests cover:
- partition_by_case_indices() function
- load_case_indices_from_finetune_data() function
- compute_test_indices() function
- infer_dataset_type() function
- get_dpo_data_path() function
- extract_seed_from_path() function
"""

import json
import os
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.train_utils import (
    partition_by_case_indices,
    compute_test_indices,
    load_case_indices_from_finetune_data,
    infer_dataset_type,
    get_dpo_data_path,
    extract_seed_from_path,
)
from utils.toolemu_utils import TOOLEMU_FULL_DATASET_SIZE

# Test fixture directory containing mock dataset files
TEST_DATA_DIR = str(Path(__file__).parent / 'fixtures' / 'dpo_data')


class TestPartitionByCaseIndices:
    """Tests for partition_by_case_indices function."""

    def test_basic_partitioning(self):
        """Test basic train/test partitioning."""
        items = [
            {'case_idx': 0, 'data': 'a'},
            {'case_idx': 1, 'data': 'b'},
            {'case_idx': 2, 'data': 'c'},
            {'case_idx': 3, 'data': 'd'},
        ]
        train_indices = {0, 2}
        test_indices = {1, 3}

        train, test, unknown = partition_by_case_indices(items, train_indices, test_indices)

        assert len(train) == 2
        assert len(test) == 2
        assert len(unknown) == 0
        assert train[0]['case_idx'] == 0
        assert train[1]['case_idx'] == 2
        assert test[0]['case_idx'] == 1
        assert test[1]['case_idx'] == 3

    def test_all_in_train(self):
        """Test when all items go to train set."""
        items = [{'case_idx': 0}, {'case_idx': 1}, {'case_idx': 2}]
        train_indices = {0, 1, 2}
        test_indices = set()

        train, test, unknown = partition_by_case_indices(items, train_indices, test_indices)

        assert len(train) == 3
        assert len(test) == 0
        assert len(unknown) == 0

    def test_all_in_test(self):
        """Test when all items go to test set."""
        items = [{'case_idx': 5}, {'case_idx': 6}]
        train_indices = set()
        test_indices = {5, 6}

        train, test, unknown = partition_by_case_indices(items, train_indices, test_indices)

        assert len(train) == 0
        assert len(test) == 2
        assert len(unknown) == 0

    def test_unknown_indices_tracked(self):
        """Test that unknown indices are tracked."""
        items = [
            {'case_idx': 0},
            {'case_idx': 1},
            {'case_idx': 99},  # Unknown
            {'case_idx': 2},
            {'case_idx': 100},  # Unknown
        ]
        train_indices = {0, 2}
        test_indices = {1}

        train, test, unknown = partition_by_case_indices(items, train_indices, test_indices)

        assert len(train) == 2
        assert len(test) == 1
        assert len(unknown) == 2
        assert 99 in unknown
        assert 100 in unknown

    def test_custom_case_idx_key(self):
        """Test with custom key name for case index."""
        items = [
            {'task_id': 0, 'data': 'a'},
            {'task_id': 1, 'data': 'b'},
            {'task_id': 2, 'data': 'c'},
        ]
        train_indices = {0, 1}
        test_indices = {2}

        train, test, unknown = partition_by_case_indices(
            items, train_indices, test_indices, case_idx_key='task_id'
        )

        assert len(train) == 2
        assert len(test) == 1
        assert len(unknown) == 0

    def test_missing_case_idx_raises_error(self):
        """Test that missing case_idx raises KeyError."""
        items = [
            {'case_idx': 0},
            {'no_case_idx': 1},  # Missing case_idx
        ]
        train_indices = {0}
        test_indices = {1}

        with pytest.raises(KeyError):
            partition_by_case_indices(items, train_indices, test_indices)

    def test_empty_items(self):
        """Test with empty items list."""
        train, test, unknown = partition_by_case_indices([], {0, 1}, {2, 3})

        assert len(train) == 0
        assert len(test) == 0
        assert len(unknown) == 0

    def test_empty_indices(self):
        """Test with empty train and test indices."""
        items = [{'case_idx': 0}, {'case_idx': 1}]

        train, test, unknown = partition_by_case_indices(items, set(), set())

        assert len(train) == 0
        assert len(test) == 0
        assert len(unknown) == 2

    def test_preserves_order(self):
        """Test that items maintain their relative order within partitions."""
        items = [
            {'case_idx': 0, 'order': 1},
            {'case_idx': 2, 'order': 2},
            {'case_idx': 1, 'order': 3},
            {'case_idx': 3, 'order': 4},
            {'case_idx': 0, 'order': 5},  # Duplicate case_idx
        ]
        train_indices = {0, 2}
        test_indices = {1, 3}

        train, test, unknown = partition_by_case_indices(items, train_indices, test_indices)

        assert len(train) == 3
        assert train[0]['order'] == 1
        assert train[1]['order'] == 2
        assert train[2]['order'] == 5
        assert len(test) == 2
        assert test[0]['order'] == 3
        assert test[1]['order'] == 4

    def test_large_dataset(self):
        """Test with larger dataset."""
        items = [{'case_idx': i, 'data': f'item_{i}'} for i in range(144)]
        train_indices = set(range(0, 72))
        test_indices = set(range(72, 144))

        train, test, unknown = partition_by_case_indices(items, train_indices, test_indices)

        assert len(train) == 72
        assert len(test) == 72
        assert len(unknown) == 0

    def test_sparse_indices(self):
        """Test with sparse/non-contiguous indices."""
        items = [
            {'case_idx': 10},
            {'case_idx': 50},
            {'case_idx': 100},
            {'case_idx': 143},
        ]
        train_indices = {10, 100}
        test_indices = {50, 143}

        train, test, unknown = partition_by_case_indices(items, train_indices, test_indices)

        assert len(train) == 2
        assert len(test) == 2
        assert len(unknown) == 0

    def test_overlapping_indices_train_wins(self):
        """Test that if an index is in both sets, train takes precedence."""
        items = [{'case_idx': 0}, {'case_idx': 1}]
        train_indices = {0, 1}
        test_indices = {1, 2}  # 1 is in both

        train, test, unknown = partition_by_case_indices(items, train_indices, test_indices)

        # Item with case_idx=1 should go to train (checked first)
        assert len(train) == 2
        assert len(test) == 0
        assert train[0]['case_idx'] == 0
        assert train[1]['case_idx'] == 1


class TestLoadCaseIndicesFromFinetuneData:
    """Tests for load_case_indices_from_finetune_data function."""

    def test_load_valid_jsonl(self, temp_test_dir):
        """Test loading case indices from valid JSONL file."""
        tmpdir = temp_test_dir
        # Create test data
        data_file = tmpdir / "test_data.jsonl"
        test_data = [
            {"case_idx": 0, "prompt": "test", "chosen": "a", "rejected": "b"},
            {"case_idx": 5, "prompt": "test", "chosen": "a", "rejected": "b"},
            {"case_idx": 10, "prompt": "test", "chosen": "a", "rejected": "b"},
            {"case_idx": 5, "prompt": "test", "chosen": "a", "rejected": "b"},  # Duplicate
        ]

        with open(data_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Load and verify
        indices = load_case_indices_from_finetune_data(str(data_file))

        assert indices == [0, 5, 10]  # Sorted and deduplicated

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_case_indices_from_finetune_data("/nonexistent/path.jsonl")

    def test_invalid_json(self, temp_test_dir):
        """Test that invalid JSON raises ValueError."""
        tmpdir = temp_test_dir
        data_file = tmpdir / "invalid.jsonl"
        with open(data_file, 'w') as f:
            f.write("not valid json\n")

        with pytest.raises(ValueError, match="JSON decode error"):
            load_case_indices_from_finetune_data(str(data_file))

    def test_missing_case_idx(self, temp_test_dir):
        """Test that missing case_idx field raises ValueError."""
        tmpdir = temp_test_dir
        data_file = tmpdir / "no_case_idx.jsonl"
        with open(data_file, 'w') as f:
            f.write('{"prompt": "test", "chosen": "a"}\n')

        with pytest.raises(ValueError, match="Missing 'case_idx'"):
            load_case_indices_from_finetune_data(str(data_file))

    def test_empty_file(self, temp_test_dir):
        """Test that empty file raises ValueError."""
        tmpdir = temp_test_dir
        data_file = tmpdir / "empty.jsonl"
        data_file.touch()

        with pytest.raises(ValueError, match="No examples found"):
            load_case_indices_from_finetune_data(str(data_file))

    def test_blank_lines(self, temp_test_dir):
        """Test that blank lines are skipped."""
        tmpdir = temp_test_dir
        data_file = tmpdir / "with_blanks.jsonl"
        with open(data_file, 'w') as f:
            f.write('{"case_idx": 1}\n')
            f.write('\n')  # Blank line
            f.write('{"case_idx": 2}\n')
            f.write('  \n')  # Whitespace line
            f.write('{"case_idx": 3}\n')

        indices = load_case_indices_from_finetune_data(str(data_file))
        assert indices == [1, 2, 3]


class TestInferDatasetType:
    """Test infer_dataset_type() function"""

    def test_source_models_with_split_suffix(self):
        """Test source models with _split suffix (no seed)"""
        assert infer_dataset_type("Qwen3-8B_safe_split", dpo_data_dir=TEST_DATA_DIR) == "safe"
        assert infer_dataset_type("Qwen3-8B_both_split", dpo_data_dir=TEST_DATA_DIR) == "both"
        assert infer_dataset_type("Qwen3-8B_help_split", dpo_data_dir=TEST_DATA_DIR) == "help"
        assert infer_dataset_type("Qwen3-8B_safe_diff2_split", dpo_data_dir=TEST_DATA_DIR) == "safe_diff2"
        assert infer_dataset_type("Qwen3-8B_help_diff2_split", dpo_data_dir=TEST_DATA_DIR) == "help_diff2"
        assert infer_dataset_type("Qwen3-8B_both_diff2_split", dpo_data_dir=TEST_DATA_DIR) == "both_diff2"

    def test_source_models_with_seed_suffix(self):
        """Test source models with _split suffix including seed"""
        assert infer_dataset_type("Qwen3-8B_help_s100_split", dpo_data_dir=TEST_DATA_DIR) == "help_s100"
        assert infer_dataset_type("Qwen3-8B_safe_s42_split", dpo_data_dir=TEST_DATA_DIR) == "safe_s42"
        assert infer_dataset_type("Qwen3-8B_safe_diff2_s100_split", dpo_data_dir=TEST_DATA_DIR) == "safe_diff2_s100"
        assert infer_dataset_type("Qwen3-8B_both_diff2_s100_split", dpo_data_dir=TEST_DATA_DIR) == "both_diff2_s100"

    def test_finetuned_models_no_seed(self):
        """Test finetuned model directory names without seed"""
        assert infer_dataset_type("models_trained_Qwen-8B_int4_safe_final", dpo_data_dir=TEST_DATA_DIR) == "safe"
        assert infer_dataset_type("models_trained_Qwen-8B_int4_help_final", dpo_data_dir=TEST_DATA_DIR) == "help"
        assert infer_dataset_type("models_trained_Qwen-8B_int4_both_final", dpo_data_dir=TEST_DATA_DIR) == "both"
        assert infer_dataset_type("models_trained_Qwen-8B_int4_safe_diff2_final", dpo_data_dir=TEST_DATA_DIR) == "safe_diff2"

    def test_finetuned_models_with_seed(self):
        """Test finetuned model directory names with seed"""
        assert infer_dataset_type("models_trained_Qwen-8B_int4_help_s100_final", dpo_data_dir=TEST_DATA_DIR) == "help_s100"
        assert infer_dataset_type("models_trained_Qwen-8B_int4_safe_s42_final", dpo_data_dir=TEST_DATA_DIR) == "safe_s42"
        assert infer_dataset_type("models_trained_Qwen-8B_int4_both_diff2_s100_final", dpo_data_dir=TEST_DATA_DIR) == "both_diff2_s100"

    def test_exclude_seed_option(self):
        """Test include_seed=False strips seed suffix"""
        assert infer_dataset_type("Qwen3-8B_safe_s42_split", include_seed=False, dpo_data_dir=TEST_DATA_DIR) == "safe"
        assert infer_dataset_type("models_trained_Qwen-8B_int4_help_s100_final", include_seed=False, dpo_data_dir=TEST_DATA_DIR) == "help"
        assert infer_dataset_type("Qwen3-8B_safe_diff2_s100_split", include_seed=False, dpo_data_dir=TEST_DATA_DIR) == "safe_diff2"

    def test_compound_datasets(self):
        """Test compound dataset names detected correctly"""
        assert infer_dataset_type("model_safe_diff2.jsonl", dpo_data_dir=TEST_DATA_DIR) == "safe_diff2"
        assert infer_dataset_type("model_help_diff2_checkpoint.jsonl", dpo_data_dir=TEST_DATA_DIR) == "help_diff2"
        assert infer_dataset_type("model_both_diff2_checkpoint.jsonl", dpo_data_dir=TEST_DATA_DIR) == "both_diff2"

    def test_source_model_returns_none(self):
        """Test that source model files (no dataset suffix) return None"""
        assert infer_dataset_type("some_random_file.jsonl", dpo_data_dir=TEST_DATA_DIR) is None
        assert infer_dataset_type("gpt-4o-mini_emu-gpt-4o-mini.jsonl", dpo_data_dir=TEST_DATA_DIR) is None
        assert infer_dataset_type("Qwen_Qwen3-8B_emu-Qwen3-32B_eval-Qwen3-32B_int4.jsonl", dpo_data_dir=TEST_DATA_DIR) is None

    def test_sequential_finetuning_returns_all_datasets(self):
        """Test that sequential finetuning model names return ALL dataset names concatenated"""
        # For sequential finetuning, we want all datasets concatenated (without config like beta)
        # Format: {source_model}_{stage1_dataset}_{stage1_config}_{stage2_dataset}_{stage2_config}_s{N}
        # Example: Qwen2.5-7B_help_gpt5m_beta-0.05_safe_gpt5m_beta-0.05_s42 -> help_gpt5m_safe_gpt5m_s42
        assert infer_dataset_type("Qwen2.5-7B_help_gpt5m_beta-0.05_safe_gpt5m_beta-0.05_s42", dpo_data_dir=TEST_DATA_DIR) == "help_gpt5m_safe_gpt5m_s42"
        assert infer_dataset_type("Qwen2.5-7B_safe_gpt5m_beta-0.02_help_gpt5m_beta-0.05_s0", dpo_data_dir=TEST_DATA_DIR) == "safe_gpt5m_help_gpt5m_s0"

        # Without gpt5m suffix
        assert infer_dataset_type("Llama-8B_help_beta-0.1_safe_beta-0.05_s42", dpo_data_dir=TEST_DATA_DIR) == "help_safe_s42"
        assert infer_dataset_type("Llama-8B_safe_beta-0.1_help_beta-0.05_s99", dpo_data_dir=TEST_DATA_DIR) == "safe_help_s99"

        # Three-stage sequential finetuning
        assert infer_dataset_type("Qwen-8B_help_gpt5m_beta-0.05_safe_gpt5m_beta-0.05_both_beta-0.01_s42", dpo_data_dir=TEST_DATA_DIR) == "help_gpt5m_safe_gpt5m_both_s42"

    def test_sequential_finetuning_with_diff2(self):
        """Test sequential finetuning with compound dataset names like safe_diff2"""
        assert infer_dataset_type("Qwen-8B_help_beta-0.05_safe_diff2_beta-0.05_s42", dpo_data_dir=TEST_DATA_DIR) == "help_safe_diff2_s42"
        assert infer_dataset_type("Qwen-8B_safe_diff2_beta-0.05_help_beta-0.05_s0", dpo_data_dir=TEST_DATA_DIR) == "safe_diff2_help_s0"


class TestGetDpoDataPath:
    """Test get_dpo_data_path() function"""

    def test_simple_datasets(self):
        """Test simple dataset types map to correct paths for existing files"""
        assert get_dpo_data_path("safe", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/safe.jsonl"
        assert get_dpo_data_path("help", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/help.jsonl"
        assert get_dpo_data_path("both", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/both.jsonl"

    def test_compound_datasets(self):
        """Test compound dataset types"""
        assert get_dpo_data_path("safe_diff2", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/safe_diff2.jsonl"
        assert get_dpo_data_path("help_diff2", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/help_diff2.jsonl"
        assert get_dpo_data_path("both_diff2", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/both_diff2.jsonl"

    def test_seed_suffix_ignored(self):
        """Test that seed suffixes are stripped"""
        assert get_dpo_data_path("safe_s42", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/safe.jsonl"
        assert get_dpo_data_path("help_s100", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/help.jsonl"
        assert get_dpo_data_path("safe_diff2_s100", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/safe_diff2.jsonl"
        assert get_dpo_data_path("help_diff2_s100", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/help_diff2.jsonl"
        assert get_dpo_data_path("both_diff2_s42", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/both_diff2.jsonl"

    def test_case_sensitive(self):
        """Test case sensitivity - dataset types must match file names exactly"""
        # Uppercase should fail
        with pytest.raises(ValueError, match="DPO data file not found"):
            get_dpo_data_path("SAFE", dpo_data_dir=TEST_DATA_DIR)
        with pytest.raises(ValueError, match="DPO data file not found"):
            get_dpo_data_path("Help", dpo_data_dir=TEST_DATA_DIR)
        # Lowercase should work
        assert get_dpo_data_path("safe", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/safe.jsonl"
        assert get_dpo_data_path("help", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/help.jsonl"

    def test_nonexistent_dataset_raises_error(self):
        """Test that nonexistent dataset files raise ValueError"""
        with pytest.raises(ValueError, match="DPO data file not found"):
            get_dpo_data_path("unknown", dpo_data_dir=TEST_DATA_DIR)
        with pytest.raises(ValueError, match="DPO data file not found"):
            get_dpo_data_path("invalid_type", dpo_data_dir=TEST_DATA_DIR)
        # diff2 alone doesn't exist (only safe_diff2, help_diff2, both_diff2)
        with pytest.raises(ValueError, match="DPO data file not found"):
            get_dpo_data_path("diff2", dpo_data_dir=TEST_DATA_DIR)

    def test_custom_dpo_data_dir(self, tmp_path):
        """Test using a custom dpo_data_dir"""
        # Create a temporary DPO data directory
        custom_dir = tmp_path / "custom_dpo"
        custom_dir.mkdir()

        # Create a test file
        test_file = custom_dir / "test.jsonl"
        test_file.write_text('{"test": "data"}')

        # Should find the file in the custom directory
        result = get_dpo_data_path("test", str(custom_dir))
        assert result == str(test_file)

    def test_missing_directory_raises_error(self):
        """Test that missing dpo_data_dir raises ValueError"""
        with pytest.raises(ValueError, match="DPO data directory not found"):
            get_dpo_data_path("safe", "/nonexistent/path")

    def test_sequential_finetuning_uses_last_dataset(self):
        """Test that sequential finetuning dataset types use the LAST dataset for the path"""
        # For sequential finetuning, infer_dataset_type returns concatenated datasets
        # get_dpo_data_path should use the last one to find the file
        assert get_dpo_data_path("help_gpt5m_safe_gpt5m", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/safe_gpt5m.jsonl"
        assert get_dpo_data_path("safe_gpt5m_help_gpt5m", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/help_gpt5m.jsonl"
        assert get_dpo_data_path("help_safe", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/safe.jsonl"
        assert get_dpo_data_path("safe_help", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/help.jsonl"
        # Three-stage sequential finetuning
        assert get_dpo_data_path("help_gpt5m_safe_gpt5m_both", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/both.jsonl"

    def test_sequential_finetuning_with_diff2(self):
        """Test sequential finetuning with compound dataset names like safe_diff2"""
        assert get_dpo_data_path("help_safe_diff2", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/safe_diff2.jsonl"
        assert get_dpo_data_path("safe_diff2_help", dpo_data_dir=TEST_DATA_DIR) == f"{TEST_DATA_DIR}/help.jsonl"


class TestExtractSeedFromPath:
    """Test extract_seed_from_path() function"""

    def test_extract_seed_from_filename(self):
        """Test extracting seed from filenames"""
        assert extract_seed_from_path("model_s100_split") == 100
        assert extract_seed_from_path("Qwen3-8B_safe_s42_split") == 42
        assert extract_seed_from_path("path/to/model_s999.jsonl") == 999

    def test_extract_seed_from_directory(self):
        """Test extracting seed from directory paths"""
        assert extract_seed_from_path("/path/to/models_trained_s100/file.jsonl") == 100
        assert extract_seed_from_path("Qwen-8B_help_s42") == 42

    def test_no_seed_returns_none(self):
        """Test that missing seed returns None"""
        assert extract_seed_from_path("model_no_seed.jsonl") is None
        assert extract_seed_from_path("gpt-4o-mini.jsonl") is None
        assert extract_seed_from_path("random_file") is None

    def test_duplicate_same_seed_ok(self):
        """Test that multiple _s{N} with the same value is fine"""
        assert extract_seed_from_path("model_s42_most_s42") == 42

    def test_ambiguous_seeds_raises_error(self):
        """Test that multiple _s{N} with different values raises ValueError"""
        with pytest.raises(ValueError, match="Ambiguous seed"):
            extract_seed_from_path("model_s100_s200")


class TestFixedSplitMode:
    """Test suite for fixed 144-case split mode (compute_test_indices)."""

    def test_fixed_split_returns_72_test_indices(self):
        """Test that fixed split returns exactly 72 test indices (half of 144)."""
        test_indices = compute_test_indices(seed=42)
        assert len(test_indices) == TOOLEMU_FULL_DATASET_SIZE // 2, \
            f"Expected {TOOLEMU_FULL_DATASET_SIZE // 2} test indices, got {len(test_indices)}"

    def test_fixed_split_indices_in_valid_range(self):
        """Test that all test indices are in valid range [0, 143]."""
        test_indices = compute_test_indices(seed=42)
        for idx in test_indices:
            assert 0 <= idx < TOOLEMU_FULL_DATASET_SIZE, \
                f"Test index {idx} is outside valid range [0, {TOOLEMU_FULL_DATASET_SIZE - 1}]"

    def test_fixed_split_deterministic(self):
        """Test that fixed split is deterministic with same seed."""
        result1 = compute_test_indices(seed=42)
        result2 = compute_test_indices(seed=42)
        assert result1 == result2, "Fixed split should be deterministic"

    def test_fixed_split_different_seeds(self):
        """Test that different seeds produce different fixed splits."""
        result1 = compute_test_indices(seed=42)
        result2 = compute_test_indices(seed=123)
        assert result1 != result2, "Different seeds should produce different splits"

    def test_fixed_split_sorted(self):
        """Test that returned test indices are sorted."""
        test_indices = compute_test_indices(seed=42)
        assert test_indices == sorted(test_indices), "Test indices should be sorted"

    def test_fixed_split_no_duplicates(self):
        """Test that there are no duplicate indices."""
        test_indices = compute_test_indices(seed=42)
        assert len(test_indices) == len(set(test_indices)), "Test indices should have no duplicates"

    def test_train_test_no_overlap(self):
        """Test that train and test sets don't overlap."""
        test_indices = set(compute_test_indices(seed=42))
        train_indices = set(range(TOOLEMU_FULL_DATASET_SIZE)) - test_indices

        overlap = train_indices & test_indices
        assert len(overlap) == 0, f"Train and test sets should not overlap, found {overlap}"

    def test_train_test_cover_all_cases(self):
        """Test that train + test cover all 144 cases."""
        test_indices = set(compute_test_indices(seed=42))
        train_indices = set(range(TOOLEMU_FULL_DATASET_SIZE)) - test_indices

        all_indices = train_indices | test_indices
        expected = set(range(TOOLEMU_FULL_DATASET_SIZE))
        assert all_indices == expected, "Train + test should cover all cases"

    def test_fixed_split_case_consistency(self):
        """Test that same case gets same assignment with same seed.

        This is the key property: case 42 is always train or always test
        with the same seed, regardless of which dataset the case appears in.
        """
        # Get fixed split
        fixed_test = set(compute_test_indices(seed=42))

        # Verify case 42's assignment is deterministic
        case_42_in_test = 42 in fixed_test

        # Run multiple times to verify consistency
        for _ in range(10):
            result = compute_test_indices(seed=42)
            assert (42 in result) == case_42_in_test, \
                "Case assignment should be consistent across calls"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
