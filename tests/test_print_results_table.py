#!/usr/bin/env python3
"""Tests for print_results_table.py"""

import sys
from pathlib import Path

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "analysis"))

from print_results_table import (
    split_sort_key,
    get_model_type,
)
from utils.toolemu_utils import parse_toolemu_filename
from utils.train_utils import infer_dataset_type as infer_dataset_split

# Test fixture directory containing mock dataset files
TEST_DATA_DIR = str(Path(__file__).parent / 'fixtures' / 'dpo_data')


class TestInferDatasetSplit:
    """Test the infer_dataset_split function"""

    def test_source_models_with_split_suffix(self):
        """Test source models with _split suffix (no seed)"""
        assert infer_dataset_split("Qwen3-8B_safe_split", dpo_data_dir=TEST_DATA_DIR) == "safe"
        assert infer_dataset_split("Qwen3-8B_both_split", dpo_data_dir=TEST_DATA_DIR) == "both"
        assert infer_dataset_split("Qwen3-8B_help_split", dpo_data_dir=TEST_DATA_DIR) == "help"
        assert infer_dataset_split("Qwen3-8B_safe_diff2_split", dpo_data_dir=TEST_DATA_DIR) == "safe_diff2"
        assert infer_dataset_split("Qwen3-8B_help_diff2_split", dpo_data_dir=TEST_DATA_DIR) == "help_diff2"

    def test_source_models_with_seed_suffix(self):
        """Test source models with seed in suffix"""
        assert infer_dataset_split("Qwen3-8B_safe_s42_split", dpo_data_dir=TEST_DATA_DIR) == "safe_s42"
        assert infer_dataset_split("Qwen3-8B_safe_diff2_s100_split", dpo_data_dir=TEST_DATA_DIR) == "safe_diff2_s100"

    def test_compound_dataset_with_seed_regression(self):
        """Regression test: compound dataset with seed should preserve full pattern."""
        # This was the bug - both_diff2_s100 was incorrectly matched as diff2_s100
        assert infer_dataset_split("Qwen3-8B_both_diff2_s100_split", dpo_data_dir=TEST_DATA_DIR) == "both_diff2_s100"
        assert infer_dataset_split("Yi-1.5-9B_both_diff2_s100_split", dpo_data_dir=TEST_DATA_DIR) == "both_diff2_s100"

    def test_finetuned_models_with_seed(self):
        """Test finetuned models with seed in directory name (identified by seed suffix)"""
        assert infer_dataset_split("Qwen-8B_int4_safe_s42", dpo_data_dir=TEST_DATA_DIR) == "safe_s42"
        assert infer_dataset_split("Qwen-8B_int4_both_s0", dpo_data_dir=TEST_DATA_DIR) == "both_s0"
        assert infer_dataset_split("Qwen-8B_int4_safe_diff2_s100", dpo_data_dir=TEST_DATA_DIR) == "safe_diff2_s100"
        assert infer_dataset_split("Yi-1.5-9B_int4_both_diff2_s42", dpo_data_dir=TEST_DATA_DIR) == "both_diff2_s42"

    def test_source_models_with_dpo_prefix(self):
        """Test source models with dpo prefix in split suffix"""
        assert infer_dataset_split("Qwen3-8B_dpo_help_split", dpo_data_dir=TEST_DATA_DIR) == "dpo_help"
        assert infer_dataset_split("Qwen3-8B_dpo_safe_diff2_split", dpo_data_dir=TEST_DATA_DIR) == "dpo_safe_diff2"

    def test_complex_finetuned_configs(self):
        """Test finetuned models with complex training configs"""
        assert infer_dataset_split("Qwen-8B_int4_both_b-1_ml-8192_precomp_s42", dpo_data_dir=TEST_DATA_DIR) == "both_s42"


class TestParseToolemuFilename:
    """Test parse_toolemu_filename function for extracting models from filenames"""

    def test_source_model_filename(self):
        """Test extracting models from source model filenames"""
        filename = "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_unified_report.json"
        result = parse_toolemu_filename(filename)
        assert result['agent_model'] == "Qwen_Qwen3-8B"
        assert result['emu_model'] == "Qwen_Qwen3-32B"
        assert result['eval_model'] == "Qwen_Qwen3-32B"
        assert result['quantization'] == "int4"

    def test_finetuned_model(self):
        """Test extracting models from finetuned model filename (identified by seed)"""
        filename = "Qwen-8B_int4_safe_s42_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_unified_report.json"
        result = parse_toolemu_filename(filename)
        assert result['agent_model'] == "Qwen-8B_int4_safe_s42"
        assert result['emu_model'] == "Qwen_Qwen3-32B"
        assert result['eval_model'] == "Qwen_Qwen3-32B"
        assert result['quantization'] == "int4"

    def test_finetuned_model_help(self):
        """Test extracting models from finetuned model filename with help dataset"""
        filename = "Qwen-8B_help_s42_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_unified_report.json"
        result = parse_toolemu_filename(filename)
        assert result['agent_model'] == "Qwen-8B_help_s42"
        assert result['emu_model'] == "Qwen_Qwen3-32B"
        assert result['eval_model'] == "Qwen_Qwen3-32B"

    def test_model_with_range(self):
        """Test parsing filename with task range"""
        filename = "Qwen-8B_help_s42_emu-Qwen_Qwen3-8B_eval-Qwen_Qwen3-32B_int4_r0-72_unified_report.json"
        result = parse_toolemu_filename(filename)
        assert result['agent_model'] == "Qwen-8B_help_s42"
        # range values may be returned as strings or ints depending on implementation
        assert int(result['range_start']) == 0
        assert int(result['range_end']) == 72

class TestSplitSortKey:
    """Test split_sort_key function for sorting dataset splits"""

    def test_none_sorts_first(self):
        """Test that None (source models without split) sorts first"""
        assert split_sort_key(None) < split_sort_key('both')
        assert split_sort_key(None) < split_sort_key('safe')
        assert split_sort_key(None) < split_sort_key('help')

    def test_alphabetical_ordering(self):
        """Test that datasets sort alphabetically"""
        splits = ['safe', 'both', 'help', 'help_diff2']
        sorted_splits = sorted(splits, key=split_sort_key)
        # Alphabetical: both, help, help_diff2, safe
        expected = ['both', 'help', 'help_diff2', 'safe']
        assert sorted_splits == expected

    def test_compound_datasets_alphabetical(self):
        """Test compound datasets sort alphabetically"""
        splits = ['both_diff2', 'safe_diff2', 'help_diff2']
        sorted_splits = sorted(splits, key=split_sort_key)
        # Alphabetical: both_diff2, help_diff2, safe_diff2
        expected = ['both_diff2', 'help_diff2', 'safe_diff2']
        assert sorted_splits == expected

    def test_seed_variants_group_with_base(self):
        """Test that seed variants group with their base dataset"""
        splits = ['both', 'both_s100', 'safe', 'both_s42']
        sorted_splits = sorted(splits, key=split_sort_key)
        # both and its seeds grouped, then safe
        expected = ['both', 'both_s100', 'both_s42', 'safe']
        assert sorted_splits == expected

    def test_compound_seed_variants(self):
        """Test seed variants of compound datasets"""
        splits = ['both_diff2', 'both_diff2_s100', 'safe_diff2']
        sorted_splits = sorted(splits, key=split_sort_key)
        # both_diff2 and its seed grouped, then safe_diff2
        expected = ['both_diff2', 'both_diff2_s100', 'safe_diff2']
        assert sorted_splits == expected

    def test_mixed_datasets_and_ss(self):
        """Test mix of base datasets and seed variants"""
        splits = ['safe_diff2', 'both', 'both_diff2_s100', 'safe', 'both_s42', 'help']
        sorted_splits = sorted(splits, key=split_sort_key)
        # Alphabetically: both+seeds, both_diff2+seeds, help, safe, safe_diff2
        expected = ['both', 'both_s42', 'both_diff2_s100', 'help', 'safe', 'safe_diff2']
        assert sorted_splits == expected

    def test_empty_list(self):
        """Test sorting empty list"""
        assert sorted([], key=split_sort_key) == []

    def test_single_split(self):
        """Test sorting single split"""
        assert sorted(['safe'], key=split_sort_key) == ['safe']
        assert sorted(['safe_s42'], key=split_sort_key) == ['safe_s42']

    def test_sort_key_structure(self):
        """Test that sort key returns expected string structure"""
        key = split_sort_key('both_s100')
        assert isinstance(key, str)
        assert key == 'both'  # Seed stripped for sorting

    def test_none_key_structure(self):
        """Test that None returns expected string structure"""
        key = split_sort_key(None)
        assert isinstance(key, str)
        assert key == ''  # Empty string sorts first

    def test_dpo_prefix_sorting(self):
        """Test that dpo prefixed splits sort alphabetically by base dataset"""
        splits = ['dpo_help', 'help', 'dpo_both', 'both']
        sorted_splits = sorted(splits, key=split_sort_key)
        # After stripping prefix, all sort by base name: both, both, help, help
        # Python's sort is stable, so original order is preserved for equal keys
        expected = ['dpo_both', 'both', 'dpo_help', 'help']
        assert sorted_splits == expected

    def test_dpo_prefix_with_compound_datasets(self):
        """Test dpo prefixes with compound datasets"""
        splits = ['dpo_help_diff2', 'help_diff2', 'dpo_both']
        sorted_splits = sorted(splits, key=split_sort_key)
        # After stripping prefix, sort by: both, help_diff2, help_diff2
        # Stable sort preserves original order for equal keys
        expected = ['dpo_both', 'dpo_help_diff2', 'help_diff2']
        assert sorted_splits == expected

    def test_dpo_prefix_with_ss(self):
        """Test dpo prefixes with seed variants"""
        splits = ['dpo_help', 'help', 'dpo_help_s42']
        sorted_splits = sorted(splits, key=split_sort_key)
        # After stripping prefix and seeds, all sort by "help"
        # Stable sort preserves original order
        expected = ['dpo_help', 'help', 'dpo_help_s42']
        assert sorted_splits == expected

    def test_dpo_key_structure(self):
        """Test that dpo prefixed splits return correct key structure"""
        key = split_sort_key('dpo_help')
        assert isinstance(key, str)
        assert key == 'help'  # Prefix stripped

    def test_sequential_finetuning_alphabetical(self):
        """Test that sequential finetuning datasets sort alphabetically by full name."""
        # Sequential finetuning names sort alphabetically
        splits = ['help_gpt5m', 'safe_gpt5m', 'help_gpt5m_safe_gpt5m', 'safe_gpt5m_help_gpt5m']
        sorted_splits = sorted(splits, key=split_sort_key)
        # Alphabetical order
        expected = ['help_gpt5m', 'help_gpt5m_safe_gpt5m', 'safe_gpt5m', 'safe_gpt5m_help_gpt5m']
        assert sorted_splits == expected


class TestSplitDisplayLogic:
    """Integration tests to verify all splits in data are displayed"""

    def test_all_splits_included_in_sorting(self):
        """Test that sorting doesn't drop any splits from the input"""
        splits = ['both', 'safe', 'both_diff2_s100', 'safe_diff2_s200', 'help']
        sorted_splits = sorted(splits, key=split_sort_key)
        # Verify no splits were dropped
        assert len(sorted_splits) == len(splits)
        assert set(sorted_splits) == set(splits)

    def test_seed_variants_grouped_with_base(self):
        """Test that seed variants are grouped with their base dataset"""
        splits = ['safe', 'both_diff2_s100', 'safe_s42']
        sorted_splits = sorted(splits, key=split_sort_key)

        # Ensure seed-based splits are present
        assert 'both_diff2_s100' in sorted_splits
        assert 'safe_s42' in sorted_splits
        # Safe and its seed should be adjacent
        assert sorted_splits.index('safe_s42') == sorted_splits.index('safe') + 1

    def test_all_dataset_types_represented(self):
        """Test comprehensive list of all dataset split types"""
        # All possible dataset splits
        all_splits = [
            'both', 'safe', 'help', 'safe_diff2', 'help_diff2', 'both_diff2',
            'both_s100', 'safe_s42', 'safe_diff2_s100'
        ]
        sorted_splits = sorted(all_splits, key=split_sort_key)

        # Verify all are present
        assert len(sorted_splits) == len(all_splits)
        for split in all_splits:
            assert split in sorted_splits

        # Verify alphabetical ordering (seeds stripped for sorting key)
        # After stripping seeds: both, both, both_diff2, help, help_diff2, safe, safe, safe_diff2, safe_diff2
        # Stable sort preserves original order for equal keys
        expected = [
            'both', 'both_s100', 'both_diff2',
            'help', 'help_diff2',
            'safe', 'safe_s42', 'safe_diff2', 'safe_diff2_s100'
        ]
        assert sorted_splits == expected

    def test_compound_dataset_with_seed_not_dropped(self):
        """Regression test: compound dataset with seed should not be dropped or misparsed"""
        # This test would catch the bug where both_diff2_s100 was incorrectly matched
        splits = ['both_diff2_s100', 'both', 'safe']
        sorted_splits = sorted(splits, key=split_sort_key)

        # Should have all 3 splits
        assert len(sorted_splits) == 3
        assert 'both_diff2_s100' in sorted_splits
        # After stripping seeds: both_diff2, both, safe
        # Stable sort preserves original order for equal keys
        assert sorted_splits == ['both', 'both_diff2_s100', 'safe']

    def test_dpo_splits_grouping(self):
        """Test that dpo split variants group correctly with base dataset"""
        splits = ['dpo_help', 'help', 'both', 'dpo_both']
        sorted_splits = sorted(splits, key=split_sort_key)

        # After stripping prefix: both, both, help, help
        # Stable sort preserves original order for equal keys
        assert sorted_splits == ['both', 'dpo_both', 'dpo_help', 'help']


class TestFullSortingBehavior:
    """Integration tests for full sorting logic including base vs finetuned models"""

    def test_base_and_finetuned_models_sort_order(self):
        """Test that each source model is immediately followed by its corresponding finetuned model"""
        # Simulate results with dataset_split and model_name
        # Finetuned models are identified by seed suffix (_s{N})
        results = [
            {'model_name': 'Qwen-8B_int4_help_s42', 'dataset_split': 'help_s42'},
            {'model_name': 'Qwen3-8B_dpo_help_split', 'dataset_split': 'dpo_help'},
        ]

        # Sort using the corrected logic
        sorted_results = sorted(results, key=lambda x: (
            split_sort_key(x['dataset_split']),  # base dataset (help)
            ('_split' not in x['model_name']),  # source models first (False < True)
        ))

        # Expected order:
        # 1. Qwen3-8B_dpo_help_split (base, dpo_help) - help + base
        # 2. Qwen-8B_int4_help_s42 (finetuned, help_s42) - help + finetuned
        expected_order = [
            'Qwen3-8B_dpo_help_split',
            'Qwen-8B_int4_help_s42',
        ]

        actual_order = [r['model_name'] for r in sorted_results]
        assert actual_order == expected_order

    def test_multiple_datasets_with_dpo_variants(self):
        """Test sorting with multiple datasets, each having dpo variants"""
        # Finetuned models are identified by seed suffix (_s{N})
        results = [
            {'model_name': 'Qwen-8B_int4_help_s42', 'dataset_split': 'help_s42'},
            {'model_name': 'Qwen3-8B_dpo_help_split', 'dataset_split': 'dpo_help'},
            {'model_name': 'Qwen-8B_int4_both_s42', 'dataset_split': 'both_s42'},
            {'model_name': 'Qwen3-8B_dpo_both_split', 'dataset_split': 'dpo_both'},
        ]

        sorted_results = sorted(results, key=lambda x: (
            split_sort_key(x['dataset_split']),  # base dataset
            ('_split' not in x['model_name']),  # source models first
        ))

        expected_order = [
            # "both" + "dpo" group: base then finetuned
            'Qwen3-8B_dpo_both_split',
            'Qwen-8B_int4_both_s42',
            # "help" + "dpo" group: base then finetuned
            'Qwen3-8B_dpo_help_split',
            'Qwen-8B_int4_help_s42',
        ]

        actual_order = [r['model_name'] for r in sorted_results]
        assert actual_order == expected_order


class TestGetModelType:
    """Test get_model_type function for display type strings.

    Finetuned models are identified by having a 'dpo_' prefix.
    """

    def test_finetuned_with_dpo_prefix(self):
        """Test finetuned models (with dpo_ prefix) return 'dpo'"""
        result = {'model_name': 'dpo_Qwen-8B_int4_safe_s42', 'training_config': ''}
        assert get_model_type(result) == "dpo"

        result = {'model_name': 'dpo_Qwen2.5-7B_safe_gpt5m_s0', 'training_config': ''}
        assert get_model_type(result) == "dpo"

    def test_source_models(self):
        """Test source models (no dpo_ prefix) return 'base'"""
        result = {'model_name': 'Qwen_Qwen3-8B', 'training_config': ''}
        assert get_model_type(result) == "base"

        result = {'model_name': 'gpt-4o-mini', 'training_config': ''}
        assert get_model_type(result) == "base"

        result = {'model_name': 'Qwen-8B_int4_safe', 'training_config': ''}  # No dpo_ prefix = source
        assert get_model_type(result) == "base"

    def test_finetuned_with_training_config(self):
        """Test that finetuned models with training_config show config"""
        result = {'model_name': 'dpo_Qwen-8B_int4_safe_s42', 'training_config': 'β=0.05'}
        assert get_model_type(result) == "dpo(β=0.05)"
