#!/usr/bin/env python3
"""
Unit tests for src/utils/model_name_utils.py

Tests cover:
- extract_source_model() function
- nickname_to_directory() function
- sanitize_model_name() function
- model_sort_key() function
"""

import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.model_name_utils import (
    extract_source_model,
    sanitize_model_name,
    nickname_to_directory,
    model_sort_key,
)


class TestExtractSourceModel:
    """Test extract_source_model function for extracting source model names.

    Finetuned models are identified by having a 'dpo_' prefix.
    The source model is extracted by stripping the prefix and dataset/seed suffixes.
    """

    def test_finetuned_with_dpo_prefix(self):
        """Test extracting source model from finetuned models (identified by dpo_ prefix)"""
        # Basic finetuned models with dpo_ prefix
        assert extract_source_model("dpo_Qwen3-8B_int4_safe_s42") == "Qwen3-8B"
        assert extract_source_model("dpo_Qwen3-32B_int4_diff2_s100") == "Qwen3-32B"
        assert extract_source_model("dpo_Llama-8B_int4_safe_s0") == "Llama3.1-8B"  # Llama-8B expands to Llama3.1-8B
        assert extract_source_model("dpo_Qwen2.5-7B_safe_gpt5m_s42") == "Qwen2.5-7B"

    def test_finetuned_with_complex_config(self):
        """Test extracting source model from finetuned models with complex configs"""
        assert extract_source_model("dpo_Qwen3-8B_int4_1A6000_all_b-1_ml-8192_precomp_s42") == "Qwen3-8B"
        assert extract_source_model("dpo_Qwen2.5-7B_safe_gpt5m_beta-0.05_s42") == "Qwen2.5-7B"

    def test_sequential_finetuning(self):
        """Test extracting source model from sequential finetuning (multiple training stages)"""
        # Sequential finetuning: model trained on dataset1, then trained on dataset2
        # Format: dpo_{source_model}_{stage1_dataset}_{stage1_config}_{stage2_dataset}_{stage2_config}_s{N}
        assert extract_source_model("dpo_Qwen2.5-7B_most_gpt5m_beta-0.05_safe_gpt5m_beta-0.05_s42") == "Qwen2.5-7B"
        assert extract_source_model("dpo_Qwen2.5-7B_safe_gpt5m_beta-0.02_most_gpt5m_beta-0.05_s0") == "Qwen2.5-7B"
        assert extract_source_model("dpo_Llama-8B_most_beta-0.1_safe_beta-0.05_s99") == "Llama3.1-8B"  # Llama-8B expands to Llama3.1-8B
        # Three-stage sequential finetuning
        assert extract_source_model("dpo_Qwen3-8B_most_gpt5m_beta-0.05_safe_gpt5m_beta-0.05_really_gpt5m_beta-0.01_s42") == "Qwen3-8B"

    def test_source_model_directories(self):
        """Test extracting model name from source model directories (org_model format)"""
        assert extract_source_model("Qwen_Qwen3-8B") == "Qwen3-8B"
        assert extract_source_model("Qwen_Qwen3-32B") == "Qwen3-32B"
        assert extract_source_model("Qwen_Qwen2.5-7B-Instruct") == "Qwen2.5-7B"
        assert extract_source_model("Qwen_Qwen2.5-32B-Instruct") == "Qwen2.5-32B"
        assert extract_source_model("meta-llama_Llama-3.1-8B-Instruct") == "Llama3.1-8B"

    def test_partitioned_source_models(self):
        """Test extracting model name from partitioned source models (has _s{N} but no dpo_ prefix)"""
        # These are source models partitioned by seed, NOT finetuned models
        assert extract_source_model("Qwen_Qwen3-8B_s2") == "Qwen3-8B"
        assert extract_source_model("Qwen_Qwen2.5-7B-Instruct_s42") == "Qwen2.5-7B"
        assert extract_source_model("meta-llama_Llama-3.1-8B-Instruct_s0") == "Llama3.1-8B"

    def test_regular_model_names(self):
        """Test that regular model names are returned as-is"""
        assert extract_source_model("Qwen3-8B") == "Qwen3-8B"
        assert extract_source_model("gpt-4o-mini") == "gpt-4o-mini"

    def test_edge_cases(self):
        """Test edge cases that shouldn't match org_model pattern"""
        # Underscore but not org_model pattern (has numbers in org part)
        assert extract_source_model("gpt-4_turbo") == "gpt-4_turbo"
        # No uppercase in model part
        assert extract_source_model("some_lowercase") == "some_lowercase"


class TestNicknameToDirectory:
    """Test nickname_to_directory function for converting nicknames to directory names."""

    def test_qwen_models(self):
        """Test Qwen model nicknames"""
        assert nickname_to_directory("Qwen2.5-7B") == "Qwen_Qwen2.5-7B-Instruct"
        assert nickname_to_directory("Qwen2.5-32B") == "Qwen_Qwen2.5-32B-Instruct"
        assert nickname_to_directory("Qwen3-8B") == "Qwen_Qwen3-8B"
        assert nickname_to_directory("Qwen3-32B") == "Qwen_Qwen3-32B"

    def test_llama_models(self):
        """Test Llama model nicknames"""
        assert nickname_to_directory("Llama3.1-8B") == "meta-llama_Llama-3.1-8B-Instruct"
        assert nickname_to_directory("Llama3.1-70B") == "meta-llama_Llama-3.1-70B-Instruct"
        assert nickname_to_directory("Llama3.3-70B") == "meta-llama_Llama-3.3-70B-Instruct"

    def test_yi_models(self):
        """Test Yi model nicknames"""
        assert nickname_to_directory("Yi-1.5-9B") == "01-ai_Yi-1.5-9B-Chat-16K"
        assert nickname_to_directory("Yi-1.5-34B") == "01-ai_Yi-1.5-34B-Chat-16K"
        assert nickname_to_directory("Yi-34B") == "01-ai_Yi-34B-Chat"

    def test_unknown_returned_as_is(self):
        """Test that unknown names are returned unchanged"""
        assert nickname_to_directory("gpt-4o-mini") == "gpt-4o-mini"
        assert nickname_to_directory("Qwen-8B_int4_s42") == "Qwen-8B_int4_s42"  # Finetuned model name
        assert nickname_to_directory("Qwen_Qwen3-8B") == "Qwen_Qwen3-8B"  # Already a directory


class TestSanitizeModelName:
    """Test sanitize_model_name() function"""

    def test_source_model_sanitization(self):
        """Test basic model name sanitization (replace / and spaces)"""
        assert sanitize_model_name("Qwen/Qwen3-8B") == "Qwen_Qwen3-8B"
        assert sanitize_model_name("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama_Llama-3.1-8B-Instruct"
        assert sanitize_model_name("gpt-4o-mini") == "gpt-4o-mini"

    def test_source_model_with_spaces(self):
        """Test sanitization of model names with spaces"""
        assert sanitize_model_name("My Model Name") == "My_Model_Name"
        assert sanitize_model_name("path/to/my model") == "path_to_my_model"

    def test_dpo_final_model(self):
        """Test DPO final model - should have dpo_ prefix and exclude _final suffix"""
        result = sanitize_model_name("Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_most/final")
        assert result == "dpo_Qwen-8B_int4_most"

        result = sanitize_model_name("meta-llama/Llama-3.1-8B+dpo_output/Llama-8B_int4_safe/final")
        assert result == "dpo_Llama-8B_int4_safe"

        result = sanitize_model_name("Qwen/Qwen3-32B+dpo_output/Qwen-32B_int8_really/final")
        assert result == "dpo_Qwen-32B_int8_really"

    def test_dpo_checkpoint_model(self):
        """Test DPO checkpoint model - should have dpo_ prefix and include checkpoint number"""
        result = sanitize_model_name("Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_most/checkpoints/checkpoint-1600")
        assert result == "dpo_Qwen-8B_int4_most_1600"

        result = sanitize_model_name("meta-llama/Llama-3.1-8B+dpo_output/Llama-8B_int4_safe/checkpoints/checkpoint-500")
        assert result == "dpo_Llama-8B_int4_safe_500"

    def test_dpo_recent_checkpoint(self):
        """Test DPO recent checkpoint format"""
        result = sanitize_model_name("Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_most/recent_checkpoints/checkpoint-3200")
        assert result == "dpo_Qwen-8B_int4_most_3200"

    def test_dpo_merged_final_model(self):
        """Test DPO merged model - should have dpo_ prefix"""
        result = sanitize_model_name("Qwen/Qwen3-8B+dpo_merged/Qwen-8B_help_s42/final")
        assert result == "dpo_Qwen-8B_help_s42"

        result = sanitize_model_name("meta-llama/Llama-3.1-8B+dpo_merged/Llama-8B_safe_s0/final")
        assert result == "dpo_Llama-8B_safe_s0"

    def test_dpo_adapters_final_model(self):
        """Test DPO adapters model - should have dpo_ prefix"""
        result = sanitize_model_name("Qwen/Qwen3-8B+dpo_adapters/Qwen-8B_help_s42/final")
        assert result == "dpo_Qwen-8B_help_s42"

    def test_dpo_fallback_pattern(self):
        """Test DPO models that don't match expected patterns fall back to simple replacement"""
        # Missing known adapter prefix - should use fallback (no dpo_ prefix)
        result = sanitize_model_name("Qwen/Qwen3-8B+custom_path/model_dir/final")
        assert result == "custom_path_model_dir_final"

        # No final or checkpoint pattern - extracts directory name from dpo_output/ with dpo_ prefix
        result = sanitize_model_name("Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_most/some_other_subdir")
        assert result == "dpo_Qwen-8B_int4_most"

        # Epoch checkpoint format - extracts directory name from dpo_output/ with dpo_ prefix
        result = sanitize_model_name("Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_most/epoch_checkpoints/checkpoint-epoch-2")
        assert result == "dpo_Qwen-8B_int4_most"

    def test_dpo_complex_directory_names(self):
        """Test DPO models with complex directory names"""
        # Directory with multiple components
        result = sanitize_model_name("Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_most_b-2_ml-16384/final")
        assert result == "dpo_Qwen-8B_int4_most_b-2_ml-16384"

        # Checkpoint with complex directory
        result = sanitize_model_name("Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_most_b-2_ml-16384/checkpoints/checkpoint-800")
        assert result == "dpo_Qwen-8B_int4_most_b-2_ml-16384_800"

    def test_no_plus_sign(self):
        """Test models without + (not DPO models)"""
        assert sanitize_model_name("regular/model/path") == "regular_model_path"
        assert sanitize_model_name("Qwen_Qwen3-8B") == "Qwen_Qwen3-8B"

    def test_direct_merged_model_path(self):
        """Test direct dpo_merged/ paths (without + syntax) get dpo_ prefix"""
        # Basic merged model path
        assert sanitize_model_name("dpo_merged/Llama-8B_help_s42") == "dpo_Llama-8B_help_s42"
        assert sanitize_model_name("dpo_merged/Qwen-8B_safe_gpt5m_s0") == "dpo_Qwen-8B_safe_gpt5m_s0"

        # Complex names with beta values
        assert sanitize_model_name("dpo_merged/Llama-8B_help_gpt5m_beta-0.05_s0") == "dpo_Llama-8B_help_gpt5m_beta-0.05_s0"

        # dpo_output paths (should also work)
        assert sanitize_model_name("dpo_output/Llama-8B_help_s42") == "dpo_Llama-8B_help_s42"

        # dpo_trained paths
        assert sanitize_model_name("dpo_trained/Qwen-8B_safe_s1") == "dpo_Qwen-8B_safe_s1"

        # dpo_adapters paths
        assert sanitize_model_name("dpo_adapters/Llama-8B_help_s0") == "dpo_Llama-8B_help_s0"

        # With trailing subpath (should ignore it)
        assert sanitize_model_name("dpo_merged/Llama-8B_help_s42/final") == "dpo_Llama-8B_help_s42"
        assert sanitize_model_name("dpo_merged/Llama-8B_help_s42/some/extra/path") == "dpo_Llama-8B_help_s42"

    def test_empty_string(self):
        """Test edge case of empty string"""
        assert sanitize_model_name("") == ""

    def test_already_sanitized(self):
        """Test that already sanitized names pass through correctly"""
        assert sanitize_model_name("Qwen_Qwen3-8B") == "Qwen_Qwen3-8B"
        assert sanitize_model_name("already_sanitized_name") == "already_sanitized_name"


class TestModelSortKey:
    """Tests for model_sort_key function."""

    def test_sort_by_series(self):
        """Models should be grouped by series (Llama, Qwen, etc.)."""
        models = ["Qwen3-8B", "Llama3.1-8B", "Phi-4"]
        sorted_models = sorted(models, key=model_sort_key)
        # Llama should come before Phi, which should come before Qwen
        assert sorted_models.index("Llama3.1-8B") < sorted_models.index("Phi-4")
        assert sorted_models.index("Phi-4") < sorted_models.index("Qwen3-8B")

    def test_sort_by_size_within_version(self):
        """Within a version, models should be sorted by size (smaller first)."""
        models = ["Llama3.1-70B", "Llama3.1-8B", "Llama3.1-3B"]
        sorted_models = sorted(models, key=model_sort_key)
        assert sorted_models == ["Llama3.1-3B", "Llama3.1-8B", "Llama3.1-70B"]

    def test_sort_qwen_by_version_then_size(self):
        """Qwen models sorted by version (2.5 before 3), then by size."""
        models = ["Qwen3-32B", "Qwen2.5-7B", "Qwen3-8B", "Qwen2.5-32B"]
        sorted_models = sorted(models, key=model_sort_key)
        assert sorted_models == ["Qwen2.5-7B", "Qwen2.5-32B", "Qwen3-8B", "Qwen3-32B"]

    def test_gpt_models_at_end(self):
        """GPT models should always be sorted to the end."""
        models = ["gpt-4o", "Llama3.1-8B", "gpt-5-mini", "Qwen3-8B"]
        sorted_models = sorted(models, key=model_sort_key)
        # Non-GPT models should come first
        assert sorted_models[:2] == ["Llama3.1-8B", "Qwen3-8B"]
        # GPT models at the end
        assert set(sorted_models[2:]) == {"gpt-4o", "gpt-5-mini"}

    def test_gpt_models_sorted_among_themselves(self):
        """GPT models should be sorted: 4o-mini, 4o, 5-nano, 5-mini, 5, 5.1-mini, 5.1, 5.2."""
        models = ["gpt-5", "gpt-4o-mini", "gpt-5-mini", "gpt-4o", "gpt-5.1", "gpt-5.1-mini", "gpt-5-nano", "gpt-5.2"]
        sorted_models = sorted(models, key=model_sort_key)
        assert sorted_models == ["gpt-4o-mini", "gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-5.1-mini", "gpt-5.1", "gpt-5.2"]

    def test_llama_version_ordering(self):
        """Llama 3.1 and 3.2 and 3.3 should be ordered correctly."""
        models = ["Llama3.3-70B", "Llama3.1-70B", "Llama3.2-3B", "Llama3.1-8B"]
        sorted_models = sorted(models, key=model_sort_key)
        # Within Llama: first by version (3.1 < 3.2 < 3.3), then by size
        assert sorted_models == ["Llama3.1-8B", "Llama3.1-70B", "Llama3.2-3B", "Llama3.3-70B"]

    def test_mixed_series_comprehensive(self):
        """Comprehensive test with all model types."""
        models = [
            "gpt-5",
            "Mixtral-8x7B",
            "Qwen3-8B",
            "Phi-4",
            "Llama3.1-8B",
            "gpt-4o-mini",
            "Qwen2.5-7B",
        ]
        sorted_models = sorted(models, key=model_sort_key)
        # Expected order: Llama, Mixtral, Phi, Qwen (2.5 then 3), then GPT
        assert sorted_models == [
            "Llama3.1-8B",
            "Mixtral-8x7B",
            "Phi-4",
            "Qwen2.5-7B",
            "Qwen3-8B",
            "gpt-4o-mini",
            "gpt-5",
        ]

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert sorted([], key=model_sort_key) == []

    def test_single_model(self):
        """Single model should return list with that model."""
        assert sorted(["Qwen3-8B"], key=model_sort_key) == ["Qwen3-8B"]

    def test_model_sort_key_returns_tuple(self):
        """model_sort_key should return a comparable tuple."""
        key1 = model_sort_key("Llama3.1-8B")
        key2 = model_sort_key("Qwen3-8B")
        assert isinstance(key1, tuple)
        assert isinstance(key2, tuple)
        assert key1 < key2  # Llama before Qwen

    def test_unknown_models_sorted_alphabetically(self):
        """Unknown models should be sorted alphabetically after known series but before GPT."""
        models = ["gpt-4o", "UnknownModel-10B", "Llama3.1-8B", "AnotherUnknown"]
        sorted_models = sorted(models, key=model_sort_key)
        # Llama first, then unknown models alphabetically, then GPT
        assert sorted_models[0] == "Llama3.1-8B"
        assert sorted_models[-1] == "gpt-4o"
        # Unknown models in the middle, alphabetically
        assert sorted_models[1] == "AnotherUnknown"
        assert sorted_models[2] == "UnknownModel-10B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
