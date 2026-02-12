"""Tests for merge_lora.py and related utilities."""

import json
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.model_name_utils import extract_adapter_dir_name
from utils.train_utils import load_adapter_config


class TestExtractAdapterDirName:
    """Tests for extract_adapter_dir_name()."""

    def test_dpo_output_path(self):
        """Test extraction from dpo_output path."""
        path = "dpo_output/Qwen2.5-7B_int4_most_gpt5m_beta-0.02/final"
        assert extract_adapter_dir_name(path) == "Qwen2.5-7B_int4_most_gpt5m_beta-0.02"

    def test_absolute_path(self):
        """Test extraction from absolute path."""
        path = "/nas/ucb/bplaut/safety-persists-llm-agents/dpo_output/Qwen-8B_int4_most/final"
        assert extract_adapter_dir_name(path) == "Qwen-8B_int4_most"

    def test_trailing_slash(self):
        """Test that trailing slash is handled."""
        path = "dpo_output/Qwen-8B_int4_most/final/"
        assert extract_adapter_dir_name(path) == "Qwen-8B_int4_most"

    def test_multi_stage_name(self):
        """Test extraction of multi-stage finetuned model name."""
        path = "dpo_output/Qwen2.5-7B_int4_most_gpt5m_beta-0.02_int4_safe_diff2_gpt5m_beta-0.05/final"
        assert extract_adapter_dir_name(path) == "Qwen2.5-7B_int4_most_gpt5m_beta-0.02_int4_safe_diff2_gpt5m_beta-0.05"

    def test_missing_final_raises(self):
        """Test that path not ending in /final raises ValueError."""
        with pytest.raises(ValueError, match="must end with '/final'"):
            extract_adapter_dir_name("dpo_output/Qwen-8B_int4_most")

    def test_checkpoint_path_raises(self):
        """Test that checkpoint paths raise ValueError."""
        with pytest.raises(ValueError, match="must end with '/final'"):
            extract_adapter_dir_name("dpo_output/Qwen-8B_int4_most/checkpoints/checkpoint-100")

    def test_empty_path_raises(self):
        """Test that empty-ish paths raise ValueError."""
        with pytest.raises(ValueError):
            extract_adapter_dir_name("/final")


class TestLoadAdapterConfig:
    """Tests for load_adapter_config()."""

    def test_valid_config(self, tmp_path):
        """Test loading valid adapter config."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        config = {
            "base_model_name_or_path": "Qwen/Qwen3-8B",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj"]
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        result = load_adapter_config(str(adapter_dir))
        assert result["base_model_name_or_path"] == "Qwen/Qwen3-8B"
        assert result["r"] == 16

    def test_missing_config_raises(self, tmp_path):
        """Test that missing config raises FileNotFoundError."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="adapter_config.json not found"):
            load_adapter_config(str(adapter_dir))

    def test_invalid_json_raises(self, tmp_path):
        """Test that invalid JSON raises ValueError."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("not valid json {")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_adapter_config(str(adapter_dir))

    def test_missing_source_model_field_raises(self, tmp_path):
        """Test that missing source_model_name_or_path (or legacy base_model_name_or_path) raises KeyError."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        config = {"r": 16, "lora_alpha": 32}  # Missing source_model_name_or_path
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        with pytest.raises(KeyError, match="source_model_name_or_path"):
            load_adapter_config(str(adapter_dir))


class TestInferOutputDir:
    """Tests for infer_output_dir() in merge_lora.py."""

    def test_dpo_output_to_trained(self):
        """Test dpo_output -> dpo_trained conversion."""
        from merge_lora import infer_output_dir

        path = "dpo_output/Qwen2.5-7B_int4_most/final"
        assert infer_output_dir(path) == "dpo_trained/Qwen2.5-7B_int4_most"

    def test_unknown_prefix_defaults_to_dpo(self):
        """Test that unknown prefix defaults to dpo_trained."""
        from merge_lora import infer_output_dir

        path = "custom_output/Qwen-8B_int4_most/final"
        assert infer_output_dir(path) == "dpo_trained/Qwen-8B_int4_most"


class TestMergeLoraValidations:
    """Tests for validation logic in merge_lora_to_source()."""

    def test_non_final_path_raises(self):
        """Test that non-/final paths raise ValueError."""
        from merge_lora import merge_lora_to_source

        with pytest.raises(ValueError, match="must end with '/final'"):
            merge_lora_to_source(
                adapter_path="dpo_output/Qwen-8B_int4_most",
                output_dir="dpo_trained/test"
            )

    def test_nonexistent_adapter_raises(self, tmp_path):
        """Test that nonexistent adapter path raises FileNotFoundError."""
        from merge_lora import merge_lora_to_source

        with pytest.raises(FileNotFoundError, match="does not exist"):
            merge_lora_to_source(
                adapter_path=str(tmp_path / "nonexistent/final"),
                output_dir=str(tmp_path / "output")
            )

    def test_existing_model_config_raises(self, tmp_path):
        """Test that existing config.json raises FileExistsError."""
        from merge_lora import merge_lora_to_source

        # Create adapter dir with config
        adapter_dir = tmp_path / "adapter" / "final"
        adapter_dir.mkdir(parents=True)
        config = {"base_model_name_or_path": "Qwen/Qwen3-8B"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        # Create output dir with config.json (indicates model already saved)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "config.json").write_text("{}")

        with pytest.raises(FileExistsError, match="already exists"):
            merge_lora_to_source(
                adapter_path=str(adapter_dir),
                output_dir=str(output_dir)
            )

    def test_existing_model_weights_raises(self, tmp_path):
        """Test that existing model weights raise FileExistsError."""
        from merge_lora import merge_lora_to_source

        # Create adapter dir with config
        adapter_dir = tmp_path / "adapter" / "final"
        adapter_dir.mkdir(parents=True)
        config = {"base_model_name_or_path": "Qwen/Qwen3-8B"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        # Create output dir with model weights
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "model.safetensors").write_text("")

        with pytest.raises(FileExistsError, match="Model weights already exist"):
            merge_lora_to_source(
                adapter_path=str(adapter_dir),
                output_dir=str(output_dir)
            )

    def test_existing_merge_config_raises(self, tmp_path):
        """Test that existing merge_config.json raises FileExistsError."""
        from merge_lora import merge_lora_to_source

        # Create adapter dir with config
        adapter_dir = tmp_path / "adapter" / "final"
        adapter_dir.mkdir(parents=True)
        config = {"base_model_name_or_path": "Qwen/Qwen3-8B"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        # Create output dir with merge_config.json
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "merge_config.json").write_text("{}")

        with pytest.raises(FileExistsError, match="already exists"):
            merge_lora_to_source(
                adapter_path=str(adapter_dir),
                output_dir=str(output_dir)
            )
