import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add src/analysis to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "analysis"))

from analyze_token_limits import analyze_token_limits


@pytest.fixture
def temp_trajectory_dir():
    """Create temp directory with test trajectory files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_dir = Path(tmpdir) / "trajectories"

        # Model 1: 2/3 hit token limit
        model1_dir = traj_dir / "Qwen_Qwen3-8B"
        model1_dir.mkdir(parents=True)
        with open(model1_dir / "Qwen_Qwen3-8B_emu-x_eval-y_r0-3_0101_000000.jsonl", "w") as f:
            f.write(json.dumps({"output": "Agent stopped: token limit exceeded", "case_idx": 0}) + "\n")
            f.write(json.dumps({"output": "Task completed successfully", "case_idx": 1}) + "\n")
            f.write(json.dumps({"output": "Agent stopped: token limit exceeded", "case_idx": 2}) + "\n")

        # Model 2: 0/2 hit token limit
        model2_dir = traj_dir / "meta-llama_Llama-3.1-8B-Instruct"
        model2_dir.mkdir(parents=True)
        with open(model2_dir / "meta-llama_Llama-3.1-8B-Instruct_emu-x_eval-y_r0-2_0101_000000.jsonl", "w") as f:
            f.write(json.dumps({"output": "Done", "case_idx": 0}) + "\n")
            f.write(json.dumps({"output": "Completed", "case_idx": 1}) + "\n")

        yield traj_dir


def test_analyze_token_limits_counts(temp_trajectory_dir):
    """Test that token limit hits are counted correctly."""
    stats = analyze_token_limits(temp_trajectory_dir)

    assert stats["Qwen_Qwen3-8B"]["total"] == 3
    assert stats["Qwen_Qwen3-8B"]["token_limit_hit"] == 2
    assert stats["meta-llama_Llama-3.1-8B-Instruct"]["total"] == 2
    assert stats["meta-llama_Llama-3.1-8B-Instruct"]["token_limit_hit"] == 0


def test_analyze_token_limits_skips_eval_files(temp_trajectory_dir):
    """Test that evaluation files are skipped."""
    # Add an eval file that should be ignored
    eval_file = temp_trajectory_dir / "Qwen_Qwen3-8B" / "Qwen_Qwen3-8B_emu-x_eval-y_r0-3_0101_000000_eval_agent_safe.jsonl"
    with open(eval_file, "w") as f:
        f.write(json.dumps({"output": "Agent stopped: token limit exceeded"}) + "\n")

    stats = analyze_token_limits(temp_trajectory_dir)

    # Should still be 3, not 4
    assert stats["Qwen_Qwen3-8B"]["total"] == 3


def test_analyze_token_limits_empty_dir():
    """Test with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = analyze_token_limits(Path(tmpdir))
        assert stats == {}
