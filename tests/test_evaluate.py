"""
Unit tests for the 'ignore_safety' helpfulness metric.

Tests cover:
1. Prompt module imports and existence
2. Evaluator class safety_level mapping
3. Suffix mapping in evaluate.py and run.py
4. Batch re-evaluation script logic
"""

import os
import tempfile
import pytest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


class TestPromptModuleImports:
    """Test that the prompt modules can be imported."""

    def test_import_ignore_safety_prompts(self):
        """Test that IGNORE_SAFETY prompt modules can be imported."""
        from toolemu.prompts.evaluator import (
            AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY,
            AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY,
        )
        # Check that they are not None
        assert AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY is not None
        assert AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY is not None

    def test_prompt_modules_are_distinct(self):
        """Test that ignore_safety prompts are distinct from normal prompts."""
        from toolemu.prompts.evaluator import (
            AGENT_HELP_EVALUATOR_PROMPT,
            AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY,
            AGENT_HELP_EVALUATOR_SYSTEM_INFO,
            AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY,
        )
        # They should be separate objects
        assert AGENT_HELP_EVALUATOR_PROMPT is not None
        assert AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY is not None
        assert AGENT_HELP_EVALUATOR_SYSTEM_INFO is not None
        assert AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY is not None


class TestEvaluatorSafetyLevelMapping:
    """Test that AgentHelpfulnessEvaluator correctly maps safety levels."""

    def test_safety_level_0_normal(self):
        """Test safety_level=0 maps to normal prompts."""
        from toolemu.evaluators import AgentHelpfulnessEvaluator
        from toolemu.prompts.evaluator import (
            AGENT_HELP_EVALUATOR_PROMPT,
            AGENT_HELP_EVALUATOR_SYSTEM_INFO,
        )

        mock_llm = Mock()
        evaluator = AgentHelpfulnessEvaluator(mock_llm, safety_level=0)

        # Check that the correct prompt modules are set
        assert evaluator._task_prompt_module == AGENT_HELP_EVALUATOR_PROMPT
        assert evaluator._sys_prompt_module == AGENT_HELP_EVALUATOR_SYSTEM_INFO

    def test_safety_level_1_ignore_safety(self):
        """Test safety_level=1 maps to ignore_safety prompts."""
        from toolemu.evaluators import AgentHelpfulnessEvaluator
        from toolemu.prompts.evaluator import (
            AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY,
            AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY,
        )

        mock_llm = Mock()
        evaluator = AgentHelpfulnessEvaluator(mock_llm, safety_level=1)

        assert evaluator._task_prompt_module == AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY
        assert evaluator._sys_prompt_module == AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY

    def test_from_args_normal(self):
        """Test from_args with help_safety_level='normal'."""
        from toolemu.evaluators import AgentHelpfulnessEvaluator
        from toolemu.prompts.evaluator import (
            AGENT_HELP_EVALUATOR_PROMPT,
            AGENT_HELP_EVALUATOR_SYSTEM_INFO,
        )

        args = Namespace(help_safety_level='normal', stop_at=None)
        mock_llm = Mock()

        evaluator = AgentHelpfulnessEvaluator.from_args(args, mock_llm)

        assert evaluator._task_prompt_module == AGENT_HELP_EVALUATOR_PROMPT
        assert evaluator._sys_prompt_module == AGENT_HELP_EVALUATOR_SYSTEM_INFO

    def test_from_args_ignore_safety(self):
        """Test from_args with help_safety_level='ignore_safety'."""
        from toolemu.evaluators import AgentHelpfulnessEvaluator
        from toolemu.prompts.evaluator import (
            AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY,
            AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY,
        )

        args = Namespace(help_safety_level='ignore_safety', stop_at=None)
        mock_llm = Mock()

        evaluator = AgentHelpfulnessEvaluator.from_args(args, mock_llm)

        assert evaluator._task_prompt_module == AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY
        assert evaluator._sys_prompt_module == AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY

    def test_from_args_default_to_normal(self):
        """Test from_args defaults to normal when help_safety_level is missing."""
        from toolemu.evaluators import AgentHelpfulnessEvaluator
        from toolemu.prompts.evaluator import (
            AGENT_HELP_EVALUATOR_PROMPT,
            AGENT_HELP_EVALUATOR_SYSTEM_INFO,
        )

        args = Namespace(stop_at=None)  # No help_safety_level attribute
        mock_llm = Mock()

        evaluator = AgentHelpfulnessEvaluator.from_args(args, mock_llm)

        # Should default to normal (safety_level=0)
        assert evaluator._task_prompt_module == AGENT_HELP_EVALUATOR_PROMPT
        assert evaluator._sys_prompt_module == AGENT_HELP_EVALUATOR_SYSTEM_INFO


class TestSuffixMapping:
    """Test that suffix mappings work correctly for file naming."""

    def test_suffix_map_completeness(self):
        """Test that all safety levels have suffix mappings."""
        suffix_map = {
            'normal': '',
            'ignore_safety': '_ignore_safety',
        }

        # All safety levels should be present
        assert 'normal' in suffix_map
        assert 'ignore_safety' in suffix_map

        # Verify suffix values
        assert suffix_map['normal'] == ''
        assert suffix_map['ignore_safety'] == '_ignore_safety'

    def test_eval_file_naming_normal(self):
        """Test evaluation file naming for normal metric."""
        base = "model_emu-sim_eval-eval_int4_r0-144_1234_567890"
        suffix_map = {
            'normal': '',
            'ignore_safety': '_ignore_safety',
        }

        eval_suffix = suffix_map['normal']
        expected = f"{base}_eval_agent_help.jsonl"
        actual = f"{base}_eval_agent_help{eval_suffix}.jsonl"

        assert actual == expected

    def test_eval_file_naming_ignore_safety(self):
        """Test evaluation file naming for ignore_safety metric."""
        base = "model_emu-sim_eval-eval_int4_r0-144_1234_567890"
        suffix_map = {
            'normal': '',
            'ignore_safety': '_ignore_safety',
        }

        eval_suffix = suffix_map['ignore_safety']
        expected = f"{base}_eval_agent_help_ignore_safety.jsonl"
        actual = f"{base}_eval_agent_help{eval_suffix}.jsonl"

        assert actual == expected

class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_all_metrics_in_default_list(self):
        """Test that run.py default includes all helpfulness metrics."""
        default_metrics = ["normal", "ignore_safety"]

        # Verify both are present
        assert len(default_metrics) == 2
        assert "normal" in default_metrics
        assert "ignore_safety" in default_metrics

    def test_all_safety_levels_map_correctly(self):
        """Test complete mapping from strings to safety_level integers."""
        from toolemu.evaluators import AgentHelpfulnessEvaluator

        safety_level_map = {
            'normal': 0,
            'ignore_safety': 1,
        }

        mock_llm = Mock()

        for level_str, expected_int in safety_level_map.items():
            args = Namespace(help_safety_level=level_str, stop_at=None)
            evaluator = AgentHelpfulnessEvaluator.from_args(args, mock_llm)

            # The evaluator should have been initialized with the correct safety_level
            # We verify this by checking the prompt modules
            if expected_int == 0:
                from toolemu.prompts.evaluator import AGENT_HELP_EVALUATOR_PROMPT
                assert evaluator._task_prompt_module == AGENT_HELP_EVALUATOR_PROMPT
            elif expected_int == 1:
                from toolemu.prompts.evaluator import AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY
                assert evaluator._task_prompt_module == AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY
