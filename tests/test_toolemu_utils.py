#!/usr/bin/env python3
"""
Unit tests for src/utils/toolemu_utils.py

Tests cover:
- ToolEmuFilePaths class (path conversions, constants, file type checking)
- extract_score() function (score extraction with various scenarios)
- extract_case_range_from_filename() function
- load_jsonl() function
- build_toolemu_filename() function
- parse_toolemu_filename() function
- load_and_validate_all_eval_files() function
- validate_trajectory_eval_alignment() function
- load_and_validate_trajectory_eval_pair() function
- validate_no_overlapping_files() function
- find_trajectory_files() function
- compute_eval_consistency() function
- check_trajectory_exists() function
"""

import json
import os
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.toolemu_utils import (
    IncompleteTrajectoryError,
    ToolEmuFilePaths,
    extract_score,
    load_and_validate_all_eval_files,
    extract_case_range_from_filename,
    validate_trajectory_eval_alignment,
    validate_trajectory_completeness,
    validate_no_overlapping_files,
    load_and_validate_trajectory_eval_pair,
    load_jsonl,
    build_toolemu_filename,
    parse_toolemu_filename,
    find_trajectory_files,
    compute_eval_consistency,
    check_trajectory_exists,
)
from utils.train_utils import compute_test_indices

# Import shared test helpers from conftest
from conftest import write_jsonl, create_trajectory_data, create_eval_data


class TestToolEmuFilePaths:
    """Tests for ToolEmuFilePaths class."""

    def test_constants(self):
        """Test that all expected constants are defined."""
        assert len(ToolEmuFilePaths.EVAL_TYPES) == 3
        assert 'agent_safe' in ToolEmuFilePaths.EVAL_TYPES
        assert 'agent_help' in ToolEmuFilePaths.EVAL_TYPES
        assert 'agent_help_ignore_safety' in ToolEmuFilePaths.EVAL_TYPES

        assert len(ToolEmuFilePaths.METRIC_TO_EVAL_TYPE) == 3
        assert ToolEmuFilePaths.METRIC_TO_EVAL_TYPE['safe'] == 'agent_safe'
        assert ToolEmuFilePaths.METRIC_TO_EVAL_TYPE['help'] == 'agent_help'
        assert ToolEmuFilePaths.METRIC_TO_EVAL_TYPE['help_ignore_safety'] == 'agent_help_ignore_safety'

        assert len(ToolEmuFilePaths.SKIP_SUFFIXES) == 4
        assert '_eval_agent_safe.jsonl' in ToolEmuFilePaths.SKIP_SUFFIXES
        assert '_eval_agent_help_ignore_safety.jsonl' in ToolEmuFilePaths.SKIP_SUFFIXES
        assert '_unified_report.json' in ToolEmuFilePaths.SKIP_SUFFIXES

    def test_trajectory_to_eval(self):
        """Test conversion from trajectory path to eval path."""
        traj_path = "/path/to/model_r0-10_timestamp.jsonl"

        # Test each eval type
        expected = {
            'agent_safe': "/path/to/model_r0-10_timestamp_eval_agent_safe.jsonl",
            'agent_help': "/path/to/model_r0-10_timestamp_eval_agent_help.jsonl",
            'agent_help_ignore_safety': "/path/to/model_r0-10_timestamp_eval_agent_help_ignore_safety.jsonl",
        }

        for eval_type, expected_path in expected.items():
            result = ToolEmuFilePaths.trajectory_to_eval(traj_path, eval_type)
            assert result == expected_path

    def test_trajectory_to_eval_invalid_path(self):
        """Test that non-.jsonl paths raise ValueError."""
        with pytest.raises(ValueError, match="must end with .jsonl"):
            ToolEmuFilePaths.trajectory_to_eval("/path/to/file.txt", "agent_safe")

    def test_eval_to_trajectory(self):
        """Test conversion from eval path to trajectory path."""
        test_cases = [
            ("/path/to/model_r0-10_timestamp_eval_agent_safe.jsonl",
             "/path/to/model_r0-10_timestamp.jsonl"),
            ("/path/to/model_r0-10_timestamp_eval_agent_help.jsonl",
             "/path/to/model_r0-10_timestamp.jsonl"),
            ("/path/to/model_r0-10_timestamp_eval_agent_help_ignore_safety.jsonl",
             "/path/to/model_r0-10_timestamp.jsonl"),
        ]

        for eval_path, expected_traj_path in test_cases:
            result = ToolEmuFilePaths.eval_to_trajectory(eval_path)
            assert result == expected_traj_path

    def test_extract_eval_type(self):
        """Test extraction of eval type from eval file path."""
        test_cases = [
            ("/path/to/model_eval_agent_safe.jsonl", "agent_safe"),
            ("/path/to/model_eval_agent_help.jsonl", "agent_help"),
            ("/path/to/model_eval_agent_help_ignore_safety.jsonl", "agent_help_ignore_safety"),
            ("/path/to/model.jsonl", None),  # Not an eval file
        ]

        for eval_path, expected_type in test_cases:
            result = ToolEmuFilePaths.extract_eval_type(eval_path)
            assert result == expected_type

    def test_is_trajectory_file(self):
        """Test detection of trajectory files."""
        # Should be trajectory files
        assert ToolEmuFilePaths.is_trajectory_file("model_r0-10_timestamp.jsonl") == True
        assert ToolEmuFilePaths.is_trajectory_file("/path/to/model.jsonl") == True

        # Should NOT be trajectory files (eval files)
        assert ToolEmuFilePaths.is_trajectory_file("model_eval_agent_safe.jsonl") == False
        assert ToolEmuFilePaths.is_trajectory_file("model_eval_agent_help.jsonl") == False
        assert ToolEmuFilePaths.is_trajectory_file("model_eval_agent_help_ignore_safety.jsonl") == False

        # Should NOT be trajectory files (other suffixes)
        assert ToolEmuFilePaths.is_trajectory_file("model_unified_report.json") == False

        # Should NOT be trajectory files (wrong extension)
        assert ToolEmuFilePaths.is_trajectory_file("model.txt") == False

    def test_get_eval_suffix(self):
        """Test getting eval file suffix from metric name."""
        test_cases = [
            ('safe', '_eval_agent_safe.jsonl'),
            ('help', '_eval_agent_help.jsonl'),
            ('help_ignore_safety', '_eval_agent_help_ignore_safety.jsonl'),
        ]

        for metric, expected_suffix in test_cases:
            result = ToolEmuFilePaths.get_eval_suffix(metric)
            assert result == expected_suffix

    def test_get_eval_suffix_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            ToolEmuFilePaths.get_eval_suffix("invalid_metric")


class TestExtractScore:
    """Tests for extract_score function."""

    def test_extract_safety_score(self):
        """Test extracting safety (ToolCallRisk) score."""
        eval_data = {
            'eval_scores': {
                'ToolCallRisk': 3.0
            }
        }

        score = extract_score(eval_data, 'agent_safe', case_idx=0)
        assert score == 3.0

    def test_extract_helpfulness_score(self):
        """Test extracting helpfulness score from all help variants."""
        eval_data = {
            'eval_scores': {
                'Helpfulness': 2.0
            }
        }

        for eval_type in ['agent_help', 'agent_help_ignore_safety']:
            score = extract_score(eval_data, eval_type, case_idx=0)
            assert score == 2.0

    def test_extract_score_missing_not_allowed(self):
        """Test that missing score raises error when allow_missing=False."""
        eval_data = {'eval_scores': {}}

        with pytest.raises(ValueError, match="Missing ToolCallRisk score"):
            extract_score(eval_data, 'agent_safe', case_idx=0, allow_missing=False)

        with pytest.raises(ValueError, match="Missing Helpfulness score"):
            extract_score(eval_data, 'agent_help', case_idx=0, allow_missing=False)

    def test_extract_score_missing_allowed(self):
        """Test that missing score returns None when allow_missing=True."""
        eval_data = {'eval_scores': {}}

        score = extract_score(eval_data, 'agent_safe', case_idx=0, allow_missing=True)
        assert score is None

        score = extract_score(eval_data, 'agent_help', case_idx=0, allow_missing=True)
        assert score is None

    def test_extract_score_invalid_eval_type(self):
        """Test that invalid eval_type raises ValueError."""
        eval_data = {'eval_scores': {'ToolCallRisk': 3.0}}

        with pytest.raises(ValueError, match="Unknown eval_type"):
            extract_score(eval_data, 'invalid_type', case_idx=0)

    def test_extract_score_multi_replicate_returns_median(self):
        """Test that multi-replicate format returns median, not mean."""
        # Scores: [1, 1, 3] -> median=1, mean=1.67
        eval_data = {
            'replicates': [
                {'eval_scores': {'Helpfulness': 1}},
                {'eval_scores': {'Helpfulness': 1}},
                {'eval_scores': {'Helpfulness': 3}},
            ],
            'consistency': {'mean': 1.67}  # Mean would be ~1.67, but we want median=1
        }
        score = extract_score(eval_data, 'agent_help', case_idx=0)
        assert score == 1.0  # Median of [1, 1, 3] is 1, not mean of 1.67

    def test_extract_score_multi_replicate_even_count(self):
        """Test median with even number of replicates."""
        # Scores: [1, 2, 2, 3] -> median=2.0 (average of middle two)
        eval_data = {
            'replicates': [
                {'eval_scores': {'ToolCallRisk': 1}},
                {'eval_scores': {'ToolCallRisk': 2}},
                {'eval_scores': {'ToolCallRisk': 2}},
                {'eval_scores': {'ToolCallRisk': 3}},
            ]
        }
        score = extract_score(eval_data, 'agent_safe', case_idx=0)
        assert score == 2.0

    def test_extract_score_multi_replicate_skips_none_scores(self):
        """Test that None scores in replicates are skipped."""
        eval_data = {
            'replicates': [
                {'eval_scores': {'Helpfulness': 1}},
                {'eval_scores': {'Helpfulness': None}},
                {'eval_scores': {'Helpfulness': 3}},
            ]
        }
        score = extract_score(eval_data, 'agent_help', case_idx=0)
        assert score == 2.0  # Median of [1, 3] is 2.0

    def test_extract_score_multi_replicate_missing_eval_scores(self):
        """Test that missing eval_scores in replicates are skipped."""
        eval_data = {
            'replicates': [
                {'eval_scores': {'Helpfulness': 1}},
                {},  # Missing eval_scores entirely
                {'eval_scores': {'Helpfulness': 3}},
            ]
        }
        score = extract_score(eval_data, 'agent_help', case_idx=0)
        assert score == 2.0  # Median of [1, 3] is 2.0

    def test_extract_score_multi_replicate_all_none_raises(self):
        """Test that all-None replicates raises error when allow_missing=False."""
        eval_data = {
            'replicates': [
                {'eval_scores': {'Helpfulness': None}},
                {'eval_scores': {'Helpfulness': None}},
            ]
        }
        with pytest.raises(ValueError, match="No valid replicate scores"):
            extract_score(eval_data, 'agent_help', case_idx=0, allow_missing=False)

    def test_extract_score_multi_replicate_all_none_allowed(self):
        """Test that all-None replicates returns None when allow_missing=True."""
        eval_data = {
            'replicates': [
                {'eval_scores': {'Helpfulness': None}},
                {'eval_scores': {'Helpfulness': None}},
            ]
        }
        score = extract_score(eval_data, 'agent_help', case_idx=0, allow_missing=True)
        assert score is None

    def test_extract_score_multi_replicate_mean(self):
        """Test mean aggregation for multi-replicate format."""
        eval_data = {
            'replicates': [
                {'eval_scores': {'Helpfulness': 1}},
                {'eval_scores': {'Helpfulness': 2}},
                {'eval_scores': {'Helpfulness': 3}},
            ]
        }
        # Median of [1, 2, 3] is 2.0
        score_median = extract_score(eval_data, 'agent_help', case_idx=0, aggregation="median")
        assert score_median == 2.0

        # Mean of [1, 2, 3] is 2.0
        score_mean = extract_score(eval_data, 'agent_help', case_idx=0, aggregation="mean")
        assert score_mean == 2.0

    def test_extract_score_multi_replicate_mean_vs_median_different(self):
        """Test that mean and median give different results when appropriate."""
        eval_data = {
            'replicates': [
                {'eval_scores': {'ToolCallRisk': 0}},
                {'eval_scores': {'ToolCallRisk': 0}},
                {'eval_scores': {'ToolCallRisk': 3}},
            ]
        }
        # Median of [0, 0, 3] is 0
        score_median = extract_score(eval_data, 'agent_safe', case_idx=0, aggregation="median")
        assert score_median == 0.0

        # Mean of [0, 0, 3] is 1.0
        score_mean = extract_score(eval_data, 'agent_safe', case_idx=0, aggregation="mean")
        assert score_mean == 1.0

    def test_extract_score_invalid_aggregation(self):
        """Test that invalid aggregation parameter raises error."""
        eval_data = {
            'replicates': [
                {'eval_scores': {'Helpfulness': 1}},
                {'eval_scores': {'Helpfulness': 2}},
            ]
        }
        with pytest.raises(ValueError, match="aggregation must be 'median' or 'mean'"):
            extract_score(eval_data, 'agent_help', case_idx=0, aggregation="invalid")

    def test_extract_score_aggregation_single_replicate_ignored(self):
        """Test that aggregation parameter is ignored for single-replicate format."""
        eval_data = {'eval_scores': {'Helpfulness': 2}}
        # For single-replicate, aggregation should not matter
        score_median = extract_score(eval_data, 'agent_help', case_idx=0, aggregation="median")
        score_mean = extract_score(eval_data, 'agent_help', case_idx=0, aggregation="mean")
        assert score_median == 2
        assert score_mean == 2


class TestExtractCaseRange:
    """Tests for extract_case_range_from_filename function."""

    def test_extract_basic_range(self):
        """Test extracting range from typical filename."""
        filename = "model_r30-45_timestamp.jsonl"
        start, end = extract_case_range_from_filename(filename)
        assert start == 30
        assert end == 45

    def test_extract_range_with_path(self):
        """Test extracting range from full path."""
        filepath = "/path/to/output/model_r0-143_timestamp.jsonl"
        start, end = extract_case_range_from_filename(filepath)
        assert start == 0
        assert end == 143

    def test_extract_range_no_match(self):
        """Test that filename without range returns None."""
        filename = "model_timestamp.jsonl"
        result = extract_case_range_from_filename(filename)
        assert result is None

    def test_extract_range_eval_file(self):
        """Test extracting range from eval file."""
        filename = "model_r10-20_timestamp_eval_agent_safe.jsonl"
        start, end = extract_case_range_from_filename(filename)
        assert start == 10
        assert end == 20


class TestLoadJsonl:
    """Tests for load_jsonl function."""

    def test_load_valid_jsonl(self, temp_test_dir):
        """Test loading a valid JSONL file."""
        tmpdir = temp_test_dir
        test_data = [
            {'id': 1, 'name': 'Alice', 'score': 95.5},
            {'id': 2, 'name': 'Bob', 'score': 87.0},
            {'id': 3, 'name': 'Charlie', 'score': 92.3}
        ]

        jsonl_file = tmpdir / "test_valid.jsonl"
        write_jsonl(jsonl_file, test_data)

        result = load_jsonl(str(jsonl_file))
        assert result == test_data
        assert len(result) == 3

    def test_load_empty_file(self, temp_test_dir):
        """Test loading an empty JSONL file."""
        tmpdir = temp_test_dir
        jsonl_file = tmpdir / "test_empty.jsonl"

        # Create empty file
        jsonl_file.touch()

        result = load_jsonl(str(jsonl_file))
        assert result == []

    def test_load_with_empty_lines(self, temp_test_dir):
        """Test loading JSONL file with empty lines (should skip them)."""
        tmpdir = temp_test_dir
        jsonl_file = tmpdir / "test_with_blanks.jsonl"

        # Write file with empty lines
        with open(jsonl_file, 'w') as f:
            f.write('{"id": 1, "value": "first"}\n')
            f.write('\n')  # Empty line
            f.write('{"id": 2, "value": "second"}\n')
            f.write('   \n')  # Whitespace line
            f.write('{"id": 3, "value": "third"}\n')

        result = load_jsonl(str(jsonl_file))
        assert len(result) == 3
        assert result[0]['id'] == 1
        assert result[1]['id'] == 2
        assert result[2]['id'] == 3

    def test_load_nonexistent_file(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/path/file.jsonl")

    def test_load_invalid_json(self, temp_test_dir):
        """Test that malformed JSON raises ValueError with line number."""
        tmpdir = temp_test_dir
        jsonl_file = tmpdir / "test_invalid.jsonl"

        # Write file with invalid JSON on line 2
        with open(jsonl_file, 'w') as f:
            f.write('{"id": 1, "value": "valid"}\n')
            f.write('{"id": 2, "value": invalid}\n')  # Missing quotes
            f.write('{"id": 3, "value": "also valid"}\n')

        with pytest.raises(ValueError, match="line 2"):
            load_jsonl(str(jsonl_file))

    def test_load_invalid_json_with_description(self, temp_test_dir):
        """Test that description appears in error message."""
        tmpdir = temp_test_dir
        jsonl_file = tmpdir / "test_invalid.jsonl"

        with open(jsonl_file, 'w') as f:
            f.write('{invalid json}\n')

        with pytest.raises(ValueError, match="trajectory file"):
            load_jsonl(str(jsonl_file), description="trajectory file")

    def test_load_unicode_content(self, temp_test_dir):
        """Test loading JSONL with unicode characters."""
        tmpdir = temp_test_dir
        test_data = [
            {'text': 'Hello 世界', 'emoji': '🎉'},
            {'text': 'Привет мир', 'symbol': '→'},
            {'text': 'Café ñoño', 'value': 42}
        ]

        jsonl_file = tmpdir / "test_unicode.jsonl"
        write_jsonl(jsonl_file, test_data)

        result = load_jsonl(str(jsonl_file))
        assert result == test_data
        assert result[0]['text'] == 'Hello 世界'
        assert result[0]['emoji'] == '🎉'

    def test_load_nested_structures(self, temp_test_dir):
        """Test loading JSONL with complex nested structures."""
        tmpdir = temp_test_dir
        test_data = [
            {
                'case_idx': 0,
                'case': {
                    'Toolkits': ['GmailToolkit', 'FileSystem'],
                    'User Instruction': 'Send an email'
                },
                'intermediate_steps': [
                    ['action1', 'observation1'],
                    ['action2', 'observation2']
                ],
                'metadata': {
                    'model': 'gpt-4',
                    'scores': [1, 2, 3]
                }
            }
        ]

        jsonl_file = tmpdir / "test_nested.jsonl"
        write_jsonl(jsonl_file, test_data)

        result = load_jsonl(str(jsonl_file))
        assert result == test_data
        assert len(result[0]['intermediate_steps']) == 2
        # Note: JSON doesn't preserve tuple vs list - tuples become lists

    def test_load_large_file(self, temp_test_dir):
        """Test loading a large JSONL file (performance check)."""
        tmpdir = temp_test_dir
        # Create 1000 records
        test_data = [
            {'id': i, 'value': f'item_{i}', 'score': i * 1.5}
            for i in range(1000)
        ]

        jsonl_file = tmpdir / "test_large.jsonl"
        write_jsonl(jsonl_file, test_data)

        result = load_jsonl(str(jsonl_file))
        assert len(result) == 1000
        assert result[0]['id'] == 0
        assert result[999]['id'] == 999

    def test_load_single_line(self, temp_test_dir):
        """Test loading JSONL with single line."""
        tmpdir = temp_test_dir
        test_data = [{'single': 'item'}]

        jsonl_file = tmpdir / "test_single.jsonl"
        write_jsonl(jsonl_file, test_data)

        result = load_jsonl(str(jsonl_file))
        assert len(result) == 1
        assert result[0] == {'single': 'item'}

    def test_load_malformed_json_middle(self, temp_test_dir):
        """Test error reporting for malformed JSON in middle of file."""
        tmpdir = temp_test_dir
        jsonl_file = tmpdir / "test_malformed_middle.jsonl"

        with open(jsonl_file, 'w') as f:
            f.write('{"line": 1}\n')
            f.write('{"line": 2}\n')
            f.write('{"line": 3}\n')
            f.write('{"line": "four" invalid}\n')  # Malformed
            f.write('{"line": 5}\n')

        with pytest.raises(ValueError) as exc_info:
            load_jsonl(str(jsonl_file))

        error_msg = str(exc_info.value)
        assert "line 4" in error_msg
        assert str(jsonl_file) in error_msg

    def test_load_returns_list_of_dicts(self, temp_test_dir):
        """Test that load_jsonl returns list of dictionaries."""
        tmpdir = temp_test_dir
        test_data = [{'a': 1}, {'b': 2}, {'c': 3}]

        jsonl_file = tmpdir / "test_types.jsonl"
        write_jsonl(jsonl_file, test_data)

        result = load_jsonl(str(jsonl_file))
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)


class TestBuildToolEmuFilename:
    """Tests for build_toolemu_filename function."""

    def test_basic_filename_none_omitted(self):
        """Test building basic filename - none is omitted since it's the default."""
        result = build_toolemu_filename(
            agent_model="Qwen/Qwen3-8B",
            emu_model="Qwen/Qwen3-32B",
            eval_model="Qwen/Qwen3-32B",
            quantization="none"
        )
        # none should NOT appear in the filename (it's the default)
        assert result == "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B"
        assert "_none" not in result

    def test_filename_with_range_none_omitted(self):
        """Test building filename with task range - none is omitted."""
        result = build_toolemu_filename(
            agent_model="gpt-4o-mini",
            emu_model="gpt-4o-mini",
            eval_model="gpt-4o-mini",
            quantization="none",
            case_range=(0, 15)
        )
        # none should NOT appear in the filename
        assert result == "gpt-4o-mini_emu-gpt-4o-mini_eval-gpt-4o-mini_r0-15"
        assert "_none" not in result

    def test_filename_with_timestamp(self):
        """Test building filename with timestamp."""
        result = build_toolemu_filename(
            agent_model="meta-llama/Llama-3.1-8B-Instruct",
            emu_model="Qwen/Qwen3-32B",
            eval_model="Qwen/Qwen3-32B",
            quantization="int8",
            timestamp="0511_193906"
        )
        assert result == "meta-llama_Llama-3.1-8B-Instruct_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int8_0511_193906"

    def test_filename_with_all_params_int4_included(self):
        """Test building filename with all parameters - int4 is included."""
        result = build_toolemu_filename(
            agent_model="Qwen/Qwen3-8B",
            emu_model="gpt-4o",
            eval_model="gpt-4o",
            quantization="int4",
            case_range=(30, 45),
            timestamp="1205_142030"
        )
        assert result == "Qwen_Qwen3-8B_emu-gpt-4o_eval-gpt-4o_int4_r30-45_1205_142030"

    def test_filename_with_suffix_int4_included(self):
        """Test building filename with custom suffix - int4 is included."""
        result = build_toolemu_filename(
            agent_model="gpt-4o-mini",
            emu_model="gpt-4o-mini",
            eval_model="gpt-4o-mini",
            quantization="int4",
            suffix="_eval_agent_safe"
        )
        assert result == "gpt-4o-mini_emu-gpt-4o-mini_eval-gpt-4o-mini_int4_eval_agent_safe"

    def test_filename_sanitization_int4_included(self):
        """Test that model names with slashes are sanitized - int4 is included."""
        result = build_toolemu_filename(
            agent_model="01-ai/Yi-1.5-9B-Chat-16K",
            emu_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            eval_model="meta-llama/Llama-3.1-70B-Instruct",
            quantization="int4"
        )
        expected = "01-ai_Yi-1.5-9B-Chat-16K_emu-mistralai_Mixtral-8x7B-Instruct-v0.1_eval-meta-llama_Llama-3.1-70B-Instruct_int4"
        assert result == expected

    def test_filename_no_quantization(self):
        """Test building filename with API models (no quantization)."""
        result = build_toolemu_filename(
            agent_model="gpt-4o",
            emu_model="gpt-4o",
            eval_model="gpt-4o",
            quantization=""
        )
        assert result == "gpt-4o_emu-gpt-4o_eval-gpt-4o"

    def test_filename_int8_included(self):
        """Test that int8 quantization IS included in filename."""
        result = build_toolemu_filename(
            agent_model="Qwen/Qwen3-8B",
            emu_model="Qwen/Qwen3-32B",
            eval_model="Qwen/Qwen3-32B",
            quantization="int8"
        )
        assert "_int8" in result
        assert result == "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int8"

    def test_filename_none_omitted(self):
        """Test that 'none' quantization is omitted from filename."""
        result = build_toolemu_filename(
            agent_model="Qwen/Qwen3-8B",
            emu_model="Qwen/Qwen3-32B",
            eval_model="Qwen/Qwen3-32B",
            quantization="none"
        )
        assert result == "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B"
        assert "_none" not in result

    def test_filename_with_lora_adapter_int4_included(self):
        """Test building filename with LoRA adapter path in agent model - int4 is included."""
        result = build_toolemu_filename(
            agent_model="Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_safe/final",
            emu_model="Qwen/Qwen3-32B",
            eval_model="Qwen/Qwen3-32B",
            quantization="int4"
        )
        assert "_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4" in result
        assert "/" not in result


class TestParseToolEmuFilename:
    """Tests for parse_toolemu_filename function."""

    def test_parse_basic_trajectory_file(self):
        """Test parsing basic trajectory filename with all components."""
        result = parse_toolemu_filename(
            "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-15_0511_193906.jsonl"
        )
        assert result['agent_model'] == "Qwen_Qwen3-8B"
        assert result['emu_model'] == "Qwen_Qwen3-32B"
        assert result['eval_model'] == "Qwen_Qwen3-32B"
        assert result['quantization'] == "int4"
        assert result['range_start'] == 0
        assert result['range_end'] == 15
        assert result['timestamp'] == "0511_193906"
        assert result['suffix'] == ""

    def test_parse_with_eval_suffix(self):
        """Test parsing evaluation file with eval suffix."""
        result = parse_toolemu_filename(
            "gpt-4o-mini_emu-gpt-4o-mini_eval-gpt-4o-mini_int4_r30-45_1205_142030_eval_agent_help.jsonl"
        )
        assert result['agent_model'] == "gpt-4o-mini"
        assert result['emu_model'] == "gpt-4o-mini"
        assert result['eval_model'] == "gpt-4o-mini"
        assert result['quantization'] == "int4"
        assert result['range_start'] == 30
        assert result['range_end'] == 45
        assert result['timestamp'] == "1205_142030"
        assert result['suffix'] == "_eval_agent_help"

    def test_parse_unified_report(self):
        """Test parsing unified report filename (no range or timestamp)."""
        result = parse_toolemu_filename(
            "Qwen_Qwen3-32B_emu-gpt-5_eval-gpt-5_int8_unified_report.json"
        )
        assert result['agent_model'] == "Qwen_Qwen3-32B"
        assert result['emu_model'] == "gpt-5"
        assert result['eval_model'] == "gpt-5"
        assert result['quantization'] == "int8"
        assert result['range_start'] is None
        assert result['range_end'] is None
        assert result['timestamp'] is None
        assert result['suffix'] == "_unified_report"

    def test_parse_without_extension(self):
        """Test parsing filename without extension."""
        result = parse_toolemu_filename(
            "meta-llama_Llama-3.1-8B-Instruct_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-144"
        )
        assert result['agent_model'] == "meta-llama_Llama-3.1-8B-Instruct"
        assert result['emu_model'] == "Qwen_Qwen3-32B"
        assert result['eval_model'] == "Qwen_Qwen3-32B"
        assert result['quantization'] == "int4"
        assert result['range_start'] == 0
        assert result['range_end'] == 144
        assert result['timestamp'] is None

    def test_parse_no_quantization_defaults_to_none(self):
        """Test parsing filename without quantization - defaults to none."""
        result = parse_toolemu_filename(
            "gpt-4o_emu-gpt-4o_eval-gpt-4o_r10-20_0101_120000"
        )
        assert result['agent_model'] == "gpt-4o"
        assert result['emu_model'] == "gpt-4o"
        assert result['eval_model'] == "gpt-4o"
        assert result['quantization'] == "none"
        assert result['range_start'] == 10
        assert result['range_end'] == 20
        assert result['timestamp'] == "0101_120000"

    def test_parse_no_range_no_timestamp_backwards_compat(self):
        """Test parsing old-style filename with int4 - backwards compatible."""
        result = parse_toolemu_filename(
            "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4"
        )
        assert result['agent_model'] == "Qwen_Qwen3-8B"
        assert result['emu_model'] == "Qwen_Qwen3-32B"
        assert result['eval_model'] == "Qwen_Qwen3-32B"
        # Old files with _int4 should still parse correctly
        assert result['quantization'] == "int4"
        assert result['range_start'] is None
        assert result['range_end'] is None
        assert result['timestamp'] is None

    def test_parse_new_style_no_quantization(self):
        """Test parsing new-style filename without quantization - defaults to none."""
        result = parse_toolemu_filename(
            "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B"
        )
        assert result['agent_model'] == "Qwen_Qwen3-8B"
        assert result['emu_model'] == "Qwen_Qwen3-32B"
        assert result['eval_model'] == "Qwen_Qwen3-32B"
        assert result['quantization'] == "none"
        assert result['range_start'] is None
        assert result['range_end'] is None
        assert result['timestamp'] is None

    def test_parse_complex_model_names(self):
        """Test parsing with complex model names containing underscores and hyphens."""
        result = parse_toolemu_filename(
            "01-ai_Yi-1.5-9B-Chat-16K_emu-mistralai_Mixtral-8x7B-Instruct-v0.1_eval-meta-llama_Llama-3.1-70B-Instruct_int4_r0-15_0511_120000.jsonl"
        )
        assert result['agent_model'] == "01-ai_Yi-1.5-9B-Chat-16K"
        assert result['emu_model'] == "mistralai_Mixtral-8x7B-Instruct-v0.1"
        assert result['eval_model'] == "meta-llama_Llama-3.1-70B-Instruct"
        assert result['quantization'] == "int4"
        assert result['range_start'] == 0
        assert result['range_end'] == 15

    def test_parse_dpo_model(self):
        """Test parsing filename with DPO adapter path in agent model."""
        result = parse_toolemu_filename(
            "Qwen-8B_int4_1A6000_safe_1600_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-15_0511_120000.jsonl"
        )
        assert result['agent_model'] == "Qwen-8B_int4_1A6000_safe_1600"
        assert result['emu_model'] == "Qwen_Qwen3-32B"
        assert result['eval_model'] == "Qwen_Qwen3-32B"

    def test_parse_with_split_suffix(self):
        """Test parsing filename with dataset split suffix in agent model."""
        result = parse_toolemu_filename(
            "Qwen_Qwen3-8B_really_split_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_int4_r0-15_0511_120000.jsonl"
        )
        assert result['agent_model'] == "Qwen_Qwen3-8B_really_split"
        assert result['emu_model'] == "Qwen_Qwen3-32B"
        assert result['eval_model'] == "Qwen_Qwen3-32B"

    def test_parse_missing_sim_marker(self):
        """Test parsing filename missing _emu- marker."""
        with pytest.raises(ValueError, match="missing required '_emu-' marker"):
            parse_toolemu_filename("model_name_eval-evaluator_int4")

    def test_parse_missing_eval_marker(self):
        """Test parsing filename missing _eval- marker."""
        with pytest.raises(ValueError, match="missing required '_eval-' marker"):
            parse_toolemu_filename("model_name_emu-simulator_int4")

    def test_parse_empty_filename(self):
        """Test parsing empty filename."""
        with pytest.raises(ValueError, match="empty"):
            parse_toolemu_filename("")

    def test_parse_malformed_range(self):
        """Test parsing filename with malformed range."""
        with pytest.raises(ValueError, match="range format"):
            parse_toolemu_filename("model_emu-sim_eval-eval_int4_r0-abc_0511_120000")

    def test_parse_fp16_quantization(self):
        """Test parsing filename with fp16 quantization."""
        result = parse_toolemu_filename(
            "Qwen_Qwen3-8Bnt_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_fp16_r0-15_2601_110226.jsonl"
        )
        assert result['agent_model'] == "Qwen_Qwen3-8Bnt"
        assert result['emu_model'] == "Qwen_Qwen3-8B"
        assert result['eval_model'] == "gpt-5-mini"
        assert result['quantization'] == "fp16"
        assert result['range_start'] == 0
        assert result['range_end'] == 15

    def test_parse_bf16_quantization(self):
        """Test parsing filename with bf16 quantization."""
        result = parse_toolemu_filename(
            "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-Qwen_Qwen3-32B_bf16_r0-15_0511_120000.jsonl"
        )
        assert result['eval_model'] == "Qwen_Qwen3-32B"
        assert result['quantization'] == "bf16"


class TestLoadAndValidateAllEvalFiles:
    """Tests for load_and_validate_all_eval_files function."""

    def test_successful_load_all_eval_files(self, temp_test_dir):
        """Test successful loading and validation of all eval files."""
        tmpdir = temp_test_dir
        # Create trajectory file (r10-13 means indices 10, 11, 12 - range is [start, end))
        case_indices = [10, 11, 12]
        traj_data = create_trajectory_data(case_indices)
        traj_file = os.path.join(tmpdir, "model_r10-13_test.jsonl")
        write_jsonl(traj_file, traj_data)

        # Create all 5 eval files
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_data = create_eval_data(case_indices, eval_type)
            eval_file = ToolEmuFilePaths.trajectory_to_eval(traj_file, eval_type)
            write_jsonl(eval_file, eval_data)

        # Load and validate
        trajectories, evaluations_dict = load_and_validate_all_eval_files(traj_file)

        # Check trajectories
        assert len(trajectories) == 3
        assert [t['case_idx'] for t in trajectories] == case_indices

        # Check all eval types loaded
        assert len(evaluations_dict) == 3
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            assert eval_type in evaluations_dict
            assert len(evaluations_dict[eval_type]) == 3

    def test_missing_eval_file_raises_incomplete_error(self, temp_test_dir):
        """Test that missing eval file raises IncompleteTrajectoryError."""
        tmpdir = temp_test_dir
        # Create trajectory file
        case_indices = [10, 11, 12]
        traj_data = create_trajectory_data(case_indices)
        traj_file = os.path.join(tmpdir, "model_r10-13_test.jsonl")
        write_jsonl(traj_file, traj_data)

        # Create only 2 of 3 eval files (missing agent_help_ignore_safety)
        for eval_type in ['agent_safe', 'agent_help']:
            eval_data = create_eval_data(case_indices, eval_type)
            eval_file = ToolEmuFilePaths.trajectory_to_eval(traj_file, eval_type)
            write_jsonl(eval_file, eval_data)

        # Should raise IncompleteTrajectoryError about missing file
        with pytest.raises(IncompleteTrajectoryError, match="Missing required evaluation files"):
            load_and_validate_all_eval_files(traj_file)

    def test_misaligned_case_indices_raises_error(self, temp_test_dir):
        """Test that misaligned case indices raise ValueError."""
        tmpdir = temp_test_dir
        # Create trajectory file with case_indices [10, 11, 12]
        case_indices = [10, 11, 12]
        traj_data = create_trajectory_data(case_indices)
        traj_file = os.path.join(tmpdir, "model_r10-13_test.jsonl")
        write_jsonl(traj_file, traj_data)

        # Create eval files, but one has wrong alignment
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            if eval_type == 'agent_safe':
                # Create misaligned data by modifying eval_id values
                eval_data = create_eval_data(case_indices, eval_type)
                # Make eval_ids wrong: should be [0,1,2] but set to [5,6,7]
                for i, item in enumerate(eval_data):
                    item['eval_id'] = i + 5
            else:
                eval_data = create_eval_data(case_indices, eval_type)

            eval_file = ToolEmuFilePaths.trajectory_to_eval(traj_file, eval_type)
            write_jsonl(eval_file, eval_data)

        # Should raise eval_id mismatch error (caught before alignment check)
        with pytest.raises(ValueError, match="Eval ID mismatch"):
            load_and_validate_all_eval_files(traj_file)

    def test_allow_empty_scores(self, temp_test_dir):
        """Test that allow_empty_scores=True permits missing scores."""
        tmpdir = temp_test_dir
        # Create trajectory file
        case_indices = [10, 11, 12]
        traj_data = create_trajectory_data(case_indices)
        traj_file = os.path.join(tmpdir, "model_r10-13_test.jsonl")
        write_jsonl(traj_file, traj_data)

        # Create eval files with some missing scores
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            scores = [2.0, None, 3.0]  # Middle one missing
            eval_data = create_eval_data(case_indices, eval_type, scores)
            eval_file = ToolEmuFilePaths.trajectory_to_eval(traj_file, eval_type)
            write_jsonl(eval_file, eval_data)

        # Should succeed with allow_empty_scores=True
        trajectories, evaluations_dict = load_and_validate_all_eval_files(
            traj_file, allow_empty_scores=True
        )

        assert len(trajectories) == 3
        assert len(evaluations_dict) == 3

    def test_count_mismatch_raises_incomplete_error(self, temp_test_dir):
        """Test that trajectory count mismatch with filename range raises IncompleteTrajectoryError."""
        tmpdir = temp_test_dir
        # Create trajectory file with range _r10-15 but only 3 trajectories (should be 5)
        case_indices = [10, 11, 12]  # Only 3, but filename says 10-15 (5 expected)
        traj_data = create_trajectory_data(case_indices)
        traj_file = os.path.join(tmpdir, "model_r10-15_test.jsonl")
        write_jsonl(traj_file, traj_data)

        # Create all eval files
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_data = create_eval_data(case_indices, eval_type)
            eval_file = ToolEmuFilePaths.trajectory_to_eval(traj_file, eval_type)
            write_jsonl(eval_file, eval_data)

        # Should raise IncompleteTrajectoryError due to count mismatch
        with pytest.raises(IncompleteTrajectoryError, match="expected 5.*got 3"):
            load_and_validate_all_eval_files(traj_file)


class TestValidateTrajectoryCompleteness:
    """Tests for validate_trajectory_completeness function."""

    def test_count_matches_range(self):
        """Test that matching count passes validation."""
        case_indices = [10, 11, 12]
        trajectories = create_trajectory_data(case_indices)
        filepath = "/path/to/model_r10-13_test.jsonl"  # Range [10,13) = 3 expected

        is_valid, mismatch_info = validate_trajectory_completeness(trajectories, filepath)
        assert is_valid == True
        assert mismatch_info is None

    def test_count_mismatch(self):
        """Test that mismatched count fails validation."""
        case_indices = [10, 11]  # Only 2 trajectories
        trajectories = create_trajectory_data(case_indices)
        filepath = "/path/to/model_r10-15_test.jsonl"  # Range [10,15) = 5 expected

        is_valid, mismatch_info = validate_trajectory_completeness(trajectories, filepath)
        assert is_valid == False
        assert mismatch_info == (5, 2)  # (expected, actual)

    def test_no_range_in_filename(self):
        """Test that files without range in filename pass validation."""
        case_indices = [10, 11, 12]
        trajectories = create_trajectory_data(case_indices)
        filepath = "/path/to/model_test.jsonl"  # No range

        is_valid, mismatch_info = validate_trajectory_completeness(trajectories, filepath)
        assert is_valid == True
        assert mismatch_info is None

    def test_test_seed_indices_match(self):
        """Test that count mismatch passes if case indices match test indices for seed."""
        from utils.train_utils import compute_test_indices

        # Use seed 42 and get actual test indices for range [0, 15)
        test_seed = 42
        test_indices = compute_test_indices(test_seed)
        test_in_range = [i for i in test_indices if 0 <= i < 15]

        # Create trajectories with only the test indices
        trajectories = create_trajectory_data(test_in_range)
        filepath = "/path/to/model_r0-15_test.jsonl"  # Range [0,15) = 15 expected

        # Without test_seed, should fail (count mismatch)
        is_valid, mismatch_info = validate_trajectory_completeness(trajectories, filepath)
        assert is_valid == False
        assert mismatch_info == (15, len(test_in_range))

        # With test_seed, should pass (indices match test indices)
        is_valid, mismatch_info = validate_trajectory_completeness(trajectories, filepath, test_seed=test_seed)
        assert is_valid == True
        assert mismatch_info is None

    def test_test_seed_indices_mismatch(self):
        """Test that validation fails if indices don't match test indices for seed."""
        from utils.train_utils import compute_test_indices

        test_seed = 42
        test_indices = compute_test_indices(test_seed)
        test_in_range = [i for i in test_indices if 0 <= i < 15]

        # Create trajectories with different indices (not the expected test indices)
        wrong_indices = [0, 1, 2, 3, 4]  # Arbitrary indices, likely not matching test split
        trajectories = create_trajectory_data(wrong_indices)
        filepath = "/path/to/model_r0-15_test.jsonl"

        # With test_seed, should fail (indices don't match)
        is_valid, mismatch_info = validate_trajectory_completeness(trajectories, filepath, test_seed=test_seed)
        assert is_valid == False
        assert mismatch_info == (15, len(wrong_indices))


class TestValidateAlignment:
    """Tests for validate_trajectory_eval_alignment function."""

    def test_valid_alignment(self):
        """Test that valid alignment passes without error."""
        case_indices = [10, 11, 12]
        trajectories = create_trajectory_data(case_indices)
        evaluations = create_eval_data(case_indices, 'agent_safe')

        # Should return (True, "")
        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, evaluations, "model_r10-13_test.jsonl", "model_r10-13_test_eval_agent_safe.jsonl"
        )
        assert is_valid == True

    def test_mismatched_length(self):
        """Test that mismatched lengths return error."""
        trajectories = create_trajectory_data([10, 11, 12])
        evaluations = create_eval_data([10, 11], 'agent_safe')  # Only 2

        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, evaluations, "model_r10-13_traj.jsonl", "model_r10-13_traj_eval.jsonl"
        )
        assert is_valid == False
        assert "Length mismatch" in error_msg

    def test_unsorted_trajectories(self):
        """Test that unsorted trajectories return error."""
        trajectories = create_trajectory_data([10, 12, 11])  # Out of order
        evaluations = create_eval_data([10, 12, 11], 'agent_safe')

        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, evaluations, "model_r10-13_traj.jsonl", "model_r10-13_traj_eval.jsonl"
        )
        assert is_valid == False
        # Unsorted data is also misaligned, so either error message is valid
        assert ("not sorted" in error_msg or "Alignment error" in error_msg)

    def test_unsorted_evaluations(self):
        """Test that unsorted evaluations return error."""
        trajectories = create_trajectory_data([10, 11, 12])
        eval_data = create_eval_data([10, 11, 12], 'agent_safe')
        eval_data[0]['eval_id'] = 1  # Make it unsorted
        eval_data[1]['eval_id'] = 0

        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, eval_data, "model_r10-13_traj.jsonl", "model_r10-13_traj_eval.jsonl"
        )
        assert is_valid == False
        # Unsorted eval_ids are caught as eval_id mismatch (eval_id != line_number)
        assert "Eval ID mismatch" in error_msg

    def test_case_idx_gaps_allowed_without_test_seed(self):
        """Test that gaps in case_idx are allowed when no test_seed provided (for test sets)."""
        trajectories = create_trajectory_data([10, 11, 13])  # Gap at 12
        evaluations = create_eval_data([10, 11, 13], 'agent_safe')

        # Without test_seed, sparse case indices are allowed
        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, evaluations, "model_r10-14_test.jsonl", "model_r10-14_test_eval_agent_safe.jsonl"
        )
        assert is_valid == True

    def test_count_mismatch_allowed_if_traj_eval_match(self):
        """Test that count mismatch vs filename is allowed if traj/eval have matching lengths and are sorted.

        This tests the case where r0-16 (16 cases expected) only has 14 cases,
        but both traj and eval files have 14 matching, sorted entries.
        """
        # Create 14 cases even though filename says r0-16 (16 expected)
        case_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 14 cases
        trajectories = create_trajectory_data(case_indices)
        evaluations = create_eval_data(case_indices, 'agent_safe')

        # Filename says r0-16 (16 cases), but we only have 14
        # Should still pass because traj/eval lengths match and are sorted
        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, evaluations, "model_r0-16_test.jsonl", "model_r0-16_test_eval_agent_safe.jsonl"
        )
        assert is_valid == True, f"Expected valid but got error: {error_msg}"

    def test_count_mismatch_fails_if_traj_eval_mismatch(self):
        """Test that validation fails when traj and eval have different lengths."""
        # 14 trajectories but only 12 evaluations
        traj_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        eval_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        trajectories = create_trajectory_data(traj_indices)
        evaluations = create_eval_data(eval_indices, 'agent_safe')

        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, evaluations, "model_r0-16_test.jsonl", "model_r0-16_test_eval_agent_safe.jsonl"
        )
        assert is_valid == False
        assert "Length mismatch" in error_msg

    def test_count_mismatch_fails_if_traj_unsorted(self):
        """Test that validation fails when trajectories are not sorted by case_idx."""
        # 14 cases but unsorted
        case_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 12]  # 12 and 13 swapped
        trajectories = create_trajectory_data(case_indices)
        evaluations = create_eval_data(case_indices, 'agent_safe')

        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, evaluations, "model_r0-16_test.jsonl", "model_r0-16_test_eval_agent_safe.jsonl"
        )
        assert is_valid == False
        assert "not sorted" in error_msg

    def test_count_mismatch_fails_if_eval_unsorted(self):
        """Test that validation fails when evaluations are not sorted by eval_id."""
        case_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        trajectories = create_trajectory_data(case_indices)
        evaluations = create_eval_data(case_indices, 'agent_safe')

        # Swap eval_ids to make them unsorted
        evaluations[12]['eval_id'] = 13
        evaluations[13]['eval_id'] = 12

        is_valid, error_msg = validate_trajectory_eval_alignment(
            trajectories, evaluations, "model_r0-16_test.jsonl", "model_r0-16_test_eval_agent_safe.jsonl"
        )
        assert is_valid == False
        # eval_id must match line number, so swapping causes mismatch
        assert "Eval ID mismatch" in error_msg

    # Note: test_seed completeness validation (check 6) was removed from
    # validate_trajectory_eval_alignment. Completeness checking is now done
    # separately by validate_trajectory_completeness.


class TestLoadAndValidatePair:
    """Tests for load_and_validate_trajectory_eval_pair function."""

    def test_successful_load(self, temp_test_dir):
        """Test successful loading and validation."""
        tmpdir = temp_test_dir
        case_indices = [5, 6, 7]

        # Create files with range in filename
        traj_data = create_trajectory_data(case_indices)
        eval_data = create_eval_data(case_indices, 'agent_safe')

        traj_file = os.path.join(tmpdir, "test_r5-8_traj.jsonl")
        eval_file = os.path.join(tmpdir, "test_r5-8_traj_eval_agent_safe.jsonl")

        write_jsonl(traj_file, traj_data)
        write_jsonl(eval_file, eval_data)

        # Load
        trajectories, evaluations = load_and_validate_trajectory_eval_pair(
            traj_file, eval_file
        )

        assert len(trajectories) == 3
        assert len(evaluations) == 3
        assert [t['case_idx'] for t in trajectories] == case_indices

    def test_nonexistent_files(self):
        """Test that nonexistent files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_and_validate_trajectory_eval_pair(
                "/nonexistent/traj.jsonl",
                "/nonexistent/eval.jsonl"
            )

    def test_count_mismatch_raises_incomplete_error(self, temp_test_dir):
        """Test that trajectory count mismatch with filename range raises IncompleteTrajectoryError."""
        tmpdir = temp_test_dir
        # Create trajectory file with range _r5-10 but only 3 trajectories (should be 5)
        case_indices = [5, 6, 7]  # Only 3, but filename says 5-10 (5 expected)
        traj_data = create_trajectory_data(case_indices)
        eval_data = create_eval_data(case_indices, 'agent_safe')

        traj_file = os.path.join(tmpdir, "test_r5-10_traj.jsonl")
        eval_file = os.path.join(tmpdir, "test_r5-10_traj_eval_agent_safe.jsonl")

        write_jsonl(traj_file, traj_data)
        write_jsonl(eval_file, eval_data)

        # Should raise IncompleteTrajectoryError due to count mismatch
        with pytest.raises(IncompleteTrajectoryError, match="expected 5.*got 3"):
            load_and_validate_trajectory_eval_pair(traj_file, eval_file)


class TestIntegration:
    """Integration tests simulating real-world usage patterns."""

    @pytest.mark.integration
    def test_compare_source_finetuned_workflow(self, temp_test_dir):
        """Test the workflow used by compare_source_finetuned.py."""
        tmpdir = temp_test_dir
        case_indices = [10, 11, 12]

        # Create source model files
        source_traj = os.path.join(tmpdir, "source_model_r10-13_test.jsonl")
        write_jsonl(source_traj, create_trajectory_data(case_indices))

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = ToolEmuFilePaths.trajectory_to_eval(source_traj, eval_type)
            scores = [1.0, 2.0, 3.0] if eval_type == 'agent_safe' else [2.0, 2.0, 2.0]
            write_jsonl(eval_file, create_eval_data(case_indices, eval_type, scores))

        # Create finetuned model files
        ft_traj = os.path.join(tmpdir, "finetuned_model_r10-13_test.jsonl")
        write_jsonl(ft_traj, create_trajectory_data(case_indices))

        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = ToolEmuFilePaths.trajectory_to_eval(ft_traj, eval_type)
            scores = [2.0, 2.0, 3.0] if eval_type == 'agent_safe' else [3.0, 3.0, 3.0]
            write_jsonl(eval_file, create_eval_data(case_indices, eval_type, scores))

        # Simulate compare_source_finetuned workflow
        # Load source safety scores
        source_safe_file = ToolEmuFilePaths.trajectory_to_eval(source_traj, 'agent_safe')
        source_trajs, source_evals_dict = load_and_validate_all_eval_files(
            source_traj, allow_empty_scores=True
        )
        source_safe_evals = source_evals_dict['agent_safe']

        # Extract scores
        source_scores = {}
        for traj, eval_data in zip(source_trajs, source_safe_evals):
            case_idx = traj['case_idx']
            score = extract_score(eval_data, 'agent_safe', case_idx, allow_missing=True)
            if score is not None:
                source_scores[case_idx] = score

        # Verify
        assert source_scores == {10: 1.0, 11: 2.0, 12: 3.0}

    @pytest.mark.integration
    def test_prepare_dpo_data_workflow(self, temp_test_dir):
        """Test the workflow used by prepare_dpo_data.py."""
        tmpdir = temp_test_dir
        case_indices = [20, 21, 22]

        # Create trajectory file
        traj_file = os.path.join(tmpdir, "model_r20-23_test.jsonl")
        write_jsonl(traj_file, create_trajectory_data(case_indices))

        # Create eval files
        for eval_type in ToolEmuFilePaths.EVAL_TYPES:
            eval_file = ToolEmuFilePaths.trajectory_to_eval(traj_file, eval_type)
            scores = [3.0, 2.0, 1.0]
            write_jsonl(eval_file, create_eval_data(case_indices, eval_type, scores))

        # Simulate prepare_dpo_data workflow
        # Check if this is a trajectory file
        assert ToolEmuFilePaths.is_trajectory_file(os.path.basename(traj_file))

        # Get eval suffix for help_ignore_safety
        eval_suffix = ToolEmuFilePaths.get_eval_suffix('help_ignore_safety')
        assert eval_suffix == '_eval_agent_help_ignore_safety.jsonl'

        # Load trajectory and eval pair
        eval_type = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE['help_ignore_safety']
        eval_file = ToolEmuFilePaths.trajectory_to_eval(traj_file, eval_type)

        trajectories, evaluations = load_and_validate_trajectory_eval_pair(
            traj_file, eval_file, allow_empty_scores=True
        )

        # Extract scores
        trajectory_scores = []
        for traj, eval_data in zip(trajectories, evaluations):
            case_idx = traj['case_idx']
            score = extract_score(eval_data, eval_type, case_idx, allow_missing=True)
            trajectory_scores.append((case_idx, score, traj))

        # Verify
        assert len(trajectory_scores) == 3
        assert trajectory_scores[0][1] == 3.0
        assert trajectory_scores[1][1] == 2.0
        assert trajectory_scores[2][1] == 1.0

    @pytest.mark.integration
    def test_consistent_split_determinism(self, temp_test_dir):
        """Test that compute_test_indices always produces same split for same seed."""
        # Compute split multiple times with same seed
        results = [compute_test_indices(42) for _ in range(3)]

        # All results should be identical
        assert results[0] == results[1] == results[2]

        # Check expected properties
        assert len(results[0]) == 72  # Half of 144
        assert all(0 <= idx < 144 for idx in results[0])


class TestValidateNoOverlappingFiles:
    """Test validate_no_overlapping_files() function"""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test files."""
        return tmp_path

    def create_trajectory_file(self, temp_dir, subdir, filename):
        """Helper to create an empty trajectory file with proper directory structure."""
        model_dir = temp_dir / subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        filepath = model_dir / filename
        filepath.write_text('{"case_idx": 0}\n')
        return filepath

    def test_no_overlaps_adjacent_ranges(self, temp_dir):
        """Test that adjacent ranges (r0-15, r15-30) don't trigger overlap error."""
        files = [
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl"),
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r15-30_0101_120001.jsonl"),
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r30-45_0101_120002.jsonl"),
        ]
        # Should not raise
        validate_no_overlapping_files(files)

    def test_overlapping_ranges_raises_error(self, temp_dir):
        """Test that overlapping ranges raise ValueError."""
        files = [
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl"),
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r10-25_0101_120001.jsonl"),
        ]
        with pytest.raises(ValueError, match="Overlapping trajectory files detected"):
            validate_no_overlapping_files(files)

    def test_duplicate_ranges_raises_error(self, temp_dir):
        """Test that duplicate ranges (same r0-15 twice) raise ValueError."""
        files = [
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl"),
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-15_0101_130000.jsonl"),
        ]
        with pytest.raises(ValueError, match="Overlapping trajectory files detected"):
            validate_no_overlapping_files(files)

    def test_different_configs_can_have_same_ranges(self, temp_dir):
        """Test that different configurations can have the same ranges without error."""
        files = [
            self.create_trajectory_file(temp_dir, "model1", "model1_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl"),
            self.create_trajectory_file(temp_dir, "model2", "model2_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl"),
        ]
        # Should not raise - different agent models
        validate_no_overlapping_files(files)

    def test_different_quantization_separate_configs(self, temp_dir):
        """Test that different quantizations are treated as separate configurations."""
        files = [
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl"),
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int8_r0-15_0101_120000.jsonl"),
        ]
        # Should not raise - different quantizations
        validate_no_overlapping_files(files)

    def test_files_without_range_skipped(self, temp_dir):
        """Test that files without range info are skipped (full dataset runs)."""
        files = [
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_0101_120000.jsonl"),
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_0101_130000.jsonl"),
        ]
        # Should not raise - no range info means full dataset, validation skips these
        validate_no_overlapping_files(files)

    def test_empty_file_list(self, temp_dir):
        """Test that empty file list doesn't raise error."""
        validate_no_overlapping_files([])

    def test_single_file(self, temp_dir):
        """Test that single file doesn't raise error."""
        files = [
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl"),
        ]
        validate_no_overlapping_files(files)

    def test_error_message_contains_details(self, temp_dir):
        """Test that error message contains useful details about the overlap."""
        files = [
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-20_0101_120000.jsonl"),
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r15-30_0101_120001.jsonl"),
        ]
        with pytest.raises(ValueError) as exc_info:
            validate_no_overlapping_files(files)

        error_msg = str(exc_info.value)
        assert "agent=model" in error_msg
        assert "emu=sim" in error_msg
        assert "eval=eval" in error_msg
        assert "r0-20" in error_msg
        assert "r15-30" in error_msg
        assert "[15, 20)" in error_msg  # Overlap range

    def test_multiple_overlaps_all_reported(self, temp_dir):
        """Test that multiple overlaps in same config are all reported."""
        files = [
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-20_0101_120000.jsonl"),
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r15-35_0101_120001.jsonl"),
            self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r30-50_0101_120002.jsonl"),
        ]
        with pytest.raises(ValueError) as exc_info:
            validate_no_overlapping_files(files)

        error_msg = str(exc_info.value)
        # Should report both overlaps: r0-20 with r15-35, and r15-35 with r30-50
        assert error_msg.count("Configuration:") >= 2

    def test_unparseable_filename_raises_error(self, temp_dir):
        """Test that unparseable filenames raise an error (fail fast)."""
        files = [
            self.create_trajectory_file(temp_dir, "model", "invalid_filename_without_markers.jsonl"),
        ]
        with pytest.raises(ValueError, match="missing required '_emu-' marker"):
            validate_no_overlapping_files(files)


class TestFindTrajectoryFilesValidation:
    """Test find_trajectory_files() with validation enabled/disabled."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test files."""
        return tmp_path

    def create_trajectory_file(self, temp_dir, subdir, filename):
        """Helper to create an empty trajectory file with proper directory structure."""
        model_dir = temp_dir / subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        filepath = model_dir / filename
        filepath.write_text('{"case_idx": 0}\n')
        return filepath

    def test_find_with_validation_enabled_raises_on_overlap(self, temp_dir):
        """Test that find_trajectory_files raises error on overlapping files by default."""
        self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl")
        self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r10-25_0101_120001.jsonl")

        with pytest.raises(ValueError, match="Overlapping trajectory files detected"):
            find_trajectory_files(temp_dir)

    def test_find_with_validation_disabled_allows_overlap(self, temp_dir):
        """Test that find_trajectory_files allows overlaps when validation disabled."""
        self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl")
        self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r10-25_0101_120001.jsonl")

        # Should not raise when validation is disabled
        files = find_trajectory_files(temp_dir, validate_no_overlaps=False)
        assert len(files) == 2

    def test_find_with_validation_enabled_passes_clean_data(self, temp_dir):
        """Test that find_trajectory_files works with non-overlapping files."""
        self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r0-15_0101_120000.jsonl")
        self.create_trajectory_file(temp_dir, "model", "model_emu-sim_eval-eval_int4_r15-30_0101_120001.jsonl")

        files = find_trajectory_files(temp_dir)
        assert len(files) == 2


class TestComputeEvalConsistency:
    """Tests for compute_eval_consistency function."""

    def test_all_same_scores(self):
        """All replicates have the same score."""
        result = compute_eval_consistency([2, 2, 2])
        assert result["scores"] == [2, 2, 2]
        assert result["mean"] == 2.0
        assert result["std"] == 0.0
        assert result["variance"] == 0.0
        assert result["exact_match"] is True
        assert result["majority_agree"] is True
        assert result["all_but_one_agree"] is True
        assert result["within_1"] is True
        assert result["range"] == 0

    def test_all_different_scores(self):
        """All replicates have different scores."""
        result = compute_eval_consistency([0, 1, 3])
        assert result["scores"] == [0, 1, 3]
        assert result["mean"] == pytest.approx(4/3)
        assert result["exact_match"] is False
        assert result["majority_agree"] is False
        assert result["all_but_one_agree"] is False
        assert result["within_1"] is False
        assert result["range"] == 3

    def test_within_1_range(self):
        """Scores within 1 point of each other."""
        result = compute_eval_consistency([2, 3, 2])
        assert result["exact_match"] is False
        assert result["majority_agree"] is True  # 2 appears twice out of 3
        assert result["all_but_one_agree"] is True  # 2 out of 3 agree
        assert result["within_1"] is True
        assert result["range"] == 1

    def test_two_replicates(self):
        """Works with only 2 replicates."""
        result = compute_eval_consistency([1, 2])
        assert result["scores"] == [1, 2]
        assert result["mean"] == 1.5
        assert result["exact_match"] is False
        assert result["majority_agree"] is False  # 1 out of 2 is not > 50%
        assert result["all_but_one_agree"] is True  # max_count (1) >= len-1 (1)
        assert result["within_1"] is True
        assert result["range"] == 1

    def test_two_replicates_same(self):
        """Two replicates with same score."""
        result = compute_eval_consistency([2, 2])
        assert result["exact_match"] is True
        assert result["majority_agree"] is True  # 2 out of 2 is > 50%
        assert result["all_but_one_agree"] is True

    def test_single_replicate(self):
        """Works with single replicate (edge case)."""
        result = compute_eval_consistency([3])
        assert result["scores"] == [3]
        assert result["mean"] == 3.0
        assert result["std"] == 0.0
        assert result["exact_match"] is True
        assert result["majority_agree"] is True
        assert result["all_but_one_agree"] is False  # requires n >= 2
        assert result["within_1"] is True
        assert result["range"] == 0

    def test_all_but_one_agree_with_four_replicates(self):
        """3 out of 4 replicates agree = all_but_one_agree is True."""
        result = compute_eval_consistency([2, 2, 2, 1])
        assert result["exact_match"] is False
        assert result["majority_agree"] is True  # 3/4 > 50%
        assert result["all_but_one_agree"] is True  # 3 >= 4-1

    def test_all_but_one_agree_fails_with_two_outliers(self):
        """2 out of 4 replicates disagree = all_but_one_agree is False."""
        result = compute_eval_consistency([2, 2, 1, 1])
        assert result["exact_match"] is False
        assert result["majority_agree"] is False  # 2/4 is not > 50%
        assert result["all_but_one_agree"] is False  # 2 < 4-1

    def test_all_but_one_agree_with_five_replicates(self):
        """4 out of 5 replicates agree = all_but_one_agree is True."""
        result = compute_eval_consistency([3, 3, 3, 3, 1])
        assert result["majority_agree"] is True  # 4/5 > 50%
        assert result["all_but_one_agree"] is True  # 4 >= 5-1

    def test_empty_list_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_eval_consistency([])

    def test_output_types(self):
        """Verify output types are JSON-serializable."""
        result = compute_eval_consistency([1, 2, 3])
        assert isinstance(result["scores"], list)
        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)
        assert isinstance(result["variance"], float)
        assert isinstance(result["exact_match"], bool)
        assert isinstance(result["majority_agree"], bool)
        assert isinstance(result["all_but_one_agree"], bool)
        assert isinstance(result["within_1"], bool)
        assert isinstance(result["range"], int)


class TestGetAllButOneAgree:
    """Tests for _get_all_but_one_agree backward compatibility helper."""

    def test_with_field_present_true(self):
        """Returns stored value when field exists (True)."""
        from utils.toolemu_utils import _get_all_but_one_agree
        c = {"scores": [1, 2, 3], "all_but_one_agree": True}
        assert _get_all_but_one_agree(c) is True

    def test_with_field_present_false(self):
        """Returns stored value when field exists (False)."""
        from utils.toolemu_utils import _get_all_but_one_agree
        c = {"scores": [1, 2, 3], "all_but_one_agree": False}
        assert _get_all_but_one_agree(c) is False

    def test_without_field_computes_true(self):
        """Computes from scores when field missing (should be True)."""
        from utils.toolemu_utils import _get_all_but_one_agree
        c = {"scores": [2, 2, 2, 1]}  # 3 out of 4 agree
        assert _get_all_but_one_agree(c) is True

    def test_without_field_computes_false(self):
        """Computes from scores when field missing (should be False)."""
        from utils.toolemu_utils import _get_all_but_one_agree
        c = {"scores": [1, 2, 3, 0]}  # all different
        assert _get_all_but_one_agree(c) is False

    def test_single_replicate_returns_false(self):
        """Single replicate returns False (requires n >= 2)."""
        from utils.toolemu_utils import _get_all_but_one_agree
        c = {"scores": [2]}
        assert _get_all_but_one_agree(c) is False


class TestAggregateConsistencyStats:
    """Tests for aggregate_consistency_stats function."""

    def test_with_new_format_files(self):
        """Works with new files that have all_but_one_agree field."""
        from utils.toolemu_utils import aggregate_consistency_stats
        eval_preds = [
            {"consistency": {"scores": [2, 2, 2], "exact_match": True, "majority_agree": True, "all_but_one_agree": True, "within_1": True, "std": 0.0, "variance": 0.0}},
            {"consistency": {"scores": [1, 2, 3], "exact_match": False, "majority_agree": False, "all_but_one_agree": False, "within_1": False, "std": 0.8, "variance": 0.67}},
        ]
        result = aggregate_consistency_stats(eval_preds)
        assert result["all_but_one_agree_rate"] == 0.5

    def test_with_old_format_files(self):
        """Works with old files missing all_but_one_agree field."""
        from utils.toolemu_utils import aggregate_consistency_stats
        eval_preds = [
            {"consistency": {"scores": [2, 2, 2, 1], "exact_match": False, "majority_agree": True, "within_1": True, "std": 0.43, "variance": 0.19}},  # 3/4 agree -> True
            {"consistency": {"scores": [1, 2, 1, 2], "exact_match": False, "majority_agree": False, "within_1": True, "std": 0.5, "variance": 0.25}},  # 2/4 agree -> False
        ]
        result = aggregate_consistency_stats(eval_preds)
        assert result["all_but_one_agree_rate"] == 0.5

    def test_empty_returns_none(self):
        """Returns None for empty input."""
        from utils.toolemu_utils import aggregate_consistency_stats
        assert aggregate_consistency_stats([]) is None
        assert aggregate_consistency_stats([{"no_consistency": True}]) is None


class TestCheckTrajectoryExists:
    """Tests for check_trajectory_exists function."""

    def test_trajectory_exists_returns_true(self, temp_test_dir):
        """Returns (True, filepath) when trajectory file exists."""
        # Create directory structure matching sanitize_model_name output
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Create a trajectory file with the expected naming pattern
        traj_file = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_r0-15_0101_120000.jsonl"
        write_jsonl(traj_file, [{"case_idx": 0, "output": "test"}])

        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is True
        assert filepath is not None
        assert "r0-15" in filepath

    def test_trajectory_not_exists_returns_false(self, temp_test_dir):
        """Returns (False, None) when trajectory file doesn't exist."""
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Don't create any trajectory file

        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is False
        assert filepath is None

    def test_directory_not_exists_returns_false(self, temp_test_dir):
        """Returns (False, None) when output directory doesn't exist."""
        output_dir = temp_test_dir / "nonexistent"

        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is False
        assert filepath is None

    def test_ignores_eval_files(self, temp_test_dir):
        """Does not match evaluation files (only trajectory files)."""
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Create only an eval file, no trajectory file
        eval_file = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_r0-15_0101_120000_eval_agent_safe.jsonl"
        write_jsonl(eval_file, [{"eval_id": 0, "eval_scores": {"ToolCallRisk": 2}}])

        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is False
        assert filepath is None

    def test_different_range_not_matched(self, temp_test_dir):
        """Trajectory with different range is not matched."""
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Create trajectory for range 15-30
        traj_file = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_r15-30_0101_120000.jsonl"
        write_jsonl(traj_file, [{"case_idx": 15, "output": "test"}])

        # Check for range 0-15
        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is False
        assert filepath is None

    def test_finetuned_model_with_adapter_path(self, temp_test_dir):
        """Works with finetuned model using adapter path syntax."""
        output_dir = temp_test_dir / "output" / "trajectories"
        # sanitize_model_name("Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_most/final") -> "dpo_Qwen-8B_int4_most"
        model_dir = output_dir / "dpo_Qwen-8B_int4_most"
        model_dir.mkdir(parents=True)

        traj_file = model_dir / "dpo_Qwen-8B_int4_most_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_r0-15_0101_120000.jsonl"
        write_jsonl(traj_file, [{"case_idx": 0, "output": "test"}])

        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B+dpo_output/Qwen-8B_int4_most/final",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is True
        assert filepath is not None

    def test_multiple_trajectory_files_returns_first(self, temp_test_dir):
        """When multiple trajectory files match, returns first one found."""
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Create two trajectory files with same range but different timestamps
        traj_file1 = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_r0-15_0101_120000.jsonl"
        traj_file2 = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_r0-15_0102_130000.jsonl"
        write_jsonl(traj_file1, [{"case_idx": 0, "output": "test1"}])
        write_jsonl(traj_file2, [{"case_idx": 0, "output": "test2"}])

        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is True
        assert filepath is not None
        # Should return one of the matching files
        assert "r0-15" in filepath

    def test_different_simulator_not_matched(self, temp_test_dir):
        """Trajectory with different simulator is not matched."""
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Create trajectory with Qwen3-32B as simulator
        traj_file = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-gpt-5-mini_r0-15_0101_120000.jsonl"
        write_jsonl(traj_file, [{"case_idx": 0, "output": "test"}])

        # Check for Qwen3-8B as simulator (different)
        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",  # Different from file
            evaluator_model="gpt-5-mini",
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is False
        assert filepath is None

    def test_different_evaluator_not_matched(self, temp_test_dir):
        """Trajectory with different evaluator is not matched."""
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Create trajectory with gpt-4o as evaluator
        traj_file = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-8B_eval-gpt-4o_r0-15_0101_120000.jsonl"
        write_jsonl(traj_file, [{"case_idx": 0, "output": "test"}])

        # Check for gpt-5-mini as evaluator (different)
        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",  # Different from file
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is False
        assert filepath is None

    def test_with_quantization_in_filename(self, temp_test_dir):
        """Matches files that have quantization suffix (e.g., int8)."""
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Create trajectory with int8 quantization in filename
        traj_file = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_int8_r0-15_0101_120000.jsonl"
        write_jsonl(traj_file, [{"case_idx": 0, "output": "test"}])

        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range="0-15",
            output_dir=str(output_dir),
        )

        assert exists is True
        assert filepath is not None
        assert "int8" in filepath

    def test_full_dataset_no_range(self, temp_test_dir):
        """Matches full dataset runs (no range in filename) when case_range=None."""
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Create trajectory without range in filename (full dataset run)
        traj_file = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_0101_120000.jsonl"
        write_jsonl(traj_file, [{"case_idx": 0, "output": "test"}])

        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range=None,  # Full dataset
            output_dir=str(output_dir),
        )

        assert exists is True
        assert filepath is not None
        assert "_r" not in os.path.basename(filepath) or not any(c.isdigit() for c in os.path.basename(filepath).split("_r")[-1].split("_")[0])

    def test_full_dataset_ignores_ranged_files(self, temp_test_dir):
        """Full dataset check (case_range=None) should not match ranged files."""
        output_dir = temp_test_dir / "output" / "trajectories"
        model_dir = output_dir / "Qwen_Qwen3-8B"
        model_dir.mkdir(parents=True)

        # Create only a ranged trajectory (no full dataset run)
        traj_file = model_dir / "Qwen_Qwen3-8B_emu-Qwen_Qwen3-8B_eval-gpt-5-mini_r0-15_0101_120000.jsonl"
        write_jsonl(traj_file, [{"case_idx": 0, "output": "test"}])

        exists, filepath = check_trajectory_exists(
            agent_model="Qwen/Qwen3-8B",
            emulator_model="Qwen/Qwen3-8B",
            evaluator_model="gpt-5-mini",
            case_range=None,  # Full dataset - should NOT match ranged file
            output_dir=str(output_dir),
        )

        assert exists is False
        assert filepath is None


# =============================================================================
# Tests for analysis utilities
# =============================================================================

from utils.analysis_utils import calc_col_width


class TestCalcColWidth:
    """Tests for calc_col_width function."""

    def test_header_longest(self):
        """Header is longer than all values."""
        assert calc_col_width("Token Limit", ["5", "10", "100"]) == 11

    def test_value_longest(self):
        """A value is longer than header."""
        assert calc_col_width("Rate", ["0.5%", "100.0%", "5.0%"]) == 6

    def test_empty_values(self):
        """Empty values list uses header length."""
        assert calc_col_width("Model", []) == 5

    def test_converts_to_str(self):
        """Non-string values are converted to strings."""
        assert calc_col_width("Count", [1, 100, 1000]) == 5  # "Count" is 5, "1000" is 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
