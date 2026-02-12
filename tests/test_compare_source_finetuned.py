#!/usr/bin/env python3
"""Tests for compare_source_finetuned.py"""

import sys
from pathlib import Path

import pytest

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "analysis"))

from compare_source_finetuned import (
    parse_args,
    compare_scores,
    compute_mean_delta,
    format_comparison_report,
    VALID_METRICS,
)
from utils.toolemu_utils import ToolEmuFilePaths


class TestParseArgs:
    """Test argument parsing for metric selection."""

    def test_valid_metrics_list(self):
        """Test that VALID_METRICS matches ToolEmuFilePaths.METRIC_TO_EVAL_TYPE keys."""
        assert set(VALID_METRICS) == set(ToolEmuFilePaths.METRIC_TO_EVAL_TYPE.keys())

    def test_default_metrics(self):
        """Test default metric values."""
        sys.argv = [
            'compare_source_finetuned.py',
            '-b', 'Qwen3-8B',
            '-f', 'Qwen-8B_int4_most',
            '-s', 'Qwen_Qwen3-32B',
            '-e', 'Qwen_Qwen3-32B',
        ]
        args = parse_args()
        assert args.metric1 == 'safe'
        assert args.metric2 == 'help_ignore_safety'

    def test_custom_metrics(self):
        """Test specifying custom metrics."""
        sys.argv = [
            'compare_source_finetuned.py',
            '-b', 'Qwen3-8B',
            '-f', 'Qwen-8B_int4_most',
            '-s', 'Qwen_Qwen3-32B',
            '-e', 'Qwen_Qwen3-32B',
            '--metric1', 'help',
            '--metric2', 'help_ignore_safety',
        ]
        args = parse_args()
        assert args.metric1 == 'help'
        assert args.metric2 == 'help_ignore_safety'

    def test_same_metric_for_both(self):
        """Test using the same metric for both (edge case, but valid)."""
        sys.argv = [
            'compare_source_finetuned.py',
            '-b', 'Qwen3-8B',
            '-f', 'Qwen-8B_int4_most',
            '-s', 'Qwen_Qwen3-32B',
            '-e', 'Qwen_Qwen3-32B',
            '--metric1', 'safe',
            '--metric2', 'safe',
        ]
        args = parse_args()
        assert args.metric1 == 'safe'
        assert args.metric2 == 'safe'

    def test_all_valid_metric_choices(self):
        """Test that all valid metrics can be specified."""
        for metric in VALID_METRICS:
            sys.argv = [
                'compare_source_finetuned.py',
                '-b', 'Qwen3-8B',
                '-f', 'Qwen-8B_int4_most',
                '-s', 'Qwen_Qwen3-32B',
                '-e', 'Qwen_Qwen3-32B',
                '--metric1', metric,
                '--metric2', metric,
            ]
            args = parse_args()
            assert args.metric1 == metric
            assert args.metric2 == metric

    def test_invalid_metric_rejected(self):
        """Test that invalid metrics are rejected."""
        sys.argv = [
            'compare_source_finetuned.py',
            '-b', 'Qwen3-8B',
            '-f', 'Qwen-8B_int4_most',
            '-s', 'Qwen_Qwen3-32B',
            '-e', 'Qwen_Qwen3-32B',
            '--metric1', 'invalid_metric',
        ]
        with pytest.raises(SystemExit):
            parse_args()


class TestCompareScores:
    """Test the compare_scores function."""

    def test_compare_scores_basic(self):
        """Test basic score comparison."""
        base_scores = {0: 2.0, 1: 1.0, 2: 3.0}
        ft_scores = {0: 3.0, 1: 1.0, 2: 1.0}

        diffs = compare_scores(base_scores, ft_scores, "test_metric", min_diff=1)

        # Case 0: diff = 1.0 (meets threshold)
        # Case 1: diff = 0.0 (below threshold)
        # Case 2: diff = -2.0 (meets threshold)
        assert len(diffs) == 2
        assert (0, 2.0, 3.0, 1.0) in diffs
        assert (2, 3.0, 1.0, -2.0) in diffs

    def test_compare_scores_min_diff_threshold(self):
        """Test that min_diff threshold filters correctly."""
        base_scores = {0: 2.0, 1: 2.0}
        ft_scores = {0: 2.5, 1: 4.0}

        # With min_diff=1, only case 1 (diff=2.0) should be included
        diffs = compare_scores(base_scores, ft_scores, "test_metric", min_diff=1)
        assert len(diffs) == 1
        assert diffs[0][0] == 1  # case_idx

        # With min_diff=0.5, case 0 (diff=0.5) should also be included
        diffs = compare_scores(base_scores, ft_scores, "test_metric", min_diff=0)
        assert len(diffs) == 2

    def test_compare_scores_missing_cases(self, capsys):
        """Test handling of missing cases in either model."""
        base_scores = {0: 2.0, 1: 1.0}
        ft_scores = {0: 3.0, 2: 2.0}

        diffs = compare_scores(base_scores, ft_scores, "test_metric", min_diff=1)

        # Only case 0 has scores in both
        assert len(diffs) == 1
        assert diffs[0][0] == 0

        # Should print warnings for missing cases
        captured = capsys.readouterr()
        assert "case_idx 1 missing from finetuned model" in captured.out
        assert "case_idx 2 missing from source model" in captured.out

    def test_compare_scores_empty(self):
        """Test comparing empty score dicts."""
        diffs = compare_scores({}, {}, "test_metric", min_diff=1)
        assert len(diffs) == 0

    def test_compare_scores_no_diffs_above_threshold(self):
        """Test when no differences meet threshold."""
        base_scores = {0: 2.0, 1: 2.0}
        ft_scores = {0: 2.0, 1: 2.5}

        diffs = compare_scores(base_scores, ft_scores, "test_metric", min_diff=1)
        assert len(diffs) == 0


class TestComputeMeanDelta:
    """Test the compute_mean_delta function."""

    def test_compute_mean_delta_basic(self):
        """Test basic mean delta computation."""
        base_scores = {0: 1.0, 1: 2.0, 2: 3.0}
        ft_scores = {0: 2.0, 1: 3.0, 2: 4.0}

        mean_delta, num_cases = compute_mean_delta(base_scores, ft_scores)

        assert num_cases == 3
        assert mean_delta == 1.0  # All deltas are +1.0

    def test_compute_mean_delta_mixed(self):
        """Test mean delta with positive and negative changes."""
        base_scores = {0: 2.0, 1: 2.0}
        ft_scores = {0: 3.0, 1: 1.0}

        mean_delta, num_cases = compute_mean_delta(base_scores, ft_scores)

        assert num_cases == 2
        assert mean_delta == 0.0  # (+1 + -1) / 2 = 0

    def test_compute_mean_delta_partial_overlap(self):
        """Test mean delta only computed for overlapping cases."""
        base_scores = {0: 1.0, 1: 2.0}
        ft_scores = {1: 4.0, 2: 3.0}

        mean_delta, num_cases = compute_mean_delta(base_scores, ft_scores)

        # Only case 1 overlaps
        assert num_cases == 1
        assert mean_delta == 2.0  # 4.0 - 2.0

    def test_compute_mean_delta_no_overlap(self):
        """Test mean delta with no overlapping cases."""
        base_scores = {0: 1.0}
        ft_scores = {1: 2.0}

        mean_delta, num_cases = compute_mean_delta(base_scores, ft_scores)

        assert num_cases == 0
        assert mean_delta == 0.0

    def test_compute_mean_delta_empty(self):
        """Test mean delta with empty inputs."""
        mean_delta, num_cases = compute_mean_delta({}, {})
        assert num_cases == 0
        assert mean_delta == 0.0


class TestFormatComparisonReport:
    """Test the format_comparison_report function."""

    def test_format_report_includes_metrics(self):
        """Test that report includes specified metric names."""
        report = format_comparison_report(
            metric1_diffs=[],
            metric2_diffs=[],
            source_model="BaseModel",
            finetuned_model="FinetunedModel",
            metric1_mean_delta=0.0,
            metric1_total_cases=0,
            metric2_mean_delta=0.0,
            metric2_total_cases=0,
            metric1="safe",
            metric2="help_ignore_safety",
            show_all=False,
        )

        assert "safe" in report
        assert "help_ignore_safety" in report
        assert "Metrics:" in report

    def test_format_report_custom_metrics(self):
        """Test report with custom metric names."""
        report = format_comparison_report(
            metric1_diffs=[(0, 1.0, 2.0, 1.0)],
            metric2_diffs=[(0, 2.0, 1.0, -1.0)],
            source_model="BaseModel",
            finetuned_model="FinetunedModel",
            metric1_mean_delta=1.0,
            metric1_total_cases=1,
            metric2_mean_delta=-1.0,
            metric2_total_cases=1,
            metric1="help",
            metric2="help_ignore_safety",
            show_all=False,
        )

        assert "help" in report
        assert "help_ignore_safety" in report
        # Check that the change indicators use truncated names
        assert "+help" in report or "-help" in report

    def test_format_report_summary_counts(self):
        """Test that summary correctly counts improvements/regressions."""
        report = format_comparison_report(
            metric1_diffs=[
                (0, 1.0, 2.0, 1.0),   # Improved
                (1, 2.0, 1.0, -1.0),  # Worse
                (2, 1.0, 3.0, 2.0),   # Improved
            ],
            metric2_diffs=[
                (0, 2.0, 3.0, 1.0),   # Improved
            ],
            source_model="Base",
            finetuned_model="FT",
            metric1_mean_delta=0.67,
            metric1_total_cases=3,
            metric2_mean_delta=1.0,
            metric2_total_cases=1,
            metric1="safe",
            metric2="help",
            show_all=False,
        )

        # Check metric1 (safe) summary
        assert "Improved:     2 cases" in report  # safe improved 2
        assert "Worse:        1 cases" in report  # safe worse 1
        assert "Total diffs:  3 cases" in report  # safe total 3

    def test_format_report_no_diffs(self):
        """Test report when no differences found."""
        report = format_comparison_report(
            metric1_diffs=[],
            metric2_diffs=[],
            source_model="Base",
            finetuned_model="FT",
            metric1_mean_delta=0.0,
            metric1_total_cases=10,
            metric2_mean_delta=0.0,
            metric2_total_cases=10,
            metric1="safe",
            metric2="help",
            show_all=False,
        )

        assert "No score differences found" in report

    def test_format_report_model_names(self):
        """Test that model names appear in report."""
        report = format_comparison_report(
            metric1_diffs=[],
            metric2_diffs=[],
            source_model="Qwen3-8B",
            finetuned_model="Qwen-8B_int4_most",
            metric1_mean_delta=0.0,
            metric1_total_cases=0,
            metric2_mean_delta=0.0,
            metric2_total_cases=0,
            metric1="safe",
            metric2="help",
            show_all=False,
        )

        assert "Qwen3-8B" in report
        assert "Qwen-8B_int4_most" in report


class TestMetricToEvalTypeMapping:
    """Test that metric names correctly map to eval types."""

    def test_all_metrics_have_eval_types(self):
        """Test that all valid metrics have corresponding eval types."""
        for metric in VALID_METRICS:
            eval_type = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE.get(metric)
            assert eval_type is not None, f"Metric {metric} has no eval type mapping"
            assert eval_type.startswith('agent_'), f"Eval type {eval_type} should start with 'agent_'"

    def test_metric_to_eval_type_values(self):
        """Test specific metric to eval type mappings."""
        assert ToolEmuFilePaths.METRIC_TO_EVAL_TYPE['safe'] == 'agent_safe'
        assert ToolEmuFilePaths.METRIC_TO_EVAL_TYPE['help'] == 'agent_help'
        assert ToolEmuFilePaths.METRIC_TO_EVAL_TYPE['help_ignore_safety'] == 'agent_help_ignore_safety'


class TestSameMetricEdgeCase:
    """Test edge cases when the same metric is provided for both metric1 and metric2."""

    def test_format_report_same_metric_both_columns(self):
        """Test report formatting when same metric used for both columns."""
        report = format_comparison_report(
            metric1_diffs=[(0, 1.0, 2.0, 1.0), (1, 2.0, 1.0, -1.0)],
            metric2_diffs=[(0, 1.0, 2.0, 1.0), (1, 2.0, 1.0, -1.0)],  # Same diffs
            source_model="Base",
            finetuned_model="FT",
            metric1_mean_delta=0.0,
            metric1_total_cases=2,
            metric2_mean_delta=0.0,
            metric2_total_cases=2,
            metric1="safe",
            metric2="safe",  # Same metric
            show_all=False,
        )

        # Report should still be valid
        assert "SOURCE vs FINETUNED MODEL COMPARISON" in report
        # Both columns should show "safe"
        assert "Metrics:         safe, safe" in report
        # Summary should show safe twice
        lines = report.split('\n')
        safe_lines = [l for l in lines if l.strip().startswith('safe')]
        assert len(safe_lines) == 2  # Two separate summary sections

    def test_format_report_same_metric_identical_data(self):
        """Test that identical diffs for same metric don't cause issues."""
        diffs = [(0, 1.0, 3.0, 2.0)]
        report = format_comparison_report(
            metric1_diffs=diffs,
            metric2_diffs=diffs,
            source_model="Base",
            finetuned_model="FT",
            metric1_mean_delta=2.0,
            metric1_total_cases=1,
            metric2_mean_delta=2.0,
            metric2_total_cases=1,
            metric1="help",
            metric2="help",
            show_all=False,
        )

        # Should show the case once in detailed comparison (union of case_ids)
        assert "DETAILED COMPARISON" in report
        # The case should appear with both columns showing same data
        assert "1.0" in report  # base score
        assert "3.0" in report  # ft score

    def test_format_report_same_metric_empty_diffs(self):
        """Test same metric with no differences."""
        report = format_comparison_report(
            metric1_diffs=[],
            metric2_diffs=[],
            source_model="Base",
            finetuned_model="FT",
            metric1_mean_delta=0.0,
            metric1_total_cases=10,
            metric2_mean_delta=0.0,
            metric2_total_cases=10,
            metric1="safe",
            metric2="safe",
            show_all=False,
        )

        assert "No score differences found" in report
        assert "Metrics:         safe, safe" in report

    def test_compare_scores_same_metric_name(self):
        """Test compare_scores works correctly regardless of metric name."""
        base_scores = {0: 1.0, 1: 2.0}
        ft_scores = {0: 2.0, 1: 3.0}

        # Same function called twice with same metric name should give identical results
        diffs1 = compare_scores(base_scores, ft_scores, "safe", min_diff=1)
        diffs2 = compare_scores(base_scores, ft_scores, "safe", min_diff=1)

        assert diffs1 == diffs2

    def test_compute_mean_delta_same_data_twice(self):
        """Test that computing mean delta twice on same data gives same result."""
        base_scores = {0: 1.0, 1: 2.0, 2: 3.0}
        ft_scores = {0: 2.0, 1: 2.0, 2: 4.0}

        delta1, cases1 = compute_mean_delta(base_scores, ft_scores)
        delta2, cases2 = compute_mean_delta(base_scores, ft_scores)

        assert delta1 == delta2
        assert cases1 == cases2

    def test_format_report_same_long_metric_name(self):
        """Test same metric with long name (tests truncation logic)."""
        report = format_comparison_report(
            metric1_diffs=[(0, 1.0, 2.0, 1.0)],
            metric2_diffs=[(0, 1.0, 2.0, 1.0)],
            source_model="Base",
            finetuned_model="FT",
            metric1_mean_delta=1.0,
            metric1_total_cases=1,
            metric2_mean_delta=1.0,
            metric2_total_cases=1,
            metric1="help_ignore_safety",
            metric2="help_ignore_safety",
            show_all=False,
        )

        # Report should handle long metric names (truncated in table headers)
        assert "help_ignore_safety" in report
        # Table headers should be truncated to 10 chars
        assert "help_ignor" in report  # Truncated version in table

    def test_same_metric_change_indicators(self):
        """Test that change indicators work correctly with same metric."""
        report = format_comparison_report(
            metric1_diffs=[(0, 1.0, 2.0, 1.0)],  # Improved
            metric2_diffs=[(0, 1.0, 2.0, 1.0)],  # Same improvement
            source_model="Base",
            finetuned_model="FT",
            metric1_mean_delta=1.0,
            metric1_total_cases=1,
            metric2_mean_delta=1.0,
            metric2_total_cases=1,
            metric1="safe",
            metric2="safe",
            show_all=False,
        )

        # Change column should show +safe twice (or combined)
        # The exact format depends on implementation, but should contain +safe
        assert "+safe" in report
