#!/usr/bin/env python3
"""
Unit tests for scripts/analyze_metric_correlation.py

Tests for the refactored helper functions.
"""

import os
import sys
import pytest

# Add src/analysis to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis'))

from analyze_metric_correlation import (
    parse_arguments,
    compute_correlations,
    interpret_correlation,
    print_metric_stats,
    print_contingency_table,
    build_results_dict,
    METRIC_LABELS,
    SCORE_LABELS,
)


# =============================================================================
# Test Parse Arguments
# =============================================================================

class TestParseArguments:
    """Tests for parse_arguments function."""

    def test_default_values(self, monkeypatch):
        """Test default argument values."""
        monkeypatch.setattr(sys, 'argv', ['analyze_metric_correlation.py'])
        args = parse_arguments()

        assert args.metric1 == 'safe'
        assert args.metric2 == 'help_ignore_safety'
        assert args.data_dir == 'data/unprocessed_dpo_data'
        assert args.output is None

    def test_custom_metrics(self, monkeypatch):
        """Test custom metric selection."""
        monkeypatch.setattr(sys, 'argv', [
            'analyze_metric_correlation.py',
            '--metric1', 'help',
            '--metric2', 'help_ignore_safety'
        ])
        args = parse_arguments()

        assert args.metric1 == 'help'
        assert args.metric2 == 'help_ignore_safety'

    def test_short_flags(self, monkeypatch):
        """Test short flag versions."""
        monkeypatch.setattr(sys, 'argv', [
            'analyze_metric_correlation.py',
            '-m1', 'safe',
            '-m2', 'help',
            '-d', '/custom/path',
            '-o', 'output.json'
        ])
        args = parse_arguments()

        assert args.metric1 == 'safe'
        assert args.metric2 == 'help'
        assert args.data_dir == '/custom/path'
        assert args.output == 'output.json'

    def test_output_path(self, monkeypatch):
        """Test output path argument."""
        monkeypatch.setattr(sys, 'argv', [
            'analyze_metric_correlation.py',
            '--output', '/tmp/results.json'
        ])
        args = parse_arguments()

        assert args.output == '/tmp/results.json'


# =============================================================================
# Test Compute Correlations
# =============================================================================

class TestComputeCorrelations:
    """Tests for compute_correlations function."""

    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation."""
        scores1 = (1, 2, 3, 4, 5)
        scores2 = (1, 2, 3, 4, 5)

        result = compute_correlations(scores1, scores2)

        assert result['pearson_r'] == pytest.approx(1.0)
        assert result['spearman_r'] == pytest.approx(1.0)
        assert result['kendall_tau'] == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation."""
        scores1 = (1, 2, 3, 4, 5)
        scores2 = (5, 4, 3, 2, 1)

        result = compute_correlations(scores1, scores2)

        assert result['pearson_r'] == pytest.approx(-1.0)
        assert result['spearman_r'] == pytest.approx(-1.0)
        assert result['kendall_tau'] == pytest.approx(-1.0)

    def test_moderate_correlation(self):
        """Test data with moderate correlation."""
        scores1 = (1, 2, 3, 4, 5, 6)
        scores2 = (2, 1, 4, 3, 6, 5)

        result = compute_correlations(scores1, scores2)

        # Should have moderate positive correlation
        assert 0.5 < result['pearson_r'] < 1.0

    def test_returns_all_keys(self):
        """Test that all expected keys are returned."""
        scores1 = (1, 2, 3)
        scores2 = (2, 3, 4)

        result = compute_correlations(scores1, scores2)

        assert 'pearson_r' in result
        assert 'pearson_p' in result
        assert 'spearman_r' in result
        assert 'spearman_p' in result
        assert 'kendall_tau' in result
        assert 'kendall_p' in result


# =============================================================================
# Test Interpret Correlation
# =============================================================================

class TestInterpretCorrelation:
    """Tests for interpret_correlation function."""

    def test_strong_positive(self):
        """Test strong positive correlation interpretation."""
        interpretation, significance = interpret_correlation(0.6, 0.05)

        assert interpretation == 'strong positive'
        assert 'significant' in significance

    def test_very_strong_negative(self):
        """Test very strong negative correlation interpretation."""
        interpretation, significance = interpret_correlation(-0.8, 0.001)

        assert interpretation == 'very strong negative'

    def test_not_significant(self):
        """Test not significant correlation."""
        interpretation, significance = interpret_correlation(0.1, 0.5)

        assert 'not significant' in significance

    def test_boundary_values(self):
        """Test boundary values for interpretation."""
        # Exactly at 0.3 boundary (should be moderate)
        interp, _ = interpret_correlation(0.3, 0.001)
        assert interp == 'moderate positive'

        # Exactly at 0.5 boundary (should be strong)
        interp, _ = interpret_correlation(0.5, 0.001)
        assert interp == 'strong positive'

        # Exactly at 0.7 boundary (should be very strong)
        interp, _ = interpret_correlation(0.7, 0.001)
        assert interp == 'very strong positive'


# =============================================================================
# Test Print Metric Stats
# =============================================================================

class TestPrintMetricStats:
    """Tests for print_metric_stats function."""

    def test_prints_all_stats(self, capsys):
        """Test that all statistics are printed."""
        scores = (1, 2, 2, 3, 3, 3)

        print_metric_stats("Test Metric", scores)

        captured = capsys.readouterr()
        assert 'Test Metric:' in captured.out
        assert 'Mean:' in captured.out
        assert 'Std:' in captured.out
        assert 'Min:' in captured.out
        assert 'Max:' in captured.out
        assert 'Distribution:' in captured.out

    def test_correct_values(self, capsys):
        """Test that printed values are correct."""
        scores = (0, 1, 2, 3)

        print_metric_stats("Scores", scores)

        captured = capsys.readouterr()
        assert 'Mean: 1.500' in captured.out
        assert 'Min:  0' in captured.out
        assert 'Max:  3' in captured.out

    def test_distribution_format(self, capsys):
        """Test distribution format."""
        scores = (1, 1, 2, 3, 3, 3)

        print_metric_stats("Test", scores)

        captured = capsys.readouterr()
        assert '1=2' in captured.out  # score 1 appears 2 times
        assert '2=1' in captured.out  # score 2 appears 1 time
        assert '3=3' in captured.out  # score 3 appears 3 times


# =============================================================================
# Test Print Contingency Table
# =============================================================================

class TestPrintContingencyTable:
    """Tests for print_contingency_table function."""

    def test_prints_header(self, capsys):
        """Test that header is printed."""
        scores1 = (0, 1, 2)
        scores2 = (1, 2, 3)

        print_contingency_table(
            'safe', 'help', 'Safety', 'Helpfulness',
            scores1, scores2
        )

        captured = capsys.readouterr()
        assert 'Contingency Table' in captured.out
        assert 'Safety' in captured.out
        assert 'Helpfulness' in captured.out

    def test_prints_rows(self, capsys):
        """Test that rows are printed for each score value."""
        scores1 = (0, 1, 2, 3)
        scores2 = (0, 1, 2, 3)

        print_contingency_table(
            'safe', 'help', 'Safety', 'Helpfulness',
            scores1, scores2
        )

        captured = capsys.readouterr()
        # Should have rows for scores 0, 1, 2, 3
        assert 'Certain Severe (0)' in captured.out or 'Score 0' in captured.out
        assert 'Total' in captured.out

    def test_counts_are_correct(self, capsys):
        """Test that counts are calculated correctly."""
        # 2 samples with score1=1, one maps to score2=0, one to score2=1
        scores1 = (1, 1, 2)
        scores2 = (0, 1, 2)

        print_contingency_table(
            'safe', 'help', 'Safety', 'Helpfulness',
            scores1, scores2
        )

        captured = capsys.readouterr()
        # Row for score1=1 should show total of 2
        lines = captured.out.split('\n')
        # Find line with score 1 row
        for line in lines:
            if 'Possible Severe (1)' in line or 'Score 1' in line:
                assert '2' in line  # Total should be 2


# =============================================================================
# Test Build Results Dict
# =============================================================================

class TestBuildResultsDict:
    """Tests for build_results_dict function."""

    def test_returns_dict_with_all_keys(self):
        """Test that all expected keys are present."""
        all_scores = [(0, 1, 2), (1, 2, 3)]
        case_indices = (0, 1)
        scores1 = (1, 2)
        scores2 = (2, 3)
        correlations = {
            'pearson_r': 1.0,
            'pearson_p': 0.0,
            'spearman_r': 1.0,
            'spearman_p': 0.0,
            'kendall_tau': 1.0,
            'kendall_p': 0.0,
        }

        result = build_results_dict(
            'safe', 'help', all_scores,
            case_indices, scores1, scores2,
            2, 0,
            correlations, 'very strong positive', 'highly significant'
        )

        assert result['metric1'] == 'safe'
        assert result['metric2'] == 'help'
        assert result['n_samples'] == 2
        assert result['n_unique_cases'] == 2
        assert result['n_files_processed'] == 2
        assert result['n_files_with_errors'] == 0
        assert 'safe' in result
        assert 'help' in result
        assert 'correlation' in result

    def test_metric_stats_structure(self):
        """Test metric statistics structure."""
        all_scores = [(0, 1, 2), (1, 2, 3), (2, 3, 1)]
        case_indices = (0, 1, 2)
        scores1 = (1, 2, 3)
        scores2 = (2, 3, 1)
        correlations = {
            'pearson_r': 0.5,
            'pearson_p': 0.1,
            'spearman_r': 0.5,
            'spearman_p': 0.1,
            'kendall_tau': 0.5,
            'kendall_p': 0.1,
        }

        result = build_results_dict(
            'safe', 'help', all_scores,
            case_indices, scores1, scores2,
            3, 0,
            correlations, 'strong positive', 'not significant'
        )

        # Check metric1 structure
        assert 'mean' in result['safe']
        assert 'std' in result['safe']
        assert 'min' in result['safe']
        assert 'max' in result['safe']
        assert 'distribution' in result['safe']

    def test_correlation_structure(self):
        """Test correlation structure."""
        all_scores = [(0, 1, 2)]
        case_indices = (0,)
        scores1 = (1,)
        scores2 = (2,)
        correlations = {
            'pearson_r': 0.9,
            'pearson_p': 0.001,
            'spearman_r': 0.85,
            'spearman_p': 0.002,
            'kendall_tau': 0.8,
            'kendall_p': 0.003,
        }

        result = build_results_dict(
            'safe', 'help', all_scores,
            case_indices, scores1, scores2,
            1, 0,
            correlations, 'very strong positive', 'highly significant'
        )

        corr = result['correlation']
        assert corr['pearson_r'] == 0.9
        assert corr['pearson_p'] == 0.001
        assert corr['interpretation'] == 'very strong positive'
        assert corr['significance'] == 'highly significant'


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_metric_labels_has_all_metrics(self):
        """Test that METRIC_LABELS has all expected metrics."""
        expected_metrics = ['safe', 'help', 'help_ignore_safety']

        for metric in expected_metrics:
            assert metric in METRIC_LABELS

    def test_score_labels_structure(self):
        """Test that SCORE_LABELS has correct structure."""
        assert 'safe' in SCORE_LABELS
        assert 'help' in SCORE_LABELS

        # Safety labels should have 0-3
        for score in [0, 1, 2, 3]:
            assert score in SCORE_LABELS['safe']

        # Helpfulness labels should have 0-3
        for score in [0, 1, 2, 3]:
            assert score in SCORE_LABELS['help']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
