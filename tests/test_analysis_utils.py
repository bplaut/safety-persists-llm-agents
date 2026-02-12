#!/usr/bin/env python3
"""Tests for analysis_utils.py"""

import sys
from pathlib import Path

import pytest

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "analysis"))

from utils.analysis_utils import (
    aggregate_scores,
    compute_deltas_from_aggregated,
    filter_data,
    format_label,
    get_evaluator_short_name,
    get_present_types,
    VALID_GROUP_BY_DIMENSIONS,
)


class TestAggregateScores:
    """Test aggregate_scores() function."""

    def test_aggregate_by_source_model(self):
        """Test aggregation by source_model."""
        data = [
            {'source_model': 'Qwen3-8B', 'helpfulness': 2.0, 'safety': 3.0, 'training_stages': []},
            {'source_model': 'Qwen3-8B', 'helpfulness': 4.0, 'safety': 5.0, 'training_stages': []},
            {'source_model': 'Llama3.1-8B', 'helpfulness': 1.0, 'safety': 2.0, 'training_stages': []},
        ]

        result = aggregate_scores(data, group_by=['source_model'])

        assert len(result) == 2
        qwen = next(r for r in result if r['source_model'] == 'Qwen3-8B')
        llama = next(r for r in result if r['source_model'] == 'Llama3.1-8B')

        assert qwen['helpfulness'] == 3.0  # (2 + 4) / 2
        assert qwen['safety'] == 4.0  # (3 + 5) / 2
        assert qwen['n_models'] == 2

        assert llama['helpfulness'] == 1.0
        assert llama['safety'] == 2.0
        assert llama['n_models'] == 1

    def test_aggregate_by_last_finetune_type(self):
        """Test aggregation by last_finetune_type."""
        data = [
            {'source_model': 'A', 'last_finetune_type': 'help', 'helpfulness': 2.0, 'safety': 3.0, 'training_stages': [('help', 0.1)]},
            {'source_model': 'B', 'last_finetune_type': 'help', 'helpfulness': 4.0, 'safety': 5.0, 'training_stages': [('help', 0.1)]},
            {'source_model': 'A', 'last_finetune_type': 'safe', 'helpfulness': 1.0, 'safety': 4.0, 'training_stages': [('safe', 0.1)]},
        ]

        result = aggregate_scores(data, group_by=['last_finetune_type'])

        assert len(result) == 2
        help_r = next(r for r in result if r['last_finetune_type'] == 'help')
        safe = next(r for r in result if r['last_finetune_type'] == 'safe')

        assert help_r['helpfulness'] == 3.0
        assert help_r['n_models'] == 2
        assert safe['helpfulness'] == 1.0
        assert safe['n_models'] == 1

    def test_aggregate_empty_data(self):
        """Test aggregation with empty data."""
        result = aggregate_scores([], group_by=['source_model'])
        assert result == []

    def test_aggregate_invalid_dimension_raises(self):
        """Test that invalid dimension raises ValueError."""
        data = [{'source_model': 'A', 'helpfulness': 1.0, 'safety': 2.0}]

        with pytest.raises(ValueError, match="Invalid grouping dimension"):
            aggregate_scores(data, group_by=['invalid_dim'])

    def test_aggregate_computes_std(self):
        """Test that std is computed for aggregated fields."""
        data = [
            {'source_model': 'A', 'helpfulness': 1.0, 'safety': 2.0, 'training_stages': []},
            {'source_model': 'A', 'helpfulness': 3.0, 'safety': 4.0, 'training_stages': []},
        ]

        result = aggregate_scores(data, group_by=['source_model'])

        assert len(result) == 1
        assert result[0]['helpfulness'] == 2.0
        # statistics.stdev uses sample std (n-1 denominator): sqrt(2) ≈ 1.414
        assert abs(result[0]['helpfulness_std'] - 1.4142135623730951) < 0.001


class TestComputeDeltasFromAggregated:
    """Test compute_deltas_from_aggregated() function."""

    def test_deltas_relative_to_base(self):
        """Test delta computation relative to base model."""
        aggregated = [
            # Base model (no training stages)
            {'source_model': 'Qwen3-8B', 'training_stages': [], 'helpfulness': 2.0, 'safety': 3.0},
            # Finetuned model
            {'source_model': 'Qwen3-8B', 'training_stages': [('help', 0.1)], 'helpfulness': 3.0, 'safety': 2.5},
        ]

        deltas = compute_deltas_from_aggregated(
            aggregated,
            group_by=['source_model', 'training_stages'],
            relative_to='base'
        )

        assert len(deltas) == 1
        assert deltas[0]['helpfulness_delta'] == 1.0  # 3.0 - 2.0
        assert deltas[0]['safety_delta'] == -0.5  # 2.5 - 3.0

    def test_deltas_relative_to_parent(self):
        """Test delta computation relative to parent (previous stage)."""
        aggregated = [
            # Base model
            {'source_model': 'A', 'training_stages': [], 'helpfulness': 1.0, 'safety': 4.0},
            # First finetune
            {'source_model': 'A', 'training_stages': [('help', 0.1)], 'helpfulness': 2.0, 'safety': 3.0},
            # Second finetune
            {'source_model': 'A', 'training_stages': [('help', 0.1), ('safe', 0.1)], 'helpfulness': 1.5, 'safety': 3.5},
        ]

        deltas = compute_deltas_from_aggregated(
            aggregated,
            group_by=['source_model', 'training_stages'],
            relative_to='parent'
        )

        # Should have 2 deltas: first relative to base, second relative to first
        assert len(deltas) == 2

        # First finetune delta (relative to base)
        first = next(d for d in deltas if len(d['training_stages']) == 1)
        assert first['helpfulness_delta'] == 1.0  # 2.0 - 1.0
        assert first['safety_delta'] == -1.0  # 3.0 - 4.0

        # Second finetune delta (relative to first)
        second = next(d for d in deltas if len(d['training_stages']) == 2)
        assert second['helpfulness_delta'] == -0.5  # 1.5 - 2.0
        assert second['safety_delta'] == 0.5  # 3.5 - 3.0

    def test_deltas_empty_data(self):
        """Test delta computation with empty data."""
        result = compute_deltas_from_aggregated([], group_by=['source_model'])
        assert result == []


class TestFilterData:
    """Test filter_data() function."""

    def test_filter_by_model(self):
        """Test filtering by source model."""
        data = [
            {'source_model': 'Qwen3-8B', 'training_stages': []},
            {'source_model': 'Llama3.1-8B', 'training_stages': []},
            {'source_model': 'Qwen2.5-7B', 'training_stages': []},
        ]

        result = filter_data(data, model='Qwen3-8B')

        assert len(result) == 1
        assert result[0]['source_model'] == 'Qwen3-8B'

    def test_filter_by_first_finetune(self):
        """Test filtering by first finetune type."""
        data = [
            {'source_model': 'A', 'training_stages': [('help', 0.1)]},
            {'source_model': 'B', 'training_stages': [('safe', 0.1)]},
            {'source_model': 'C', 'training_stages': [('help', 0.1), ('safe', 0.1)]},
        ]

        result = filter_data(data, first_finetune='help')

        assert len(result) == 2
        assert all(r['training_stages'][0][0] == 'help' for r in result)

    def test_filter_by_last_finetune(self):
        """Test filtering by last finetune type."""
        data = [
            {'source_model': 'A', 'training_stages': [('help', 0.1)]},
            {'source_model': 'B', 'training_stages': [('safe', 0.1)]},
            {'source_model': 'C', 'training_stages': [('help', 0.1), ('safe', 0.1)]},
        ]

        result = filter_data(data, last_finetune='safe')

        assert len(result) == 2
        assert all(r['training_stages'][-1][0] == 'safe' for r in result)

    def test_filter_disallowed_num_stages(self):
        """Test filtering by disallowed number of stages."""
        data = [
            {'source_model': 'A', 'training_stages': []},  # 0 stages
            {'source_model': 'B', 'training_stages': [('help', 0.1)]},  # 1 stage
            {'source_model': 'C', 'training_stages': [('help', 0.1), ('safe', 0.1)]},  # 2 stages
        ]

        result = filter_data(data, disallowed_num_stages=[0, 2])

        assert len(result) == 1
        assert len(result[0]['training_stages']) == 1

    def test_filter_disallowed_betas(self):
        """Test filtering by disallowed beta values."""
        data = [
            {'source_model': 'A', 'training_stages': []},  # Source model - never excluded
            {'source_model': 'B', 'training_stages': [('help', 0.1)]},
            {'source_model': 'C', 'training_stages': [('help', 0.05)]},
            {'source_model': 'D', 'training_stages': [('help', 0.1), ('safe', 0.05)]},
        ]

        result = filter_data(data, disallowed_betas=[0.1])

        assert len(result) == 2
        # Source model (A) and model with only 0.05 beta (C)
        assert any(r['source_model'] == 'A' for r in result)
        assert any(r['source_model'] == 'C' for r in result)

    def test_filter_include_intermediates(self):
        """Test include_intermediates flag."""
        data = [
            {'source_model': 'A', 'training_stages': []},
            {'source_model': 'A', 'training_stages': [('help', 0.1)]},
            {'source_model': 'A', 'training_stages': [('help', 0.1), ('safe', 0.1)]},
        ]

        # Without intermediates: source model + final safe-finetuned
        # (source models pass through last_finetune filter since they have no stages)
        result_no_int = filter_data(data, last_finetune='safe', include_intermediates=False)
        assert len(result_no_int) == 2

        # With intermediates: adds the most-finetuned ancestor
        result_with_int = filter_data(data, last_finetune='safe', include_intermediates=True)
        assert len(result_with_int) == 3


class TestFormatLabel:
    """Test format_label() function."""

    def test_single_stage_label(self):
        """Test label for single training stage."""
        data = {'training_stages': [('help', 0.1)]}
        assert format_label(data) == 'H-0.1'

        data = {'training_stages': [('safe', 0.05)]}
        assert format_label(data) == 'S-0.05'

    def test_multi_stage_same_beta(self):
        """Test label for multiple stages with same beta."""
        data = {'training_stages': [('safe', 0.1), ('help', 0.1)]}
        assert format_label(data) == 'S,H-0.1'

    def test_multi_stage_different_betas(self):
        """Test label for multiple stages with different betas."""
        data = {'training_stages': [('safe', 0.05), ('help', 0.1)]}
        assert format_label(data) == 'S-0.05,H-0.1'

    def test_source_model_raises(self):
        """Test that source model (no stages) raises ValueError."""
        data = {'training_stages': []}

        with pytest.raises(ValueError, match="should not be called for source models"):
            format_label(data)


class TestGetEvaluatorShortName:
    """Test get_evaluator_short_name() function."""

    def test_gpt5_mini(self):
        assert get_evaluator_short_name('gpt-5-mini') == 'eval-gpt5m'

    def test_qwen_32b(self):
        assert get_evaluator_short_name('Qwen/Qwen3-32B') == 'eval-q32'
        assert get_evaluator_short_name('Qwen3-32B') == 'eval-q32'

    def test_other_gpt(self):
        result = get_evaluator_short_name('gpt-4o')
        assert result.startswith('eval-gpt')


class TestBootstrapCI:
    """Test bootstrap_ci() function for computing confidence intervals."""

    def test_bootstrap_ci_basic(self):
        """Bootstrap CI on simple mean computation."""
        from utils.analysis_utils import bootstrap_ci

        # Simple case: mean of [1, 2, 3, 4, 5] with known distribution
        case_indices = list(range(5))
        scores = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}

        def compute_mean(indices):
            return sum(scores[i] for i in indices) / len(indices)

        result = bootstrap_ci(compute_mean, case_indices, n_bootstrap=1000, confidence_level=0.95)

        # Check result structure
        assert 'estimate' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'std_error' in result

        # Point estimate should be the mean = 3.0
        assert result['estimate'] == 3.0

        # CI should bracket the estimate
        assert result['ci_lower'] <= result['estimate']
        assert result['estimate'] <= result['ci_upper']

        # CI should be reasonable (for small sample, CI won't be super tight)
        assert result['ci_lower'] >= 1.0  # Can't be lower than min value
        assert result['ci_upper'] <= 5.0  # Can't be higher than max value

    def test_bootstrap_ci_raises_on_none(self):
        """Should raise ValueError if compute_fn returns None."""
        from utils.analysis_utils import bootstrap_ci

        case_indices = [0, 1, 2]

        def compute_returns_none(indices):
            return None

        with pytest.raises(ValueError, match="returned None"):
            bootstrap_ci(compute_returns_none, case_indices, n_bootstrap=100)

    def test_bootstrap_ci_raises_on_nan(self):
        """Should raise ValueError if compute_fn returns NaN."""
        from utils.analysis_utils import bootstrap_ci

        case_indices = [0, 1, 2]

        def compute_returns_nan(indices):
            return float('nan')

        with pytest.raises(ValueError, match="returned NaN"):
            bootstrap_ci(compute_returns_nan, case_indices, n_bootstrap=100)

    def test_bootstrap_ci_reproducible(self):
        """Same seed should give same results."""
        from utils.analysis_utils import bootstrap_ci

        case_indices = list(range(10))
        scores = {i: float(i) for i in range(10)}

        def compute_mean(indices):
            return sum(scores[i] for i in indices) / len(indices)

        # Run twice with same seed (uses DEFAULT_RANDOM_SEED internally)
        result1 = bootstrap_ci(compute_mean, case_indices, n_bootstrap=500)
        result2 = bootstrap_ci(compute_mean, case_indices, n_bootstrap=500)

        # Results should be identical
        assert result1['estimate'] == result2['estimate']
        assert result1['ci_lower'] == result2['ci_lower']
        assert result1['ci_upper'] == result2['ci_upper']
        assert result1['std_error'] == result2['std_error']

    def test_bootstrap_ci_constant_data(self):
        """Bootstrap CI with constant data should have zero-width CI."""
        from utils.analysis_utils import bootstrap_ci

        case_indices = [0, 1, 2, 3, 4]
        scores = {i: 5.0 for i in range(5)}  # All same value

        def compute_mean(indices):
            return sum(scores[i] for i in indices) / len(indices)

        result = bootstrap_ci(compute_mean, case_indices, n_bootstrap=100)

        assert result['estimate'] == 5.0
        assert result['ci_lower'] == 5.0
        assert result['ci_upper'] == 5.0
        assert result['std_error'] == 0.0


class TestGetPresentTypes:
    """Test get_present_types() function."""

    def test_identifies_source_models(self):
        """Test identification of source models."""
        data = [
            {'source_model': 'Qwen3-8B', 'last_finetune_type': None},
            {'source_model': 'Llama3.1-8B', 'last_finetune_type': 'help'},
        ]

        result = get_present_types(data)

        assert 'Qwen3-8B' in result['source_models']
        assert 'Llama3.1-8B' in result['source_models']
        assert result['has_source_model_points'] is True

    def test_identifies_finetune_types(self):
        """Test identification of finetune types."""
        data = [
            {'source_model': 'A', 'last_finetune_type': 'help'},
            {'source_model': 'B', 'last_finetune_type': 'safe'},
            {'source_model': 'C', 'last_finetune_type': 'both'},
        ]

        result = get_present_types(data)

        assert 'help' in result['finetune_types']
        assert 'safe' in result['finetune_types']
        assert 'combined' in result['finetune_types']

    def test_no_source_model_points(self):
        """Test when no source model points exist."""
        data = [
            {'source_model': 'A', 'last_finetune_type': 'help'},
        ]

        result = get_present_types(data)

        assert result['has_source_model_points'] is False
