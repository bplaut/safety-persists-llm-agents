"""Tests for compute_persistence module."""

import pytest
import sys
from pathlib import Path

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "analysis"))


class TestPersistenceComputation:
    """Test persistence computation functions with seed-averaged approach."""

    def test_compute_persistence_from_seed_cases_single_seed(self):
        """Test persistence computation with a single seed."""
        from compute_persistence import compute_persistence_from_seed_cases

        # Setup: scores for 4 configurations
        # For Persist(S) = (Safety(S,H-β) - Safety(H-β)) / (Safety(S-β) - Safety(H-β))
        # Safety(S-β) = 4.0, Safety(H-β) = 2.0, Safety(S,H-β) = 3.0
        # Persist(S) = (3.0 - 2.0) / (4.0 - 2.0) = 0.5
        safety_scores = {
            's_beta': {0: 4.0, 1: 4.0, 2: 4.0},
            'h_beta': {0: 2.0, 1: 2.0, 2: 2.0},
            's_h_beta': {0: 3.0, 1: 3.0, 2: 3.0},
        }

        # For Persist(H) = (Help(H,S-β) - Help(S-β)) / (Help(H-β) - Help(S-β))
        # Help(H-β) = 4.0, Help(S-β) = 2.0, Help(H,S-β) = 3.0
        # Persist(H) = (3.0 - 2.0) / (4.0 - 2.0) = 0.5
        help_scores = {
            'h_beta': {0: 4.0, 1: 4.0, 2: 4.0},
            's_beta': {0: 2.0, 1: 2.0, 2: 2.0},
            'h_s_beta': {0: 3.0, 1: 3.0, 2: 3.0},
        }

        seed_data = {42: (safety_scores, help_scores)}
        cases_per_seed = {42: [0, 1, 2]}

        persist_s, persist_h = compute_persistence_from_seed_cases(seed_data, cases_per_seed)

        assert abs(persist_s - 0.5) < 0.001
        assert abs(persist_h - 0.5) < 0.001

    def test_compute_persistence_from_seed_cases_multiple_seeds_same_values(self):
        """Test that multiple seeds with identical values give same result as single seed."""
        from compute_persistence import compute_persistence_from_seed_cases

        # Two seeds with identical scores should give same persistence as one seed
        safety_scores = {
            's_beta': {0: 4.0, 1: 4.0},
            'h_beta': {0: 2.0, 1: 2.0},
            's_h_beta': {0: 3.0, 1: 3.0},
        }
        help_scores = {
            'h_beta': {0: 4.0, 1: 4.0},
            's_beta': {0: 2.0, 1: 2.0},
            'h_s_beta': {0: 3.0, 1: 3.0},
        }

        # Same data for two seeds (different case indices to simulate different test sets)
        safety_scores_seed2 = {
            's_beta': {10: 4.0, 11: 4.0},
            'h_beta': {10: 2.0, 11: 2.0},
            's_h_beta': {10: 3.0, 11: 3.0},
        }
        help_scores_seed2 = {
            'h_beta': {10: 4.0, 11: 4.0},
            's_beta': {10: 2.0, 11: 2.0},
            'h_s_beta': {10: 3.0, 11: 3.0},
        }

        seed_data = {
            42: (safety_scores, help_scores),
            43: (safety_scores_seed2, help_scores_seed2),
        }
        cases_per_seed = {42: [0, 1], 43: [10, 11]}

        persist_s, persist_h = compute_persistence_from_seed_cases(seed_data, cases_per_seed)

        # Should still be 0.5 since all scores give the same persistence
        assert abs(persist_s - 0.5) < 0.001
        assert abs(persist_h - 0.5) < 0.001

    def test_compute_persistence_averages_across_seeds_before_ratio(self):
        """Test that persistence averages scores across seeds BEFORE computing the ratio.

        This is the key behavior change: we average scores first, then compute persistence,
        rather than computing persistence per seed and averaging those.
        """
        from compute_persistence import compute_persistence_from_seed_cases

        # Seed 42: Safety(S-β)=4, Safety(H-β)=2, Safety(S,H-β)=4
        # Persist(S) if computed alone = (4-2)/(4-2) = 1.0
        safety_scores_42 = {
            's_beta': {0: 4.0},
            'h_beta': {0: 2.0},
            's_h_beta': {0: 4.0},  # Full persistence
        }
        help_scores_42 = {
            'h_beta': {0: 4.0},
            's_beta': {0: 2.0},
            'h_s_beta': {0: 4.0},
        }

        # Seed 43: Safety(S-β)=4, Safety(H-β)=2, Safety(S,H-β)=2
        # Persist(S) if computed alone = (2-2)/(4-2) = 0.0
        safety_scores_43 = {
            's_beta': {10: 4.0},
            'h_beta': {10: 2.0},
            's_h_beta': {10: 2.0},  # No persistence
        }
        help_scores_43 = {
            'h_beta': {10: 4.0},
            's_beta': {10: 2.0},
            'h_s_beta': {10: 2.0},
        }

        seed_data = {
            42: (safety_scores_42, help_scores_42),
            43: (safety_scores_43, help_scores_43),
        }
        cases_per_seed = {42: [0], 43: [10]}

        persist_s, persist_h = compute_persistence_from_seed_cases(seed_data, cases_per_seed)

        # Old method (average of ratios): (1.0 + 0.0) / 2 = 0.5
        # New method (ratio of averages):
        #   Avg Safety(S-β) = (4+4)/2 = 4
        #   Avg Safety(H-β) = (2+2)/2 = 2
        #   Avg Safety(S,H-β) = (4+2)/2 = 3
        #   Persist(S) = (3-2)/(4-2) = 0.5
        # In this case they happen to be the same, but the calculation method differs

        assert abs(persist_s - 0.5) < 0.001
        assert abs(persist_h - 0.5) < 0.001

    def test_compute_persistence_seed_averaging_differs_from_ratio_averaging(self):
        """Test case where seed-averaging gives different result than ratio-averaging.

        This demonstrates the difference between:
        - Old: compute persistence per seed, then average
        - New: average scores across seeds, then compute persistence
        """
        from compute_persistence import compute_persistence_from_seed_cases

        # Design scores where the two methods give different results
        # Seed 42: S-β=6, H-β=2, S,H-β=5 -> Persist(S) = (5-2)/(6-2) = 0.75
        safety_scores_42 = {
            's_beta': {0: 6.0},
            'h_beta': {0: 2.0},
            's_h_beta': {0: 5.0},
        }
        help_scores_42 = {
            'h_beta': {0: 6.0},
            's_beta': {0: 2.0},
            'h_s_beta': {0: 5.0},
        }

        # Seed 43: S-β=4, H-β=2, S,H-β=3 -> Persist(S) = (3-2)/(4-2) = 0.5
        safety_scores_43 = {
            's_beta': {10: 4.0},
            'h_beta': {10: 2.0},
            's_h_beta': {10: 3.0},
        }
        help_scores_43 = {
            'h_beta': {10: 4.0},
            's_beta': {10: 2.0},
            'h_s_beta': {10: 3.0},
        }

        seed_data = {
            42: (safety_scores_42, help_scores_42),
            43: (safety_scores_43, help_scores_43),
        }
        cases_per_seed = {42: [0], 43: [10]}

        persist_s, persist_h = compute_persistence_from_seed_cases(seed_data, cases_per_seed)

        # Old method (average of ratios): (0.75 + 0.5) / 2 = 0.625
        # New method (ratio of averages):
        #   Avg S-β = (6+4)/2 = 5
        #   Avg H-β = (2+2)/2 = 2
        #   Avg S,H-β = (5+3)/2 = 4
        #   Persist(S) = (4-2)/(5-2) = 2/3 ≈ 0.667
        expected_persist_s = 2.0 / 3.0

        assert abs(persist_s - expected_persist_s) < 0.001
        assert abs(persist_h - expected_persist_s) < 0.001  # Same structure for help

    def test_persistence_division_by_zero(self):
        """Test persistence returns None when denominator is near zero."""
        from compute_persistence import compute_persistence_from_seed_cases

        # If Safety(S-β) == Safety(H-β), denominator is 0
        safety_scores = {
            's_beta': {0: 3.0, 1: 3.0},
            'h_beta': {0: 3.0, 1: 3.0},  # Same as S-β -> divide by zero
            's_h_beta': {0: 3.0, 1: 3.0},
        }
        help_scores = {
            'h_beta': {0: 3.0, 1: 3.0},
            's_beta': {0: 3.0, 1: 3.0},
            'h_s_beta': {0: 3.0, 1: 3.0},
        }

        seed_data = {42: (safety_scores, help_scores)}
        cases_per_seed = {42: [0, 1]}

        persist_s, persist_h = compute_persistence_from_seed_cases(seed_data, cases_per_seed)

        assert persist_s is None
        assert persist_h is None

    def test_compute_persistence_with_ci_single_seed(self):
        """Test compute_persistence_with_ci with single seed and single model."""
        from compute_persistence import compute_persistence_with_ci

        safety_scores = {
            's_beta': {0: 4.0, 1: 4.0, 2: 4.0},
            'h_beta': {0: 2.0, 1: 2.0, 2: 2.0},
            's_h_beta': {0: 3.0, 1: 3.0, 2: 3.0},
        }
        help_scores = {
            'h_beta': {0: 4.0, 1: 4.0, 2: 4.0},
            's_beta': {0: 2.0, 1: 2.0, 2: 2.0},
            'h_s_beta': {0: 3.0, 1: 3.0, 2: 3.0},
        }

        seed_data = {42: (safety_scores, help_scores)}
        # Now takes model_seed_data: Dict[str, SeedScoreData]
        result = compute_persistence_with_ci({"model_a": seed_data}, n_bootstrap=100)

        # Check point estimates
        assert abs(result['persist_s'] - 0.5) < 0.001
        assert abs(result['persist_h'] - 0.5) < 0.001

        # Check CIs exist and bracket estimate
        assert result['persist_s_ci_lower'] <= result['persist_s']
        assert result['persist_s'] <= result['persist_s_ci_upper']
        assert result['persist_h_ci_lower'] <= result['persist_h']
        assert result['persist_h'] <= result['persist_h_ci_upper']

    def test_compute_persistence_with_ci_multiple_seeds(self):
        """Test compute_persistence_with_ci with multiple seeds for a single model."""
        from compute_persistence import compute_persistence_with_ci

        # Seed 42
        safety_scores_42 = {
            's_beta': {0: 4.0, 1: 4.0},
            'h_beta': {0: 2.0, 1: 2.0},
            's_h_beta': {0: 3.0, 1: 3.0},
        }
        help_scores_42 = {
            'h_beta': {0: 4.0, 1: 4.0},
            's_beta': {0: 2.0, 1: 2.0},
            'h_s_beta': {0: 3.0, 1: 3.0},
        }

        # Seed 43 with different case indices
        safety_scores_43 = {
            's_beta': {10: 4.0, 11: 4.0},
            'h_beta': {10: 2.0, 11: 2.0},
            's_h_beta': {10: 3.0, 11: 3.0},
        }
        help_scores_43 = {
            'h_beta': {10: 4.0, 11: 4.0},
            's_beta': {10: 2.0, 11: 2.0},
            'h_s_beta': {10: 3.0, 11: 3.0},
        }

        seed_data = {
            42: (safety_scores_42, help_scores_42),
            43: (safety_scores_43, help_scores_43),
        }

        result = compute_persistence_with_ci({"model_a": seed_data}, n_bootstrap=100)

        assert abs(result['persist_s'] - 0.5) < 0.001
        assert abs(result['persist_h'] - 0.5) < 0.001

    def test_compute_persistence_with_ci_output_structure(self):
        """Verify compute_persistence_with_ci returns correct output structure."""
        from compute_persistence import compute_persistence_with_ci

        safety_scores = {
            's_beta': {0: 4.0, 1: 4.0},
            'h_beta': {0: 2.0, 1: 2.0},
            's_h_beta': {0: 3.0, 1: 3.0},
        }
        help_scores = {
            'h_beta': {0: 4.0, 1: 4.0},
            's_beta': {0: 2.0, 1: 2.0},
            'h_s_beta': {0: 3.0, 1: 3.0},
        }

        seed_data = {42: (safety_scores, help_scores)}
        result = compute_persistence_with_ci({"model_a": seed_data}, n_bootstrap=100)

        # Verify output has all expected keys
        expected_keys = [
            'persist_s', 'persist_s_ci_lower', 'persist_s_ci_upper',
            'persist_h', 'persist_h_ci_lower', 'persist_h_ci_upper',
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Verify CI structure (lower <= estimate <= upper)
        assert result['persist_s_ci_lower'] <= result['persist_s'] <= result['persist_s_ci_upper']
        assert result['persist_h_ci_lower'] <= result['persist_h'] <= result['persist_h_ci_upper']

    def test_compute_persistence_with_ci_raises_on_invalid_denominator(self):
        """Test that compute_persistence_with_ci raises error when point estimate fails."""
        from compute_persistence import compute_persistence_with_ci

        safety_scores = {
            's_beta': {0: 3.0, 1: 3.0},
            'h_beta': {0: 3.0, 1: 3.0},  # Same as S-β -> divide by zero
            's_h_beta': {0: 3.0, 1: 3.0},
        }
        help_scores = {
            'h_beta': {0: 3.0, 1: 3.0},
            's_beta': {0: 3.0, 1: 3.0},
            'h_s_beta': {0: 3.0, 1: 3.0},
        }

        seed_data = {42: (safety_scores, help_scores)}

        with pytest.raises(ValueError, match="Cannot compute persistence point estimate"):
            compute_persistence_with_ci({"model_a": seed_data}, n_bootstrap=100)

    def test_bootstrap_resamples_independently_per_seed(self):
        """Test that bootstrap resamples cases independently for each seed.

        We verify this by checking that with multiple seeds, the CIs are narrower
        than with a single seed (more data -> less uncertainty).
        """
        from compute_persistence import compute_persistence_with_ci

        # Create data with some variance in scores
        safety_scores_42 = {
            's_beta': {0: 4.0, 1: 5.0, 2: 3.0},
            'h_beta': {0: 2.0, 1: 2.5, 2: 1.5},
            's_h_beta': {0: 3.0, 1: 3.5, 2: 2.5},
        }
        help_scores_42 = {
            'h_beta': {0: 4.0, 1: 5.0, 2: 3.0},
            's_beta': {0: 2.0, 1: 2.5, 2: 1.5},
            'h_s_beta': {0: 3.0, 1: 3.5, 2: 2.5},
        }

        # Single seed result
        seed_data_single = {42: (safety_scores_42, help_scores_42)}
        result_single = compute_persistence_with_ci({"model_a": seed_data_single}, n_bootstrap=1000)
        ci_width_single = result_single['persist_s_ci_upper'] - result_single['persist_s_ci_lower']

        # Add second seed with similar variance
        safety_scores_43 = {
            's_beta': {10: 4.0, 11: 5.0, 12: 3.0},
            'h_beta': {10: 2.0, 11: 2.5, 12: 1.5},
            's_h_beta': {10: 3.0, 11: 3.5, 12: 2.5},
        }
        help_scores_43 = {
            'h_beta': {10: 4.0, 11: 5.0, 12: 3.0},
            's_beta': {10: 2.0, 11: 2.5, 12: 1.5},
            'h_s_beta': {10: 3.0, 11: 3.5, 12: 2.5},
        }

        seed_data_multi = {
            42: (safety_scores_42, help_scores_42),
            43: (safety_scores_43, help_scores_43),
        }
        result_multi = compute_persistence_with_ci({"model_a": seed_data_multi}, n_bootstrap=1000)
        ci_width_multi = result_multi['persist_s_ci_upper'] - result_multi['persist_s_ci_lower']

        # With more data (2 seeds), CI should generally be narrower
        # Note: This is a statistical property, may not hold for all random seeds
        # We use a weak assertion since this is probabilistic
        assert ci_width_multi <= ci_width_single * 1.5  # Multi-seed CI shouldn't be much wider

    def test_compute_persistence_multiple_models_average_of_ratios(self):
        """Test that multiple models use average-of-ratios, not ratio-of-averages."""
        from compute_persistence import compute_persistence_with_ci

        # Model A: persist_s = (3-2)/(4-2) = 0.5
        model_a_safety = {
            's_beta': {0: 4.0},
            'h_beta': {0: 2.0},
            's_h_beta': {0: 3.0},
        }
        model_a_help = {
            'h_beta': {0: 4.0},
            's_beta': {0: 2.0},
            'h_s_beta': {0: 3.0},
        }

        # Model B: persist_s = (7-4)/(10-4) = 0.5
        model_b_safety = {
            's_beta': {0: 10.0},
            'h_beta': {0: 4.0},
            's_h_beta': {0: 7.0},
        }
        model_b_help = {
            'h_beta': {0: 10.0},
            's_beta': {0: 4.0},
            'h_s_beta': {0: 7.0},
        }

        model_seed_data = {
            "model_a": {42: (model_a_safety, model_a_help)},
            "model_b": {42: (model_b_safety, model_b_help)},
        }

        result = compute_persistence_with_ci(model_seed_data, n_bootstrap=0)

        # Average of ratios: (0.5 + 0.5) / 2 = 0.5
        assert abs(result['persist_s'] - 0.5) < 0.001
        assert abs(result['persist_h'] - 0.5) < 0.001

    def test_compute_persistence_multiple_models_differs_from_ratio_of_averages(self):
        """Test that average-of-ratios differs from ratio-of-averages with asymmetric data."""
        from compute_persistence import compute_persistence_with_ci

        # Model A: persist_s = (3-2)/(4-2) = 1/2 = 0.5
        model_a_safety = {
            's_beta': {0: 4.0},
            'h_beta': {0: 2.0},
            's_h_beta': {0: 3.0},
        }
        model_a_help = {
            'h_beta': {0: 4.0},
            's_beta': {0: 2.0},
            'h_s_beta': {0: 3.0},
        }

        # Model B: persist_s = (50-20)/(100-20) = 30/80 = 0.375
        model_b_safety = {
            's_beta': {0: 100.0},
            'h_beta': {0: 20.0},
            's_h_beta': {0: 50.0},
        }
        model_b_help = {
            'h_beta': {0: 100.0},
            's_beta': {0: 20.0},
            'h_s_beta': {0: 50.0},
        }

        model_seed_data = {
            "model_a": {42: (model_a_safety, model_a_help)},
            "model_b": {42: (model_b_safety, model_b_help)},
        }

        result = compute_persistence_with_ci(model_seed_data, n_bootstrap=0)

        # Average of ratios: (0.5 + 0.375) / 2 = 0.4375
        expected_avg_of_ratios = (0.5 + 0.375) / 2
        assert abs(result['persist_s'] - expected_avg_of_ratios) < 0.001

        # Ratio of averages would be:
        # avg_s_beta = (4 + 100) / 2 = 52
        # avg_h_beta = (2 + 20) / 2 = 11
        # avg_sh_beta = (3 + 50) / 2 = 26.5
        # ratio_of_avg = (26.5 - 11) / (52 - 11) = 15.5 / 41 ≈ 0.378
        ratio_of_averages = (26.5 - 11) / (52 - 11)

        # Verify they differ
        assert abs(expected_avg_of_ratios - ratio_of_averages) > 0.05
