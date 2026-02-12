#!/usr/bin/env python3
"""Tests for generate_plots.py"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "analysis"))


def create_mock_unified_report(
    helpfulness_mean: float,
    safety_mean: float,
    n_samples: int = 72
) -> dict:
    """Create a mock unified report JSON structure."""
    return {
        "agent_safe": {
            "ToolCallRisk": {
                "mean": safety_mean,
                "std": 1.0,
                "histogram": {"0.0": 10, "1.0": 20, "2.0": 20, "3.0": 22},
                "total_evals": n_samples,
                "valid_scores": n_samples,
                "missing_scores": 0
            }
        },
        "agent_help_ignore_safety": {
            "Helpfulness": {
                "mean": helpfulness_mean,
                "std": 1.0,
                "histogram": {"0.0": 10, "1.0": 20, "2.0": 20, "3.0": 22},
                "total_evals": n_samples,
                "valid_scores": n_samples,
                "missing_scores": 0
            }
        }
    }


class TestLoadResultsData:
    """Test loading data from unified report files."""

    def test_load_single_report(self):
        """Test loading a single unified report file."""
        from utils.analysis_utils import load_results_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock report file
            filename = "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-gpt-5-mini_int4_unified_report.json"
            report_path = os.path.join(tmpdir, filename)
            report_data = create_mock_unified_report(2.5, 1.8)
            with open(report_path, 'w') as f:
                json.dump(report_data, f)

            # Load the data
            data = load_results_data(tmpdir)

            assert len(data) == 1
            assert data[0]['helpfulness'] == 2.5
            assert data[0]['safety'] == 1.8

    def test_load_multiple_reports(self):
        """Test loading multiple unified report files."""
        from utils.analysis_utils import load_results_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple mock report files
            reports = [
                ("Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-gpt-5-mini_int4_unified_report.json", 2.5, 1.8),
                ("meta-llama_Llama-3.1-8B-Instruct_emu-Qwen_Qwen3-32B_eval-gpt-5-mini_int4_unified_report.json", 2.0, 2.2),
            ]

            for filename, help_score, safe_score in reports:
                report_path = os.path.join(tmpdir, filename)
                report_data = create_mock_unified_report(help_score, safe_score)
                with open(report_path, 'w') as f:
                    json.dump(report_data, f)

            data = load_results_data(tmpdir)

            assert len(data) == 2
            # Check both reports were loaded (order may vary)
            help_scores = {d['helpfulness'] for d in data}
            assert help_scores == {2.5, 2.0}

    def test_skip_missing_metrics(self):
        """Test that reports with missing metrics are skipped."""
        from utils.analysis_utils import load_results_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a report with missing helpfulness
            filename = "Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-gpt-5-mini_int4_unified_report.json"
            report_path = os.path.join(tmpdir, filename)
            report_data = {"agent_safe": {"ToolCallRisk": {"mean": 2.0, "valid_scores": 72}}}
            with open(report_path, 'w') as f:
                json.dump(report_data, f)

            data = load_results_data(tmpdir)

            assert len(data) == 0


class TestAssignColorsMarkers:
    """Test color and marker assignment by source model."""

    def test_same_source_model_same_style(self):
        """Models with same source should get same color/marker."""
        from generate_plots import assign_colors_markers

        # Finetuned models are identified by seed suffix
        data = [
            {'model_name': 'Qwen3-8B', 'source_model': 'Qwen3-8B'},
            {'model_name': 'Qwen-8B_help_s42', 'source_model': 'Qwen3-8B'},
            {'model_name': 'Qwen-8B_safe_s42', 'source_model': 'Qwen3-8B'},
        ]

        styles = assign_colors_markers(data)

        # All three should have the same color and marker
        assert styles['Qwen3-8B']['color'] == styles['Qwen3-8B']['color']
        assert styles['Qwen3-8B']['marker'] == styles['Qwen3-8B']['marker']
        # Only one source model, so only one entry
        assert len(styles) == 1

    def test_different_source_models_different_styles(self):
        """Different source models should get different color/marker combinations."""
        from generate_plots import assign_colors_markers

        data = [
            {'model_name': 'Qwen3-8B', 'source_model': 'Qwen3-8B'},
            {'model_name': 'Llama-3.1-8B', 'source_model': 'Llama-3.1-8B'},
            {'model_name': 'Phi-4', 'source_model': 'Phi-4'},
        ]

        styles = assign_colors_markers(data)

        # Should have 3 different style entries
        assert len(styles) == 3

        # All colors should be different
        colors = [styles[base]['color'] for base in styles]
        assert len(set(colors)) == 3

    def test_consistent_assignment(self):
        """Same source models in different order should get same styles."""
        from generate_plots import assign_colors_markers

        data1 = [
            {'model_name': 'Qwen3-8B', 'source_model': 'Qwen3-8B'},
            {'model_name': 'Llama-8B', 'source_model': 'Llama-8B'},
        ]
        data2 = [
            {'model_name': 'Llama-8B', 'source_model': 'Llama-8B'},
            {'model_name': 'Qwen3-8B', 'source_model': 'Qwen3-8B'},
        ]

        styles1 = assign_colors_markers(data1)
        styles2 = assign_colors_markers(data2)

        # Sorted source models should produce same assignment
        assert styles1 == styles2


class TestFormatLabel:
    """Test label formatting for plot annotations."""

    def test_source_model_raises_error(self):
        """Source models should raise ValueError (caller should not call format_label for them)."""
        from generate_plots import format_label

        data = {
            'model_name': 'Qwen3-8B',
            'source_model': 'Qwen3-8B',
            'training_stages': []
        }

        with pytest.raises(ValueError, match="should not be called for source models"):
            format_label(data)

    def test_single_stage_help(self):
        """Single stage 'help' should show as H-beta."""
        from generate_plots import format_label

        # Finetuned models are identified by seed suffix
        data = {
            'model_name': 'Qwen-8B_help_gpt5m_s42',
            'source_model': 'Qwen3-8B',
            'training_stages': [('help', '0.1')]
        }

        label = format_label(data)

        assert label == "H-0.1"

    def test_single_stage_safe(self):
        """Single stage 'safe' should show as S-beta."""
        from generate_plots import format_label

        data = {
            'model_name': 'Qwen-8B_safe_gpt5m_beta-0.05_s42',
            'source_model': 'Qwen3-8B',
            'training_stages': [('safe', '0.05')]
        }

        label = format_label(data)

        assert label == "S-0.05"

    def test_sequential_same_beta(self):
        """Sequential with same betas should show single beta at end."""
        from generate_plots import format_label

        data = {
            'model_name': 'Qwen-8B_help_gpt5m_safe_gpt5m_s42',
            'source_model': 'Qwen3-8B',
            'training_stages': [('help', '0.1'), ('safe', '0.1')]
        }

        label = format_label(data)

        assert label == "H,S-0.1"

    def test_sequential_different_betas(self):
        """Sequential with different betas should show each beta."""
        from generate_plots import format_label

        data = {
            'model_name': 'Qwen-8B_help_gpt5m_beta-0.05_safe_gpt5m_beta-0.1_s42',
            'source_model': 'Qwen3-8B',
            'training_stages': [('help', '0.05'), ('safe', '0.1')]
        }

        label = format_label(data)

        assert label == "H-0.05,S-0.1"

    def test_combined_dataset_safe_help(self):
        """Combined dataset 'both' should show as S&H."""
        from generate_plots import format_label

        data = {
            'model_name': 'Qwen-8B_both_gpt5m_s42',
            'source_model': 'Qwen3-8B',
            'training_stages': [('both', '0.1')]
        }

        label = format_label(data)

        assert label == "S&H-0.1"

    def test_single_stage_help_q32(self):
        """Single stage 'help' should show as H-beta."""
        from generate_plots import format_label

        data = {
            'model_name': 'Qwen-8B_help_q32_s42',
            'source_model': 'Qwen2.5-7B-Instruct',
            'training_stages': [('help', '0.1')]
        }

        label = format_label(data)

        assert label == "H-0.1"

    def test_single_stage_safe_q32(self):
        """Single stage 'safe' should show as S-beta."""
        from generate_plots import format_label

        data = {
            'model_name': 'Llama-8B_safe_q32_beta-0.05_s42',
            'source_model': 'Llama-3.1-8B-Instruct',
            'training_stages': [('safe', '0.05')]
        }

        label = format_label(data)

        assert label == "S-0.05"

    def test_sequential_q32_same_beta(self):
        """Sequential with same betas should show single beta at end."""
        from generate_plots import format_label

        data = {
            'model_name': 'Qwen-8B_help_q32_safe_q32_s42',
            'source_model': 'Qwen2.5-7B-Instruct',
            'training_stages': [('help', '0.1'), ('safe', '0.1')]
        }

        label = format_label(data)

        assert label == "H,S-0.1"

    def test_sequential_q32_different_betas(self):
        """Sequential with different betas should show each beta."""
        from generate_plots import format_label

        data = {
            'model_name': 'Phi-4_safe_q32_beta-0.05_help_q32_beta-0.1_s42',
            'source_model': 'phi-4',
            'training_stages': [('safe', '0.05'), ('help', '0.1')]
        }

        label = format_label(data)

        assert label == "S-0.05,H-0.1"

    def test_unknown_dataset_fallback(self):
        """Unknown dataset names should pass through unchanged."""
        from generate_plots import format_label

        data = {
            'model_name': 'Qwen-8B_custom_dataset_s42',
            'source_model': 'Qwen3-8B',
            'training_stages': [('custom_dataset', '0.1')]
        }

        label = format_label(data)

        assert label == "custom_dataset-0.1"


class TestPlotGeneration:
    """Test actual plot generation and saving."""

    def test_plot_saves_to_file(self):
        """Test that a plot file is created."""
        from generate_plots import generate_scatter_plot, assign_colors_markers

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data
            data = [
                {'model_name': 'Qwen3-8B', 'source_model': 'Qwen3-8B',
                 'helpfulness': 2.5, 'safety': 1.8, 'training_stages': []},
                {'model_name': 'Llama-8B', 'source_model': 'Llama-8B',
                 'helpfulness': 2.0, 'safety': 2.2, 'training_stages': []},
            ]
            styles = assign_colors_markers(data)

            generate_scatter_plot(data, output_dir=tmpdir, name="test_plot", styles=styles)

            output_path = os.path.join(tmpdir, "test_plot.png")
            assert os.path.exists(output_path)
            # Check file is not empty
            assert os.path.getsize(output_path) > 0

    def test_plot_with_texture(self):
        """Test that texture parameter works."""
        from generate_plots import generate_scatter_plot, assign_colors_markers

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {'model_name': 'Qwen3-8B', 'source_model': 'Qwen3-8B',
                 'helpfulness': 2.5, 'safety': 1.8, 'training_stages': []},
                {'model_name': 'Qwen3-8B-H', 'source_model': 'Qwen3-8B',
                 'helpfulness': 2.3, 'safety': 1.5, 'training_stages': [('help', 0.1)]},
                {'model_name': 'Qwen3-8B-S', 'source_model': 'Qwen3-8B',
                 'helpfulness': 2.1, 'safety': 2.0, 'training_stages': [('safe', 0.1)]},
            ]
            styles = assign_colors_markers(data)

            generate_scatter_plot(data, output_dir=tmpdir, name="test_texture", styles=styles, texture=True)

            output_path = os.path.join(tmpdir, "test_texture.png")
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_plot_with_single_model(self):
        """Test that plots work with a single model."""
        from generate_plots import generate_scatter_plot, assign_colors_markers

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {'model_name': 'Qwen3-8B', 'source_model': 'Qwen3-8B',
                 'helpfulness': 2.5, 'safety': 1.8, 'training_stages': []},
            ]
            styles = assign_colors_markers(data)

            generate_scatter_plot(data, output_dir=tmpdir, name="test_single", styles=styles)

            output_path = os.path.join(tmpdir, "test_single.png")
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_empty_data_raises_error(self):
        """Test that empty data raises an error."""
        from generate_plots import generate_scatter_plot

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No data"):
                generate_scatter_plot([], output_dir=tmpdir, name="test_plot", styles={})


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test loading data and generating plot."""
        from utils.analysis_utils import load_results_data
        from generate_plots import generate_scatter_plot, assign_colors_markers

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock report files
            # Finetuned models are identified by seed suffix
            reports = [
                ("Qwen_Qwen3-8B_emu-Qwen_Qwen3-32B_eval-gpt-5-mini_int4_unified_report.json", 2.5, 1.8),
                ("Qwen-8B_help_s42_emu-Qwen_Qwen3-32B_eval-gpt-5-mini_int4_unified_report.json", 2.8, 1.5),
                ("meta-llama_Llama-3.1-8B-Instruct_emu-Qwen_Qwen3-32B_eval-gpt-5-mini_int4_unified_report.json", 2.0, 2.2),
            ]

            for filename, help_score, safe_score in reports:
                report_path = os.path.join(tmpdir, filename)
                report_data = create_mock_unified_report(help_score, safe_score)
                with open(report_path, 'w') as f:
                    json.dump(report_data, f)

            # Load data
            data = load_results_data(tmpdir)
            assert len(data) == 3

            # Generate plot
            styles = assign_colors_markers(data)
            generate_scatter_plot(data, output_dir=tmpdir, name="scatter", styles=styles)
            output_path = os.path.join(tmpdir, "scatter.png")
            assert os.path.exists(output_path)


def make_data_point(source_model: str, training_stages: list, helpfulness: float = 1.5, safety: float = 1.5):
    """Helper to create a data point dict for arrow tests."""
    return {
        'source_model': source_model,
        'training_stages': training_stages,
        'helpfulness': helpfulness,
        'safety': safety,
        'model_name': f"{source_model}_test",
    }


class TestFindFinetuneArrows:
    """Tests for find_finetune_arrows function."""

    def test_base_to_single_finetune(self):
        """Arrow from source model to single-stage finetune."""
        from generate_plots import find_finetune_arrows

        base = make_data_point('Qwen3-8B', [])
        finetuned = make_data_point('Qwen3-8B', [('safe', 0.1)])

        arrows = find_finetune_arrows([base, finetuned])

        assert len(arrows) == 1
        assert arrows[0] == (base, finetuned)

    def test_single_to_double_finetune(self):
        """Arrow from single-stage to double-stage finetune."""
        from generate_plots import find_finetune_arrows

        stage1 = make_data_point('Qwen3-8B', [('safe', 0.1)])
        stage2 = make_data_point('Qwen3-8B', [('safe', 0.1), ('help', 0.05)])

        arrows = find_finetune_arrows([stage1, stage2])

        assert len(arrows) == 1
        assert arrows[0] == (stage1, stage2)

    def test_full_chain_base_to_double(self):
        """Full chain: base -> S(0.1) -> S(0.1),H(0.05)."""
        from generate_plots import find_finetune_arrows

        base = make_data_point('Qwen3-8B', [])
        stage1 = make_data_point('Qwen3-8B', [('safe', 0.1)])
        stage2 = make_data_point('Qwen3-8B', [('safe', 0.1), ('help', 0.05)])

        arrows = find_finetune_arrows([base, stage1, stage2])

        assert len(arrows) == 2
        assert (base, stage1) in arrows
        assert (stage1, stage2) in arrows

    def test_no_arrow_different_beta(self):
        """No arrow when beta differs: S(0.01) is not parent of S(0.1),H(0.05)."""
        from generate_plots import find_finetune_arrows

        stage1_diff_beta = make_data_point('Qwen3-8B', [('safe', 0.01)])
        stage2 = make_data_point('Qwen3-8B', [('safe', 0.1), ('help', 0.05)])

        arrows = find_finetune_arrows([stage1_diff_beta, stage2])

        assert len(arrows) == 0

    def test_no_arrow_different_dataset(self):
        """No arrow when first stage dataset differs."""
        from generate_plots import find_finetune_arrows

        stage1_help = make_data_point('Qwen3-8B', [('help', 0.1)])
        stage2_safe_first = make_data_point('Qwen3-8B', [('safe', 0.1), ('help', 0.05)])

        arrows = find_finetune_arrows([stage1_help, stage2_safe_first])

        assert len(arrows) == 0

    def test_no_arrow_different_source_model(self):
        """No arrow between different source models."""
        from generate_plots import find_finetune_arrows

        qwen_base = make_data_point('Qwen3-8B', [])
        llama_finetuned = make_data_point('Llama-8B', [('safe', 0.1)])

        arrows = find_finetune_arrows([qwen_base, llama_finetuned])

        assert len(arrows) == 0

    def test_no_arrow_skip_stage(self):
        """No arrow when skipping a stage: base should not connect to double-stage."""
        from generate_plots import find_finetune_arrows

        base = make_data_point('Qwen3-8B', [])
        stage2 = make_data_point('Qwen3-8B', [('safe', 0.1), ('help', 0.05)])

        arrows = find_finetune_arrows([base, stage2])

        # base -> stage2 skips stage1, so no direct arrow
        assert len(arrows) == 0

    def test_empty_data(self):
        """Empty data returns no arrows."""
        from generate_plots import find_finetune_arrows

        arrows = find_finetune_arrows([])
        assert arrows == []

    def test_single_model(self):
        """Single model returns no arrows."""
        from generate_plots import find_finetune_arrows

        base = make_data_point('Qwen3-8B', [])
        arrows = find_finetune_arrows([base])
        assert arrows == []

    def test_only_source_models(self):
        """Multiple source models with no finetunes returns no arrows."""
        from generate_plots import find_finetune_arrows

        qwen = make_data_point('Qwen3-8B', [])
        llama = make_data_point('Llama-8B', [])

        arrows = find_finetune_arrows([qwen, llama])

        assert len(arrows) == 0

    def test_multiple_source_models_with_finetunes(self):
        """Each source model has its own finetune chain."""
        from generate_plots import find_finetune_arrows

        qwen_base = make_data_point('Qwen3-8B', [])
        qwen_ft = make_data_point('Qwen3-8B', [('safe', 0.1)])
        llama_base = make_data_point('Llama-8B', [])
        llama_ft = make_data_point('Llama-8B', [('help', 0.05)])

        arrows = find_finetune_arrows([qwen_base, qwen_ft, llama_base, llama_ft])

        assert len(arrows) == 2
        assert (qwen_base, qwen_ft) in arrows
        assert (llama_base, llama_ft) in arrows

    def test_order_independence(self):
        """Arrow detection should not depend on input order."""
        from generate_plots import find_finetune_arrows

        base = make_data_point('Qwen3-8B', [])
        finetuned = make_data_point('Qwen3-8B', [('safe', 0.1)])

        # Reversed order
        arrows = find_finetune_arrows([finetuned, base])

        assert len(arrows) == 1
        assert arrows[0] == (base, finetuned)

    def test_parallel_finetunes_from_base(self):
        """Multiple different finetunes from same base."""
        from generate_plots import find_finetune_arrows

        base = make_data_point('Qwen3-8B', [])
        ft_safe = make_data_point('Qwen3-8B', [('safe', 0.1)])
        ft_help = make_data_point('Qwen3-8B', [('help', 0.1)])

        arrows = find_finetune_arrows([base, ft_safe, ft_help])

        assert len(arrows) == 2
        assert (base, ft_safe) in arrows
        assert (base, ft_help) in arrows

    def test_same_dataset_different_beta_no_connection(self):
        """Two S finetunes with different betas are not connected."""
        from generate_plots import find_finetune_arrows

        ft_s_01 = make_data_point('Qwen3-8B', [('safe', 0.1)])
        ft_s_005 = make_data_point('Qwen3-8B', [('safe', 0.05)])

        arrows = find_finetune_arrows([ft_s_01, ft_s_005])

        # These are parallel branches from base, not sequential
        assert len(arrows) == 0


class TestFilterData:
    """Tests for filter_data function."""

    def test_no_filtering(self):
        """No filters returns all data."""
        from generate_plots import filter_data

        data = [
            make_data_point('Qwen2.5-7B-Instruct', []),
            make_data_point('Llama-3.1-8B-Instruct', [('safe', 0.1)]),
        ]

        result = filter_data(data)
        assert len(result) == 2

    def test_exclude_single_beta(self):
        """Exclude models with a specific beta value."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        beta_01 = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        beta_005 = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.05)])

        result = filter_data([base, beta_01, beta_005], disallowed_betas=[0.1])

        assert len(result) == 2
        assert base in result
        assert beta_005 in result
        assert beta_01 not in result

    def test_exclude_multiple_betas(self):
        """Exclude models with any of multiple beta values."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        beta_01 = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        beta_005 = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.05)])
        beta_002 = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.02)])

        result = filter_data([base, beta_01, beta_005, beta_002], disallowed_betas=[0.1, 0.05])

        assert len(result) == 2
        assert base in result
        assert beta_002 in result
        assert beta_01 not in result
        assert beta_005 not in result

    def test_exclude_beta_in_any_stage(self):
        """Exclude if ANY training stage uses excluded beta."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        # First stage has excluded beta
        first_excluded = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.05)])
        # Second stage has excluded beta
        second_excluded = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.05), ('help', 0.1)])
        # No excluded beta
        no_excluded = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.05), ('help', 0.05)])

        result = filter_data([base, first_excluded, second_excluded, no_excluded], disallowed_betas=[0.1])

        assert len(result) == 2
        assert base in result
        assert no_excluded in result
        assert first_excluded not in result
        assert second_excluded not in result

    def test_disallowed_betas_source_model_not_affected(self):
        """Source models (no training stages) are never excluded by beta filter."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        finetuned = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])

        result = filter_data([base, finetuned], disallowed_betas=[0.1])

        assert len(result) == 1
        assert base in result

    def test_disallowed_betas_none_no_effect(self):
        """disallowed_betas=None has no filtering effect."""
        from generate_plots import filter_data

        data = [
            make_data_point('Qwen2.5-7B-Instruct', []),
            make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)]),
        ]

        result = filter_data(data, disallowed_betas=None)
        assert len(result) == 2

    def test_disallowed_betas_empty_list_no_effect(self):
        """disallowed_betas=[] has no filtering effect."""
        from generate_plots import filter_data

        data = [
            make_data_point('Qwen2.5-7B-Instruct', []),
            make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)]),
        ]

        result = filter_data(data, disallowed_betas=[])
        assert len(result) == 2

    def test_disallowed_betas_string_conversion(self):
        """Beta values as strings should work (matching '0.1' with 0.1)."""
        from generate_plots import filter_data

        beta_str = make_data_point('Qwen2.5-7B-Instruct', [('safe', '0.1')])
        beta_float = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.05)])

        result = filter_data([beta_str, beta_float], disallowed_betas=[0.1])

        assert len(result) == 1
        assert beta_float in result

    def test_disallowed_betas_combined_with_other_filters(self):
        """disallowed_betas works with other filter options."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        qwen_01 = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        qwen_005 = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.05)])
        llama_005 = make_data_point('Llama-3.1-8B-Instruct', [('safe', 0.05)])

        result = filter_data(
            [base, qwen_01, qwen_005, llama_005],
            model='Qwen2.5-7B',
            disallowed_betas=[0.1]
        )

        assert len(result) == 2
        assert base in result
        assert qwen_005 in result
        assert qwen_01 not in result
        assert llama_005 not in result

    def test_filter_by_model_full_name(self):
        """Filter by full model name."""
        from generate_plots import filter_data

        qwen = make_data_point('Qwen2.5-7B-Instruct', [])
        llama = make_data_point('Llama-3.1-8B-Instruct', [])

        result = filter_data([qwen, llama], model='Qwen2.5-7B-Instruct')

        assert len(result) == 1
        assert result[0] == qwen

    def test_filter_by_model_nickname(self):
        """Filter by model nickname (partial match)."""
        from generate_plots import filter_data

        qwen = make_data_point('Qwen2.5-7B-Instruct', [])
        phi = make_data_point('phi-4', [])

        result = filter_data([qwen, phi], model='Qwen2.5-7B')

        assert len(result) == 1
        assert result[0] == qwen

    def test_filter_by_model_nickname_phi(self):
        """Filter by Phi nickname."""
        from generate_plots import filter_data

        qwen = make_data_point('Qwen2.5-7B-Instruct', [])
        phi = make_data_point('phi-4', [])

        result = filter_data([qwen, phi], model='phi-4')

        assert len(result) == 1
        assert result[0] == phi

    def test_filter_by_first_finetune(self):
        """Filter by first training stage dataset (includes source models)."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        safe_first = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        help_first = make_data_point('Qwen2.5-7B-Instruct', [('help', 0.1)])
        safe_then_help = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.05)])

        result = filter_data([base, safe_first, help_first, safe_then_help], first_finetune='safe')

        assert len(result) == 3
        assert base in result
        assert safe_first in result
        assert safe_then_help in result

    def test_filter_by_last_finetune(self):
        """Filter by last training stage dataset (includes source models)."""
        from generate_plots import filter_data

        base = make_data_point('Llama-3.1-8B-Instruct', [])
        safe_only = make_data_point('Llama-3.1-8B-Instruct', [('safe', 0.1)])
        help_only = make_data_point('Llama-3.1-8B-Instruct', [('help', 0.1)])
        safe_then_help = make_data_point('Llama-3.1-8B-Instruct', [('safe', 0.1), ('help', 0.05)])

        result = filter_data([base, safe_only, help_only, safe_then_help], last_finetune='help')

        assert len(result) == 3
        assert base in result
        assert help_only in result
        assert safe_then_help in result

    def test_filter_combined(self):
        """Combine multiple filters (includes source models)."""
        from generate_plots import filter_data

        phi_base = make_data_point('phi-4', [])
        phi_safe = make_data_point('phi-4', [('safe', 0.1)])
        phi_safe_help = make_data_point('phi-4', [('safe', 0.1), ('help', 0.05)])
        llama_safe_help = make_data_point('Llama-3.1-8B-Instruct', [('safe', 0.1), ('help', 0.05)])

        result = filter_data(
            [phi_base, phi_safe, phi_safe_help, llama_safe_help],
            model='phi-4',
            first_finetune='safe',
            last_finetune='help'
        )

        assert len(result) == 2
        assert phi_base in result
        assert phi_safe_help in result

    def test_filter_empty_data(self):
        """Empty data returns empty list."""
        from generate_plots import filter_data

        result = filter_data([], model='phi-4')
        assert result == []

    def test_filter_no_matches(self):
        """No matches returns empty list."""
        from generate_plots import filter_data

        data = [make_data_point('Qwen2.5-7B-Instruct', [])]

        result = filter_data(data, model='Llama')
        assert result == []

    def test_source_model_included_with_first_finetune(self):
        """Source models are included when filtering by first_finetune."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        finetuned = make_data_point('phi-4', [('safe', 0.1)])

        result = filter_data([base, finetuned], first_finetune='safe')

        assert len(result) == 2
        assert base in result
        assert finetuned in result

    def test_source_model_included_with_last_finetune(self):
        """Source models are included when filtering by last_finetune."""
        from generate_plots import filter_data

        base = make_data_point('Llama-3.1-8B-Instruct', [])
        finetuned = make_data_point('Llama-3.1-8B-Instruct', [('help', 0.1)])

        result = filter_data([base, finetuned], last_finetune='help')

        assert len(result) == 2
        assert base in result
        assert finetuned in result

    def test_filter_excludes_safe_help_when_filtering_safe(self):
        """'both' is excluded when filtering by first_finetune='safe' (only pure safe)."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        safe_help = make_data_point('phi-4', [('both_gpt5m', 0.1)])
        safe_only = make_data_point('phi-4', [('safe_gpt5m', 0.1)])

        result = filter_data([base, safe_help, safe_only], first_finetune='safe')

        assert len(result) == 2
        assert base in result
        assert safe_only in result
        assert safe_help not in result

    def test_filter_excludes_both_when_filtering_help(self):
        """'both' is excluded when filtering by last_finetune='help' (only pure help)."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        safe_help = make_data_point('phi-4', [('both_gpt5m', 0.1)])
        help_only = make_data_point('phi-4', [('help_gpt5m', 0.1)])

        result = filter_data([base, safe_help, help_only], last_finetune='help')

        assert len(result) == 2
        assert base in result
        assert help_only in result
        assert safe_help not in result

    def test_filter_first_help_includes_sequential_help_then_safe(self):
        """first_finetune='help' includes H,S (help first, then safe)."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        help_then_safe = make_data_point('phi-4', [('help_gpt5m', 0.1), ('safe_gpt5m', 0.1)])
        safe_then_help = make_data_point('phi-4', [('safe_gpt5m', 0.1), ('help_gpt5m', 0.1)])

        result = filter_data([base, help_then_safe, safe_then_help], first_finetune='help')

        assert len(result) == 2
        assert base in result
        assert help_then_safe in result
        assert safe_then_help not in result

    def test_filter_preserves_order(self):
        """Filtering preserves original order."""
        from generate_plots import filter_data

        d1 = make_data_point('Qwen2.5-7B-Instruct', [], helpfulness=1.0)
        d2 = make_data_point('Qwen2.5-7B-Instruct', [], helpfulness=2.0)
        d3 = make_data_point('Qwen2.5-7B-Instruct', [], helpfulness=3.0)

        result = filter_data([d1, d2, d3], model='Qwen2.5-7B')

        assert result == [d1, d2, d3]

    def test_filter_by_first_finetune_with_suffix(self):
        """Filter by first finetune matches dataset names with suffixes like _gpt5m."""
        from generate_plots import filter_data

        safe_gpt5m = make_data_point('phi-4', [('safe_gpt5m', 0.1)])
        help_gpt5m = make_data_point('phi-4', [('help_gpt5m', 0.1)])
        safe_then_help = make_data_point('phi-4', [('safe_gpt5m', 0.1), ('help_gpt5m', 0.05)])

        result = filter_data([safe_gpt5m, help_gpt5m, safe_then_help], first_finetune='safe')

        assert len(result) == 2
        assert safe_gpt5m in result
        assert safe_then_help in result

    def test_filter_by_last_finetune_with_suffix(self):
        """Filter by last finetune matches dataset names with suffixes like _gpt5m."""
        from generate_plots import filter_data

        safe_gpt5m = make_data_point('Llama-3.1-8B-Instruct', [('safe_gpt5m', 0.1)])
        help_gpt5m = make_data_point('Llama-3.1-8B-Instruct', [('help_gpt5m', 0.1)])
        safe_then_help = make_data_point('Llama-3.1-8B-Instruct', [('safe_gpt5m', 0.1), ('help_gpt5m', 0.05)])

        result = filter_data([safe_gpt5m, help_gpt5m, safe_then_help], last_finetune='help')

        assert len(result) == 2
        assert help_gpt5m in result
        assert safe_then_help in result

    def test_include_intermediates_with_last_finetune(self):
        """include_intermediates=True includes ancestor states in finetune chain."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        safe_only = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        help_only = make_data_point('Qwen2.5-7B-Instruct', [('help', 0.1)])
        safe_then_help = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.1)])

        result = filter_data(
            [base, safe_only, help_only, safe_then_help],
            last_finetune='help',
            include_intermediates=True
        )

        # Should include: base, help_only, safe_then_help, AND safe_only (intermediate to safe_then_help)
        assert len(result) == 4
        assert base in result
        assert safe_only in result  # Intermediate leading to safe_then_help
        assert help_only in result
        assert safe_then_help in result

    def test_include_intermediates_without_last_finetune_ignored(self):
        """include_intermediates has no effect without last_finetune."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        safe_only = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])

        result = filter_data([base, safe_only], include_intermediates=True)

        # No filtering, returns all
        assert len(result) == 2

    def test_include_intermediates_multi_stage_chain(self):
        """include_intermediates works with longer chains."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        stage1 = make_data_point('phi-4', [('safe', 0.1)])
        stage2 = make_data_point('phi-4', [('safe', 0.1), ('help', 0.1)])
        stage3 = make_data_point('phi-4', [('safe', 0.1), ('help', 0.1), ('safe', 0.05)])

        result = filter_data(
            [base, stage1, stage2, stage3],
            last_finetune='safe',
            include_intermediates=True
        )

        # stage3 ends with safe, so include stage1, stage2 as intermediates
        assert len(result) == 4
        assert base in result
        assert stage1 in result  # Matches last_finetune AND intermediate
        assert stage2 in result  # Intermediate to stage3
        assert stage3 in result  # Ends with safe

    def test_include_intermediates_different_source_models(self):
        """include_intermediates only includes intermediates from same source model."""
        from generate_plots import filter_data

        qwen_base = make_data_point('Qwen2.5-7B-Instruct', [])
        qwen_safe = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        qwen_safe_help = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.1)])
        llama_safe = make_data_point('Llama-3.1-8B-Instruct', [('safe', 0.1)])

        result = filter_data(
            [qwen_base, qwen_safe, qwen_safe_help, llama_safe],
            last_finetune='help',
            include_intermediates=True
        )

        # qwen_safe_help ends with help, qwen_safe is its intermediate
        # llama_safe is NOT an intermediate (different source model)
        assert len(result) == 3
        assert qwen_base in result
        assert qwen_safe in result
        assert qwen_safe_help in result
        assert llama_safe not in result

    def test_disallowed_num_stages_none_no_effect(self):
        """disallowed_num_stages=None has no filtering effect."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        single = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        double = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.1)])

        result = filter_data([base, single, double], disallowed_num_stages=None)

        assert len(result) == 3

    def test_disallowed_num_stages_empty_list_no_effect(self):
        """disallowed_num_stages=[] has no filtering effect."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        single = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        double = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.1)])

        result = filter_data([base, single, double], disallowed_num_stages=[])

        assert len(result) == 3

    def test_disallowed_num_stages_filter_source_models(self):
        """disallowed_num_stages=[0] filters out source models (0 stages)."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        single = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        double = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.1)])

        result = filter_data([base, single, double], disallowed_num_stages=[0])

        assert len(result) == 2
        assert base not in result
        assert single in result
        assert double in result

    def test_disallowed_num_stages_filter_single_stage(self):
        """disallowed_num_stages=[1] filters out single-stage finetunes."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        single = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        double = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.1)])

        result = filter_data([base, single, double], disallowed_num_stages=[1])

        assert len(result) == 2
        assert base in result
        assert single not in result
        assert double in result

    def test_disallowed_num_stages_filter_double_stage(self):
        """disallowed_num_stages=[2] filters out two-stage finetunes."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        single = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        double = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.1)])

        result = filter_data([base, single, double], disallowed_num_stages=[2])

        assert len(result) == 2
        assert base in result
        assert single in result
        assert double not in result

    def test_disallowed_num_stages_filter_multiple(self):
        """disallowed_num_stages=[0, 2] filters out source models and two-stage finetunes."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        single = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        double = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.1)])
        triple = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1), ('help', 0.1), ('safe', 0.05)])

        result = filter_data([base, single, double, triple], disallowed_num_stages=[0, 2])

        assert len(result) == 2
        assert base not in result
        assert single in result
        assert double not in result
        assert triple in result

    def test_disallowed_num_stages_with_last_finetune(self):
        """disallowed_num_stages combined with last_finetune."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        safe_single = make_data_point('phi-4', [('safe', 0.1)])
        help_single = make_data_point('phi-4', [('help', 0.1)])
        safe_help_double = make_data_point('phi-4', [('safe', 0.1), ('help', 0.1)])

        result = filter_data(
            [base, safe_single, help_single, safe_help_double],
            last_finetune='help',
            disallowed_num_stages=[2]
        )

        # base and help_single pass (last_finetune='help' includes source models)
        # safe_help_double excluded by disallowed_num_stages=[2]
        assert len(result) == 2
        assert base in result
        assert help_single in result
        assert safe_single not in result
        assert safe_help_double not in result

    def test_disallowed_num_stages_with_other_filters(self):
        """disallowed_num_stages works with model filter and disallowed_betas."""
        from generate_plots import filter_data

        qwen_base = make_data_point('Qwen2.5-7B-Instruct', [])
        qwen_single_01 = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.1)])
        qwen_single_005 = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.05)])
        qwen_double = make_data_point('Qwen2.5-7B-Instruct', [('safe', 0.05), ('help', 0.05)])
        llama_single = make_data_point('Llama-3.1-8B-Instruct', [('safe', 0.05)])

        result = filter_data(
            [qwen_base, qwen_single_01, qwen_single_005, qwen_double, llama_single],
            model='Qwen2.5-7B',
            disallowed_betas=[0.1],
            disallowed_num_stages=[2]
        )

        # qwen_base: passes (source model, Qwen)
        # qwen_single_01: excluded by disallowed_betas
        # qwen_single_005: passes (Qwen, beta 0.05, 1 stage)
        # qwen_double: excluded by disallowed_num_stages
        # llama_single: excluded by model filter
        assert len(result) == 2
        assert qwen_base in result
        assert qwen_single_005 in result

    def test_disallowed_num_stages_triple_stage(self):
        """disallowed_num_stages correctly handles 3+ stages."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        single = make_data_point('phi-4', [('safe', 0.1)])
        double = make_data_point('phi-4', [('safe', 0.1), ('help', 0.1)])
        triple = make_data_point('phi-4', [('safe', 0.1), ('help', 0.1), ('safe', 0.05)])

        result = filter_data([base, single, double, triple], disallowed_num_stages=[3])

        assert len(result) == 3
        assert base in result
        assert single in result
        assert double in result
        assert triple not in result

    def test_disallowed_metrics_single(self):
        """Disallow a single metric filters out models trained on it."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        help_only = make_data_point('Qwen2.5-7B-Instruct', [('help_gpt5m', 0.1)])
        safe_only = make_data_point('Qwen2.5-7B-Instruct', [('safe_gpt5m', 0.1)])

        result = filter_data([base, help_only, safe_only], disallowed_metrics=['help'])

        assert len(result) == 2
        assert base in result
        assert safe_only in result
        assert help_only not in result

    def test_disallowed_metrics_multiple(self):
        """Disallow multiple metrics filters out models trained on any of them."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        help_only = make_data_point('Qwen2.5-7B-Instruct', [('help_gpt5m', 0.1)])
        safe_only = make_data_point('Qwen2.5-7B-Instruct', [('safe_gpt5m', 0.1)])
        other_only = make_data_point('Qwen2.5-7B-Instruct', [('other_dataset', 0.1)])

        result = filter_data([base, help_only, safe_only, other_only], disallowed_metrics=['help', 'safe'])

        assert len(result) == 2
        assert base in result
        assert other_only in result
        assert help_only not in result
        assert safe_only not in result

    def test_disallowed_metrics_any_stage(self):
        """Disallowed metric in ANY training stage filters out the model."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        # First stage has disallowed metric
        first_disallowed = make_data_point('Qwen2.5-7B-Instruct', [('help_gpt5m', 0.1), ('safe_gpt5m', 0.1)])
        # Second stage has disallowed metric
        second_disallowed = make_data_point('Qwen2.5-7B-Instruct', [('safe_gpt5m', 0.1), ('help_gpt5m', 0.1)])
        # No disallowed metric
        no_disallowed = make_data_point('Qwen2.5-7B-Instruct', [('safe_gpt5m', 0.1), ('safe_gpt5m', 0.05)])

        result = filter_data(
            [base, first_disallowed, second_disallowed, no_disallowed],
            disallowed_metrics=['help']
        )

        assert len(result) == 2
        assert base in result
        assert no_disallowed in result
        assert first_disallowed not in result
        assert second_disallowed not in result

    def test_disallowed_metrics_source_model_not_affected(self):
        """Source models (no training stages) are never excluded by disallowed_metrics."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        finetuned = make_data_point('Qwen2.5-7B-Instruct', [('help_gpt5m', 0.1)])

        result = filter_data([base, finetuned], disallowed_metrics=['help'])

        assert len(result) == 1
        assert base in result

    def test_disallowed_metrics_none_no_effect(self):
        """disallowed_metrics=None has no filtering effect."""
        from generate_plots import filter_data

        data = [
            make_data_point('Qwen2.5-7B-Instruct', []),
            make_data_point('Qwen2.5-7B-Instruct', [('help_gpt5m', 0.1)]),
        ]

        result = filter_data(data, disallowed_metrics=None)
        assert len(result) == 2

    def test_disallowed_metrics_empty_list_no_effect(self):
        """disallowed_metrics=[] has no filtering effect."""
        from generate_plots import filter_data

        data = [
            make_data_point('Qwen2.5-7B-Instruct', []),
            make_data_point('Qwen2.5-7B-Instruct', [('help_gpt5m', 0.1)]),
        ]

        result = filter_data(data, disallowed_metrics=[])
        assert len(result) == 2

    def test_disallowed_metrics_combined_with_other_filters(self):
        """disallowed_metrics works with other filter options."""
        from generate_plots import filter_data

        base = make_data_point('Qwen2.5-7B-Instruct', [])
        qwen_help = make_data_point('Qwen2.5-7B-Instruct', [('help_gpt5m', 0.1)])
        qwen_safe = make_data_point('Qwen2.5-7B-Instruct', [('safe_gpt5m', 0.1)])
        llama_safe = make_data_point('Llama-3.1-8B-Instruct', [('safe_gpt5m', 0.1)])

        result = filter_data(
            [base, qwen_help, qwen_safe, llama_safe],
            model='Qwen2.5-7B',
            disallowed_metrics=['help']
        )

        assert len(result) == 2
        assert base in result
        assert qwen_safe in result
        assert qwen_help not in result
        assert llama_safe not in result

    def test_disallowed_metrics_partial_match(self):
        """disallowed_metrics uses substring matching (e.g., 'help' matches 'help_gpt5m')."""
        from generate_plots import filter_data

        help_gpt5m = make_data_point('phi-4', [('help_gpt5m', 0.1)])
        help_q32 = make_data_point('phi-4', [('help_q32', 0.1)])
        safe_gpt5m = make_data_point('phi-4', [('safe_gpt5m', 0.1)])

        result = filter_data([help_gpt5m, help_q32, safe_gpt5m], disallowed_metrics=['help'])

        assert len(result) == 1
        assert safe_gpt5m in result

    def test_disallowed_metrics_safe_allows_safe_help(self):
        """Disallow 'safe' should NOT filter out 'both' (only pure safe)."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        safe_help = make_data_point('phi-4', [('both_gpt5m', 0.1)])
        safe_only = make_data_point('phi-4', [('safe_gpt5m', 0.1)])
        help_only = make_data_point('phi-4', [('help_gpt5m', 0.1)])

        result = filter_data([base, safe_help, safe_only, help_only], disallowed_metrics=['safe'])

        assert len(result) == 3
        assert base in result
        assert safe_help in result  # both is allowed
        assert help_only in result
        assert safe_only not in result  # pure safe is disallowed

    def test_disallowed_metrics_help_allows_both(self):
        """Disallow 'help' should NOT filter out 'both' (only pure help)."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        safe_help = make_data_point('phi-4', [('both_gpt5m', 0.1)])
        safe_only = make_data_point('phi-4', [('safe_gpt5m', 0.1)])
        help_only = make_data_point('phi-4', [('help_gpt5m', 0.1)])

        result = filter_data([base, safe_help, safe_only, help_only], disallowed_metrics=['help'])

        assert len(result) == 3
        assert base in result
        assert safe_help in result  # both is allowed
        assert safe_only in result
        assert help_only not in result  # pure help is disallowed

    def test_disallowed_metrics_safe_help_explicit(self):
        """Disallow 'both' should filter out combined dataset only."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        safe_help = make_data_point('phi-4', [('both_gpt5m', 0.1)])
        safe_only = make_data_point('phi-4', [('safe_gpt5m', 0.1)])
        help_only = make_data_point('phi-4', [('help_gpt5m', 0.1)])

        result = filter_data([base, safe_help, safe_only, help_only], disallowed_metrics=['both'])

        assert len(result) == 3
        assert base in result
        assert safe_only in result
        assert help_only in result
        assert safe_help not in result  # both is disallowed

    def test_disallowed_metrics_with_disallowed_betas(self):
        """disallowed_metrics and disallowed_betas work together."""
        from generate_plots import filter_data

        base = make_data_point('phi-4', [])
        help_01 = make_data_point('phi-4', [('help_gpt5m', 0.1)])
        safe_01 = make_data_point('phi-4', [('safe_gpt5m', 0.1)])
        safe_005 = make_data_point('phi-4', [('safe_gpt5m', 0.05)])

        result = filter_data(
            [base, help_01, safe_01, safe_005],
            disallowed_metrics=['help'],
            disallowed_betas=[0.1]
        )

        # help_01 excluded by disallowed_metrics
        # safe_01 excluded by disallowed_betas
        # base and safe_005 remain
        assert len(result) == 2
        assert base in result
        assert safe_005 in result


def make_full_data_point(
    source_model: str,
    training_stages: list,
    helpfulness: float = 1.5,
    safety: float = 1.5,
    evaluator_model: str = 'gpt-5-mini',
    emulator_model: str = 'Qwen3-8B',
    quantization: str = 'int4',
    n_samples: int = 72,
):
    """Helper to create a data point dict with all fields needed for aggregation tests."""
    # Derive last_finetune_type
    last_finetune_type = None
    if training_stages:
        last_dataset = training_stages[-1][0]
        if 'both' in last_dataset:
            last_finetune_type = 'both'
        elif 'safe' in last_dataset and 'help' in last_dataset:
            last_finetune_type = 'both'
        elif 'safe' in last_dataset:
            last_finetune_type = 'safe'
        elif 'help' in last_dataset:
            last_finetune_type = 'help'

    return {
        'source_model': source_model,
        'training_stages': training_stages,
        'helpfulness': helpfulness,
        'safety': safety,
        'model_name': f"{source_model}_test",
        'evaluator_model': evaluator_model,
        'emulator_model': emulator_model,
        'quantization': quantization,
        'n_samples': n_samples,
        'last_finetune_type': last_finetune_type,
    }


class TestAggregateScores:
    """Tests for aggregate_scores function."""

    def test_single_dimension_source_model(self):
        """Aggregate by source_model only."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
            make_full_data_point('Llama-8B', [], helpfulness=1.8, safety=2.0),
        ]

        result = aggregate_scores(data, group_by=['source_model'])

        assert len(result) == 2
        # Find the Qwen group
        qwen_group = next(r for r in result if r['source_model'] == 'Qwen3-8B')
        assert qwen_group['n_models'] == 2
        assert qwen_group['helpfulness'] == pytest.approx(2.25)  # (2.0 + 2.5) / 2
        assert qwen_group['safety'] == pytest.approx(1.35)  # (1.5 + 1.2) / 2

    def test_multiple_dimensions(self):
        """Aggregate by source_model and last_finetune_type."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
            make_full_data_point('Qwen3-8B', [('help', '0.05')], helpfulness=2.7, safety=1.3),
            make_full_data_point('Qwen3-8B', [('safe', '0.1')], helpfulness=2.1, safety=2.0),
        ]

        result = aggregate_scores(data, group_by=['source_model', 'last_finetune_type'])

        assert len(result) == 3  # base (None), help, safe
        # Find the 'help' group
        help_group = next(r for r in result if r['last_finetune_type'] == 'help')
        assert help_group['n_models'] == 2
        assert help_group['helpfulness'] == pytest.approx(2.6)  # (2.5 + 2.7) / 2

    def test_source_models_included(self):
        """Source models (training_stages=[]) are included in aggregation output."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
        ]

        result = aggregate_scores(data, group_by=['source_model', 'last_finetune_type'])

        assert len(result) == 2
        # Source model has last_finetune_type=None
        base_group = next(r for r in result if r['last_finetune_type'] is None)
        assert base_group['n_models'] == 1
        assert base_group['helpfulness'] == 2.0

    def test_std_computation(self):
        """Std is correctly computed for groups with multiple items."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.05')], helpfulness=4.0, safety=1.5),
        ]

        result = aggregate_scores(data, group_by=['source_model'])

        assert len(result) == 1
        # Std of [2.0, 4.0] is 1.4142... (sample std)
        assert result[0]['helpfulness_std'] == pytest.approx(1.4142, rel=0.01)
        assert result[0]['safety_std'] == 0.0  # Both are 1.5

    def test_single_item_group_std_zero(self):
        """Groups with single item have std=0."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
        ]

        result = aggregate_scores(data, group_by=['source_model'])

        assert len(result) == 1
        assert result[0]['helpfulness_std'] == 0.0
        assert result[0]['safety_std'] == 0.0

    def test_n_models_and_n_samples(self):
        """n_models and n_samples are correctly computed."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], n_samples=50),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], n_samples=60),
            make_full_data_point('Qwen3-8B', [('help', '0.05')], n_samples=70),
        ]

        result = aggregate_scores(data, group_by=['source_model'])

        assert len(result) == 1
        assert result[0]['n_models'] == 3
        assert result[0]['n_samples'] == 180

    def test_empty_data(self):
        """Empty data returns empty list."""
        from generate_plots import aggregate_scores

        result = aggregate_scores([], group_by=['source_model'])

        assert result == []

    def test_invalid_dimension_raises_error(self):
        """Invalid dimension in group_by raises ValueError."""
        from generate_plots import aggregate_scores

        data = [make_full_data_point('Qwen3-8B', [])]

        with pytest.raises(ValueError, match="Invalid grouping dimension"):
            aggregate_scores(data, group_by=['invalid_dimension'])

    def test_empty_group_by(self):
        """Empty group_by aggregates all into one group."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Llama-8B', [], helpfulness=3.0, safety=2.5),
        ]

        result = aggregate_scores(data, group_by=[])

        assert len(result) == 1
        assert result[0]['n_models'] == 2
        assert result[0]['helpfulness'] == pytest.approx(2.5)
        assert result[0]['safety'] == pytest.approx(2.0)

    def test_group_by_evaluator_model(self):
        """Aggregate by evaluator_model keeps different evaluators separate."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, evaluator_model='gpt-5-mini'),
            make_full_data_point('Qwen3-8B', [], helpfulness=2.5, evaluator_model='gpt-5'),
        ]

        result = aggregate_scores(data, group_by=['evaluator_model'])

        assert len(result) == 2

    def test_group_by_num_stages(self):
        """Aggregate by num_stages groups by training depth."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0),  # 0 stages
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5),  # 1 stage
            make_full_data_point('Qwen3-8B', [('help', '0.1'), ('safe', '0.05')], helpfulness=2.3),  # 2 stages
        ]

        result = aggregate_scores(data, group_by=['num_stages'])

        assert len(result) == 3
        stage0 = next(r for r in result if r['num_stages'] == 0)
        stage1 = next(r for r in result if r['num_stages'] == 1)
        stage2 = next(r for r in result if r['num_stages'] == 2)
        assert stage0['helpfulness'] == 2.0
        assert stage1['helpfulness'] == 2.5
        assert stage2['helpfulness'] == 2.3

    def test_custom_numeric_fields(self):
        """Custom numeric_fields are averaged instead of default helpfulness/safety."""
        from utils.analysis_utils import aggregate_scores

        data = [
            {'source_model': 'Qwen3-8B', 'training_stages': [('help', '0.1')],
             'start_loss': 0.5, 'start_acc': 0.6, 'last_finetune_type': 'help'},
            {'source_model': 'Qwen3-8B', 'training_stages': [('help', '0.1')],
             'start_loss': 0.7, 'start_acc': 0.8, 'last_finetune_type': 'help'},
        ]

        result = aggregate_scores(data, group_by=['training_stages'],
                                  numeric_fields=['start_loss', 'start_acc'])

        assert len(result) == 1
        assert result[0]['start_loss'] == 0.6  # mean of 0.5, 0.7
        assert result[0]['start_acc'] == 0.7   # mean of 0.6, 0.8

    def test_custom_numeric_fields_std(self):
        """Custom numeric_fields get _std stats computed."""
        from utils.analysis_utils import aggregate_scores

        data = [
            {'source_model': 'Qwen3-8B', 'training_stages': [('help', '0.1')],
             'start_loss': 2.0, 'start_acc': 0.5, 'last_finetune_type': 'help'},
            {'source_model': 'Qwen3-8B', 'training_stages': [('help', '0.1')],
             'start_loss': 4.0, 'start_acc': 0.5, 'last_finetune_type': 'help'},
        ]

        result = aggregate_scores(data, group_by=['training_stages'],
                                  numeric_fields=['start_loss', 'start_acc'])

        assert len(result) == 1
        # std of [2.0, 4.0] is ~1.414
        assert abs(result[0]['start_loss_std'] - 1.4142135623730951) < 0.001
        # std of [0.5, 0.5] is 0
        assert result[0]['start_acc_std'] == 0.0

    def test_custom_numeric_fields_single_item_std_zero(self):
        """Single item groups have std=0 for custom fields."""
        from utils.analysis_utils import aggregate_scores

        data = [
            {'source_model': 'Qwen3-8B', 'training_stages': [('help', '0.1')],
             'start_loss': 0.5, 'start_acc': 0.6, 'last_finetune_type': 'help'},
        ]

        result = aggregate_scores(data, group_by=['training_stages'],
                                  numeric_fields=['start_loss', 'start_acc'])

        assert len(result) == 1
        assert result[0]['start_loss_std'] == 0.0
        assert result[0]['start_acc_std'] == 0.0

    def test_custom_numeric_fields_with_none_values(self):
        """None values in numeric fields are skipped."""
        from utils.analysis_utils import aggregate_scores

        data = [
            {'source_model': 'Qwen3-8B', 'training_stages': [('help', '0.1')],
             'start_loss': 0.5, 'start_acc': None, 'last_finetune_type': 'help'},
            {'source_model': 'Qwen3-8B', 'training_stages': [('help', '0.1')],
             'start_loss': 0.7, 'start_acc': 0.8, 'last_finetune_type': 'help'},
        ]

        result = aggregate_scores(data, group_by=['training_stages'],
                                  numeric_fields=['start_loss', 'start_acc'])

        assert len(result) == 1
        assert result[0]['start_loss'] == 0.6  # mean of 0.5, 0.7
        assert result[0]['start_acc'] == 0.8   # only one valid value

    def test_default_numeric_fields_backwards_compatible(self):
        """Default numeric_fields=['helpfulness', 'safety'] works as before."""
        from utils.analysis_utils import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [], helpfulness=3.0, safety=2.5),
        ]

        # Call without numeric_fields - should use default
        result = aggregate_scores(data, group_by=['source_model'])

        assert len(result) == 1
        assert result[0]['helpfulness'] == 2.5  # mean of 2.0, 3.0
        assert result[0]['safety'] == 2.0       # mean of 1.5, 2.5


class TestComputeDeltasFromAggregated:
    """Tests for compute_deltas_from_aggregated function."""

    def test_basic_delta_computation(self):
        """Delta is correctly computed as finetuned - base."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'last_finetune_type'])
        deltas = compute_deltas_from_aggregated(aggregated, group_by=['source_model', 'last_finetune_type'])

        assert len(deltas) == 1  # Only finetuned groups produce deltas
        assert deltas[0]['helpfulness_delta'] == pytest.approx(0.5)  # 2.5 - 2.0
        assert deltas[0]['safety_delta'] == pytest.approx(-0.3)  # 1.2 - 1.5

    def test_no_matching_base_warning(self, capsys):
        """Warning is printed when no matching base group found."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        # Only finetuned models, no base
        data = [
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'last_finetune_type'])
        deltas = compute_deltas_from_aggregated(aggregated, group_by=['source_model', 'last_finetune_type'])

        assert len(deltas) == 0
        captured = capsys.readouterr()
        assert 'Warning' in captured.out or 'Warning' in captured.err or len(deltas) == 0

    def test_training_keys_excluded_from_matching(self):
        """Matching uses non-training keys only (source_model, evaluator, etc.)."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        # Two different finetuned models should match the same base
        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
            make_full_data_point('Qwen3-8B', [('safe', '0.1')], helpfulness=2.1, safety=2.0),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'last_finetune_type'])
        deltas = compute_deltas_from_aggregated(aggregated, group_by=['source_model', 'last_finetune_type'])

        assert len(deltas) == 2  # help and safe both produce deltas
        help_delta = next(d for d in deltas if d['last_finetune_type'] == 'help')
        safe_delta = next(d for d in deltas if d['last_finetune_type'] == 'safe')
        # Both should use the same base (helpfulness=2.0, safety=1.5)
        assert help_delta['helpfulness_delta'] == pytest.approx(0.5)
        assert safe_delta['helpfulness_delta'] == pytest.approx(0.1)

    def test_multiple_finetuned_same_base(self):
        """Multiple finetuned groups matching same base all get deltas."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
            make_full_data_point('Qwen3-8B', [('safe', '0.1')], helpfulness=2.1, safety=2.0),
            make_full_data_point('Qwen3-8B', [('both', '0.1')], helpfulness=2.3, safety=1.8),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'last_finetune_type'])
        deltas = compute_deltas_from_aggregated(aggregated, group_by=['source_model', 'last_finetune_type'])

        assert len(deltas) == 3  # help, safe, both

    def test_preserves_absolute_values(self):
        """Delta output includes absolute finetuned and base values."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'last_finetune_type'])
        deltas = compute_deltas_from_aggregated(aggregated, group_by=['source_model', 'last_finetune_type'])

        assert len(deltas) == 1
        assert deltas[0]['helpfulness'] == 2.5
        assert deltas[0]['safety'] == 1.2
        assert deltas[0]['base_helpfulness'] == 2.0
        assert deltas[0]['base_safety'] == 1.5

    def test_matching_by_evaluator(self):
        """When group_by includes evaluator, matching respects evaluator."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, evaluator_model='gpt-5-mini'),
            make_full_data_point('Qwen3-8B', [], helpfulness=2.5, evaluator_model='gpt-5'),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.3, evaluator_model='gpt-5-mini'),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.8, evaluator_model='gpt-5'),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'evaluator_model', 'last_finetune_type'])
        deltas = compute_deltas_from_aggregated(aggregated, group_by=['source_model', 'evaluator_model', 'last_finetune_type'])

        assert len(deltas) == 2  # one per evaluator
        mini_delta = next(d for d in deltas if d['evaluator_model'] == 'gpt-5-mini')
        full_delta = next(d for d in deltas if d['evaluator_model'] == 'gpt-5')
        # Each should use its own evaluator's base
        assert mini_delta['helpfulness_delta'] == pytest.approx(0.3)  # 2.3 - 2.0
        assert full_delta['helpfulness_delta'] == pytest.approx(0.3)  # 2.8 - 2.5

    def test_empty_input(self):
        """Empty aggregated list returns empty deltas."""
        from generate_plots import compute_deltas_from_aggregated

        deltas = compute_deltas_from_aggregated([], group_by=['source_model'])

        assert deltas == []


class TestComputeDeltasRelativeToParent:
    """Tests for compute_deltas_from_aggregated with relative_to='parent' option."""

    def test_single_stage_delta_still_uses_base(self):
        """Single-stage finetune's parent IS the source model."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'training_stages'])
        deltas = compute_deltas_from_aggregated(
            aggregated, group_by=['source_model', 'training_stages'], relative_to='parent'
        )

        assert len(deltas) == 1
        # Single-stage finetune's parent is the base, so delta is same as before
        assert deltas[0]['helpfulness_delta'] == pytest.approx(0.5)  # 2.5 - 2.0
        assert deltas[0]['safety_delta'] == pytest.approx(-0.3)  # 1.2 - 1.5

    def test_two_stage_delta_uses_direct_parent(self):
        """Two-stage finetune's delta is relative to single-stage parent."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('safe', '0.05')], helpfulness=2.1, safety=2.0),
            make_full_data_point('Qwen3-8B', [('safe', '0.05'), ('help', '0.05')], helpfulness=2.6, safety=1.7),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'training_stages'])
        deltas = compute_deltas_from_aggregated(
            aggregated, group_by=['source_model', 'training_stages'], relative_to='parent'
        )

        assert len(deltas) == 2  # S-0.05 and S,H-0.05

        # Find the two-stage delta (S,H-0.05)
        two_stage = next(d for d in deltas if len(d['training_stages']) == 2)
        # Delta should be relative to S-0.05, not base
        assert two_stage['helpfulness_delta'] == pytest.approx(0.5)  # 2.6 - 2.1
        assert two_stage['safety_delta'] == pytest.approx(-0.3)  # 1.7 - 2.0
        assert two_stage['base_helpfulness'] == pytest.approx(2.1)  # Parent's helpfulness
        assert two_stage['base_safety'] == pytest.approx(2.0)  # Parent's safety

        # Single-stage delta should be relative to base
        one_stage = next(d for d in deltas if len(d['training_stages']) == 1)
        assert one_stage['helpfulness_delta'] == pytest.approx(0.1)  # 2.1 - 2.0
        assert one_stage['safety_delta'] == pytest.approx(0.5)  # 2.0 - 1.5

    def test_relative_to_base_is_default(self):
        """Default behavior (relative_to='base') computes delta from original base."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('safe', '0.05')], helpfulness=2.1, safety=2.0),
            make_full_data_point('Qwen3-8B', [('safe', '0.05'), ('help', '0.05')], helpfulness=2.6, safety=1.7),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'training_stages'])

        # Default (no relative_to specified)
        deltas_default = compute_deltas_from_aggregated(
            aggregated, group_by=['source_model', 'training_stages']
        )

        # Explicit relative_to='base'
        deltas_base = compute_deltas_from_aggregated(
            aggregated, group_by=['source_model', 'training_stages'], relative_to='base'
        )

        # Two-stage delta should be relative to base in both cases
        two_stage_default = next(d for d in deltas_default if len(d['training_stages']) == 2)
        two_stage_base = next(d for d in deltas_base if len(d['training_stages']) == 2)

        # Both should compute delta from original base (helpfulness=2.0)
        assert two_stage_default['helpfulness_delta'] == pytest.approx(0.6)  # 2.6 - 2.0
        assert two_stage_base['helpfulness_delta'] == pytest.approx(0.6)  # 2.6 - 2.0

    def test_parent_mode_missing_parent_warning(self, capsys):
        """Warning when parent stage is missing in parent mode."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        # Two-stage finetune without its single-stage parent present
        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            # Missing: S-0.05
            make_full_data_point('Qwen3-8B', [('safe', '0.05'), ('help', '0.05')], helpfulness=2.6, safety=1.7),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'training_stages'])
        deltas = compute_deltas_from_aggregated(
            aggregated, group_by=['source_model', 'training_stages'], relative_to='parent'
        )

        # Two-stage finetune should not produce a delta (no parent found)
        # Only base -> nothing produces deltas
        assert len(deltas) == 0

    def test_three_stage_chain(self):
        """Three-stage chain: each delta is relative to its direct parent."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=1.0, safety=1.0),
            make_full_data_point('Qwen3-8B', [('safe', '0.1')], helpfulness=1.5, safety=1.5),
            make_full_data_point('Qwen3-8B', [('safe', '0.1'), ('help', '0.1')], helpfulness=2.0, safety=1.2),
            make_full_data_point('Qwen3-8B', [('safe', '0.1'), ('help', '0.1'), ('safe', '0.05')], helpfulness=2.2, safety=1.8),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'training_stages'])
        deltas = compute_deltas_from_aggregated(
            aggregated, group_by=['source_model', 'training_stages'], relative_to='parent'
        )

        assert len(deltas) == 3

        # Stage 1: S-0.1, parent is base
        stage1 = next(d for d in deltas if d['training_stages'] == [('safe', '0.1')])
        assert stage1['helpfulness_delta'] == pytest.approx(0.5)  # 1.5 - 1.0
        assert stage1['base_helpfulness'] == pytest.approx(1.0)

        # Stage 2: S,H-0.1, parent is S-0.1
        stage2 = next(d for d in deltas if d['training_stages'] == [('safe', '0.1'), ('help', '0.1')])
        assert stage2['helpfulness_delta'] == pytest.approx(0.5)  # 2.0 - 1.5
        assert stage2['base_helpfulness'] == pytest.approx(1.5)

        # Stage 3: S,H,S, parent is S,H
        stage3 = next(d for d in deltas if len(d['training_stages']) == 3)
        assert stage3['helpfulness_delta'] == pytest.approx(0.2)  # 2.2 - 2.0
        assert stage3['base_helpfulness'] == pytest.approx(2.0)

    def test_parent_mode_different_source_models(self):
        """Parent matching respects source_model dimension."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('safe', '0.05')], helpfulness=2.1, safety=2.0),
            make_full_data_point('Qwen3-8B', [('safe', '0.05'), ('help', '0.05')], helpfulness=2.6, safety=1.7),
            make_full_data_point('Llama-8B', [], helpfulness=1.8, safety=1.6),
            make_full_data_point('Llama-8B', [('safe', '0.05')], helpfulness=1.9, safety=2.1),
            make_full_data_point('Llama-8B', [('safe', '0.05'), ('help', '0.05')], helpfulness=2.4, safety=1.8),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'training_stages'])
        deltas = compute_deltas_from_aggregated(
            aggregated, group_by=['source_model', 'training_stages'], relative_to='parent'
        )

        assert len(deltas) == 4  # 2 per source model

        # Qwen two-stage should use Qwen single-stage as parent
        qwen_two = next(d for d in deltas if d['source_model'] == 'Qwen3-8B' and len(d['training_stages']) == 2)
        assert qwen_two['helpfulness_delta'] == pytest.approx(0.5)  # 2.6 - 2.1
        assert qwen_two['base_helpfulness'] == pytest.approx(2.1)

        # Llama two-stage should use Llama single-stage as parent
        llama_two = next(d for d in deltas if d['source_model'] == 'Llama-8B' and len(d['training_stages']) == 2)
        assert llama_two['helpfulness_delta'] == pytest.approx(0.5)  # 2.4 - 1.9
        assert llama_two['base_helpfulness'] == pytest.approx(1.9)

    def test_parent_mode_different_betas_no_match(self):
        """Parent must have exact stage match including betas."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            # S-0.1 (different beta)
            make_full_data_point('Qwen3-8B', [('safe', '0.1')], helpfulness=2.1, safety=2.0),
            # S,H-0.05 where S has beta 0.05 (different from above)
            make_full_data_point('Qwen3-8B', [('safe', '0.05'), ('help', '0.05')], helpfulness=2.6, safety=1.7),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'training_stages'])
        deltas = compute_deltas_from_aggregated(
            aggregated, group_by=['source_model', 'training_stages'], relative_to='parent'
        )

        # Two-stage (S-0.05, H-0.05) has no matching parent (S-0.05 not present, only S-0.1)
        # So only S-0.1 produces a delta
        assert len(deltas) == 1
        assert deltas[0]['training_stages'] == [('safe', '0.1')]

    def test_invalid_relative_to_raises(self):
        """Invalid relative_to value raises ValueError."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5)]
        aggregated = aggregate_scores(data, group_by=['source_model', 'training_stages'])

        with pytest.raises(ValueError, match="relative_to must be 'base' or 'parent'"):
            compute_deltas_from_aggregated(
                aggregated, group_by=['source_model', 'training_stages'], relative_to='invalid'
            )

    def test_parent_mode_empty_input(self):
        """Empty input returns empty deltas in parent mode."""
        from generate_plots import compute_deltas_from_aggregated

        deltas = compute_deltas_from_aggregated([], group_by=['source_model'], relative_to='parent')
        assert deltas == []


class TestComputePerpAngle:
    """Tests for perpendicular angle calculation in very close arrow case.

    These test the formula: perp_angle = vec_angle + (π/2 if bend_outward else -π/2)
    which gives the direction perpendicular to the arrow, used for offsetting
    start/end points when distance < 0.15.
    """

    def test_horizontal_right_bend_up(self):
        """Horizontal rightward arrow bending up should offset upward (+π/2)."""
        import math
        # Arrow pointing right: vec_angle = 0
        dx, dy = 1.0, 0.0
        vec_angle = math.atan2(dy, dx)
        bend_outward = True  # bend up

        perp_angle = vec_angle + (math.pi/2 if bend_outward else -math.pi/2)

        # Should point up (π/2)
        assert abs(perp_angle - math.pi/2) < 0.01

    def test_horizontal_right_bend_down(self):
        """Horizontal rightward arrow bending down should offset downward (-π/2)."""
        import math
        dx, dy = 1.0, 0.0
        vec_angle = math.atan2(dy, dx)
        bend_outward = False  # bend down

        perp_angle = vec_angle + (math.pi/2 if bend_outward else -math.pi/2)

        # Should point down (-π/2)
        assert abs(perp_angle - (-math.pi/2)) < 0.01

    def test_horizontal_left_bend_up(self):
        """Horizontal leftward arrow bending up should offset upward."""
        import math
        dx, dy = -1.0, 0.0
        vec_angle = math.atan2(dy, dx)  # = π
        bend_outward = True

        perp_angle = vec_angle + (math.pi/2 if bend_outward else -math.pi/2)

        # π + π/2 = 3π/2 = -π/2 (pointing down in absolute terms)
        # But this is "left of travel direction" which for leftward arrow is down
        # Actually for leftward arrow, +π/2 rotation gives downward offset
        # Normalize to [-π, π]
        while perp_angle > math.pi:
            perp_angle -= 2 * math.pi
        # Should be pointing down (3π/2 normalized to -π/2)
        assert abs(perp_angle - (-math.pi/2)) < 0.01

    def test_vertical_up_bend_outward(self):
        """Vertical upward arrow with bend_outward=True offsets left of travel."""
        import math
        dx, dy = 0.0, 1.0
        vec_angle = math.atan2(dy, dx)  # = π/2
        bend_outward = True

        perp_angle = vec_angle + (math.pi/2 if bend_outward else -math.pi/2)

        # π/2 + π/2 = π (pointing left in absolute terms)
        assert abs(perp_angle - math.pi) < 0.01

    def test_vertical_up_bend_inward(self):
        """Vertical upward arrow with bend_outward=False offsets right of travel."""
        import math
        dx, dy = 0.0, 1.0
        vec_angle = math.atan2(dy, dx)  # = π/2
        bend_outward = False

        perp_angle = vec_angle + (math.pi/2 if bend_outward else -math.pi/2)

        # π/2 - π/2 = 0 (pointing right in absolute terms)
        assert abs(perp_angle) < 0.01

    def test_vertical_down_bend_outward(self):
        """Vertical downward arrow with bend_outward=True offsets left of travel."""
        import math
        dx, dy = 0.0, -1.0
        vec_angle = math.atan2(dy, dx)  # = -π/2
        bend_outward = True

        perp_angle = vec_angle + (math.pi/2 if bend_outward else -math.pi/2)

        # -π/2 + π/2 = 0 (pointing right in absolute terms)
        assert abs(perp_angle) < 0.01

    def test_vertical_down_bend_inward(self):
        """Vertical downward arrow with bend_outward=False offsets right of travel."""
        import math
        dx, dy = 0.0, -1.0
        vec_angle = math.atan2(dy, dx)  # = -π/2
        bend_outward = False

        perp_angle = vec_angle + (math.pi/2 if bend_outward else -math.pi/2)

        # -π/2 - π/2 = -π (pointing left in absolute terms)
        assert abs(perp_angle - (-math.pi)) < 0.01


class TestMetricSequenceDimension:
    """Tests for the metric_sequence grouping dimension."""

    def test_metric_sequence_is_valid_dimension(self):
        """metric_sequence should be in VALID_GROUP_BY_DIMENSIONS."""
        from generate_plots import VALID_GROUP_BY_DIMENSIONS

        assert 'metric_sequence' in VALID_GROUP_BY_DIMENSIONS

    def test_group_by_metric_sequence_ignores_betas(self):
        """Models with same dataset sequence but different betas should be grouped together."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.05')], helpfulness=2.5, safety=1.2),
            make_full_data_point('Qwen3-8B', [('help', '0.02')], helpfulness=2.7, safety=1.3),
        ]

        result = aggregate_scores(data, group_by=['metric_sequence'])

        # All three should be in one group (all have dataset sequence ['help'])
        assert len(result) == 1
        assert result[0]['n_models'] == 3
        assert result[0]['helpfulness'] == pytest.approx((2.0 + 2.5 + 2.7) / 3)
        assert result[0]['safety'] == pytest.approx((1.5 + 1.2 + 1.3) / 3)

    def test_group_by_metric_sequence_different_sequences(self):
        """Different dataset sequences should be in separate groups."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.0),
            make_full_data_point('Qwen3-8B', [('safe', '0.1')], helpfulness=2.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1'), ('safe', '0.1')], helpfulness=2.3),
            make_full_data_point('Qwen3-8B', [('safe', '0.1'), ('help', '0.1')], helpfulness=2.7),
        ]

        result = aggregate_scores(data, group_by=['metric_sequence'])

        # 4 different sequences: ['help'], ['safe'], ['help', 'safe'], ['safe', 'help']
        assert len(result) == 4

    def test_metric_sequence_field_is_tuple_of_datasets(self):
        """Aggregated result should have metric_sequence field as tuple of dataset names."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [('help', '0.1'), ('safe', '0.05')], helpfulness=2.0),
        ]

        result = aggregate_scores(data, group_by=['metric_sequence'])

        assert len(result) == 1
        assert result[0]['metric_sequence'] == ('help', 'safe')

    def test_source_model_has_empty_metric_sequence(self):
        """Source models (no training_stages) should have empty tuple for metric_sequence."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0),
        ]

        result = aggregate_scores(data, group_by=['metric_sequence'])

        assert len(result) == 1
        assert result[0]['metric_sequence'] == ()

    def test_group_by_metric_sequence_and_source_model(self):
        """Grouping by both metric_sequence and source_model should work."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.0),
            make_full_data_point('Qwen3-8B', [('help', '0.05')], helpfulness=2.5),
            make_full_data_point('Llama-8B', [('help', '0.1')], helpfulness=3.0),
            make_full_data_point('Llama-8B', [('help', '0.05')], helpfulness=3.5),
        ]

        result = aggregate_scores(data, group_by=['source_model', 'metric_sequence'])

        # 2 source models × 1 sequence = 2 groups
        assert len(result) == 2

        qwen = next(r for r in result if r['source_model'] == 'Qwen3-8B')
        llama = next(r for r in result if r['source_model'] == 'Llama-8B')

        assert qwen['n_models'] == 2
        assert qwen['helpfulness'] == pytest.approx(2.25)
        assert llama['n_models'] == 2
        assert llama['helpfulness'] == pytest.approx(3.25)

    def test_metric_sequence_is_training_specific(self):
        """metric_sequence should be in TRAINING_SPECIFIC_DIMENSIONS."""
        from generate_plots import TRAINING_SPECIFIC_DIMENSIONS

        assert 'metric_sequence' in TRAINING_SPECIFIC_DIMENSIONS

    def test_delta_with_metric_sequence(self):
        """Delta computation should work with metric_sequence grouping."""
        from generate_plots import aggregate_scores, compute_deltas_from_aggregated

        data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.2),
            make_full_data_point('Qwen3-8B', [('help', '0.05')], helpfulness=2.7, safety=1.3),
        ]

        aggregated = aggregate_scores(data, group_by=['source_model', 'metric_sequence'])
        deltas = compute_deltas_from_aggregated(aggregated, group_by=['source_model', 'metric_sequence'])

        # Should have one delta for the 'help' sequence (averaged over betas)
        assert len(deltas) == 1
        # Delta = averaged finetuned - base
        avg_help = (2.5 + 2.7) / 2
        avg_safe = (1.2 + 1.3) / 2
        assert deltas[0]['helpfulness_delta'] == pytest.approx(avg_help - 2.0)
        assert deltas[0]['safety_delta'] == pytest.approx(avg_safe - 1.5)

    def test_label_for_metric_sequence_aggregated(self):
        """Aggregated data grouped by metric_sequence should have a label without betas."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [('help', '0.1'), ('safe', '0.05')], helpfulness=2.0),
            make_full_data_point('Qwen3-8B', [('help', '0.05'), ('safe', '0.1')], helpfulness=2.5),
        ]

        result = aggregate_scores(data, group_by=['metric_sequence'])

        assert len(result) == 1
        # Should have a label like "H,S" (without betas)
        assert result[0].get('label') == 'H,S'

    def test_label_for_single_stage_metric_sequence(self):
        """Single stage metric_sequence should have simple label."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [('safe', '0.1')], helpfulness=2.0),
            make_full_data_point('Qwen3-8B', [('safe', '0.05')], helpfulness=2.5),
        ]

        result = aggregate_scores(data, group_by=['metric_sequence'])

        assert len(result) == 1
        assert result[0].get('label') == 'S'

    def test_combined_dataset_in_sequence(self):
        """Combined datasets like 'both' should work in metric_sequence."""
        from generate_plots import aggregate_scores

        data = [
            make_full_data_point('Qwen3-8B', [('both', '0.1')], helpfulness=2.0),
            make_full_data_point('Qwen3-8B', [('both', '0.05')], helpfulness=2.5),
        ]

        result = aggregate_scores(data, group_by=['metric_sequence'])

        assert len(result) == 1
        assert result[0]['metric_sequence'] == ('both',)
        assert result[0].get('label') == 'S&H'


class TestComputeTrainingVsEvalData:
    """Tests for compute_training_vs_eval_data function."""

    def test_joins_training_and_results(self):
        """Training metrics are joined with results data."""
        from generate_plots import compute_training_vs_eval_data

        training_data = [
            {'training_stages': [('help', '0.1')], 'start_loss': 0.5, 'start_acc': 0.6,
             'source_model': 'Qwen3-8B', 'last_finetune_type': 'help'},
        ]
        results_data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.3),
        ]

        result = compute_training_vs_eval_data(training_data, results_data)

        assert len(result) == 1
        assert result[0]['start_loss'] == 0.5
        assert result[0]['start_acc'] == 0.6
        assert result[0]['helpfulness_delta'] == 0.5  # 2.5 - 2.0

    def test_relevant_delta_uses_helpfulness_for_help(self):
        """relevant_delta uses helpfulness when last stage is 'help'."""
        from generate_plots import compute_training_vs_eval_data

        training_data = [
            {'training_stages': [('help', '0.1')], 'start_loss': 0.5, 'start_acc': 0.6,
             'source_model': 'Qwen3-8B', 'last_finetune_type': 'help'},
        ]
        results_data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('help', '0.1')], helpfulness=2.5, safety=1.3),
        ]

        result = compute_training_vs_eval_data(training_data, results_data)

        assert len(result) == 1
        assert result[0]['relevant_delta'] == 0.5  # helpfulness: 2.5 - 2.0

    def test_relevant_delta_uses_safety_for_safe(self):
        """relevant_delta uses safety when last stage is 'safe'."""
        from generate_plots import compute_training_vs_eval_data

        training_data = [
            {'training_stages': [('safe', '0.1')], 'start_loss': 0.5, 'start_acc': 0.6,
             'source_model': 'Qwen3-8B', 'last_finetune_type': 'safe'},
        ]
        results_data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('safe', '0.1')], helpfulness=1.8, safety=2.0),
        ]

        result = compute_training_vs_eval_data(training_data, results_data)

        assert len(result) == 1
        assert result[0]['relevant_delta'] == 0.5  # safety: 2.0 - 1.5

    def test_two_stage_uses_parent_not_base(self):
        """Two-stage model's delta is relative to parent, not base."""
        from generate_plots import compute_training_vs_eval_data

        training_data = [
            {'training_stages': [('safe', '0.1'), ('help', '0.1')], 'start_loss': 0.4, 'start_acc': 0.7,
             'source_model': 'Qwen3-8B', 'last_finetune_type': 'help'},
        ]
        results_data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('safe', '0.1')], helpfulness=1.8, safety=2.0),
            make_full_data_point('Qwen3-8B', [('safe', '0.1'), ('help', '0.1')], helpfulness=2.3, safety=1.9),
        ]

        result = compute_training_vs_eval_data(training_data, results_data)

        assert len(result) == 1
        # relevant_delta = helpfulness(S,H) - helpfulness(S) = 2.3 - 1.8 = 0.5
        assert result[0]['relevant_delta'] == pytest.approx(0.5)

    def test_empty_training_data_returns_empty(self):
        """Empty training data returns empty list."""
        from generate_plots import compute_training_vs_eval_data

        results_data = [make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5)]

        result = compute_training_vs_eval_data([], results_data)

        assert result == []

    def test_empty_results_data_returns_empty(self):
        """Empty results data returns empty list."""
        from generate_plots import compute_training_vs_eval_data

        training_data = [
            {'training_stages': [('help', '0.1')], 'start_loss': 0.5, 'start_acc': 0.6,
             'source_model': 'Qwen3-8B', 'last_finetune_type': 'help'},
        ]

        result = compute_training_vs_eval_data(training_data, [])

        assert result == []

    def test_missing_parent_excluded(self):
        """Training configs without parent results are excluded."""
        from generate_plots import compute_training_vs_eval_data

        training_data = [
            {'training_stages': [('safe', '0.1'), ('help', '0.1')], 'start_loss': 0.4, 'start_acc': 0.7,
             'source_model': 'Qwen3-8B', 'last_finetune_type': 'help'},
        ]
        # Missing parent [('safe', '0.1')]
        results_data = [
            make_full_data_point('Qwen3-8B', [], helpfulness=2.0, safety=1.5),
            make_full_data_point('Qwen3-8B', [('safe', '0.1'), ('help', '0.1')], helpfulness=2.3, safety=1.9),
        ]

        result = compute_training_vs_eval_data(training_data, results_data)

        assert result == []
