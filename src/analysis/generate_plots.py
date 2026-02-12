#!/usr/bin/env python3
"""
Generate visualization plots from ToolEmu evaluation results.

Usage:
    python src/analysis/generate_plots.py <results_dir> -o <output_dir>
"""

import argparse
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
from matplotlib.transforms import Bbox
from adjustText import adjust_text
from scipy import stats

# Thicker line for non-solid styles so dashes/dots are visible
DOT_LINESTYLE = {"pattern": (0, (1, 5)), "width": 3}
DASH_LINESTYLE = {"pattern": (0, (5, 3)), "width": 1.2}
SOLID_LINESTYLE = {"pattern": "solid", "width": 1.2}
MARKERS = ['D', 's', '^', 'v', 'P', 'X', '*', 'h', '<', '>', 'p']
TEXTURE_PATTERNS = {
    'help': ('Helpfulness', '|||'),
    'safe': ('Safety', 'oo'),
    'combined': ('S&H', 'xxx'),
}
COLORS = ['#c49f08', '#20a39e', '#ef5b5b']

# Marker sizes for different plot densities
VBIG_MARKER_SIZE, VBIG_BASE_MARKER_SIZE = 600, 700
BIG_MARKER_SIZE, BIG_BASE_MARKER_SIZE = 500, 700
MEDIUM_MARKER_SIZE, MEDIUM_BASE_MARKER_SIZE = 250, 400
SMALL_MARKER_SIZE, SMALL_BASE_MARKER_SIZE = 125, 350

# Triangle markers need to be slightly larger to appear the same size
TRIANGLE_MARKERS = {'^', 'v', '<', '>'}
TRIANGLE_SIZE_FACTOR = 1.5

# Neutral style for data aggregated over models (source_model is None)
NEUTRAL_STYLE = {'color': 'tab:blue', 'marker': 'o'}

# Default limits for delta plots
DELTA_X_LIMITS = (-1.4, 0.3)
DELTA_Y_LIMITS = (-0.1, 1.5)

matplotlib.rcParams['pdf.fonttype'] = 42 # proper fonts

def draw_finetuned_marker(ax, x, y, marker, size, hatch_pattern, hatch_color, zorder=3):
    """Draw a finetuned model marker with 3 layers: white fill, colored hatch, black outline.

    Returns tuple of (hatch_handle, outline_handle) for legend entries, or just outline if no hatch.
    """
    # Layer 0: White fill to make marker opaque
    ax.scatter(x, y, facecolors='white', edgecolors='none',
               marker=marker, s=size, zorder=zorder)
    # Layer 1: Colored hatch (hatch inherits edge color)
    hatch_handle = None
    if hatch_pattern:
        hatch_handle = ax.scatter(x, y, facecolors='none', edgecolors=hatch_color,
                   marker=marker, s=size, linewidths=0,
                   hatch=hatch_pattern, zorder=zorder)
    # Layer 2: Black outline on top
    outline_handle = ax.scatter(x, y, facecolors='none', edgecolors='black',
                        marker=marker, s=size, linewidths=1.0,
                        zorder=zorder + 0.1)
    # Return tuple for legend (matplotlib overlays tuple elements)
    if hatch_handle is not None:
        return (hatch_handle, outline_handle)
    return outline_handle

class HandlerFancyArrow(HandlerPatch):
    """Legend handler that draws a FancyArrowPatch."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        arrow = FancyArrowPatch(
            (xdescent, ydescent + height / 2),
            (xdescent + width, ydescent + height / 2),
            arrowstyle=orig_handle.get_arrowstyle(),
            color=orig_handle.get_edgecolor(),
            linestyle=orig_handle.get_linestyle(),
            linewidth=orig_handle.get_linewidth(),
            mutation_scale=orig_handle.get_mutation_scale(),
            transform=trans
        )
        return [arrow]


class BboxWrapper:
    """Wrapper for Bbox that provides get_window_extent() for adjustText compatibility."""

    def __init__(self, bbox: Bbox):
        self._bbox = bbox

    def get_window_extent(self, renderer=None):
        return self._bbox

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.analysis_utils import (
    short_dataset_name, load_results_data, aggregate_scores,
    compute_deltas_from_aggregated, filter_data, format_label,
    get_present_types, _is_ancestor, VALID_GROUP_BY_DIMENSIONS,
    TRAINING_SPECIFIC_DIMENSIONS, load_training_metrics, get_evaluator_short_name
)
from utils.model_name_utils import clean_model_name, model_sort_key

def assign_colors_markers(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Assign consistent colors and markers for each source model."""
    # Get unique source models, sorted for consistent assignment
    source_models = sorted(set(d['source_model'] for d in data), key=model_sort_key)

    styles = {}
    for i, source_model in enumerate(source_models):
        styles[source_model] = {
            'color': COLORS[i % len(COLORS)],
            'marker': MARKERS[i % len(MARKERS)],
        }

    return styles


def get_point_hatch(data: Dict[str, Any]) -> str:
    """Determine hatch pattern based on last finetune type. Uses TEXTURE_PATTERNS."""
    last_finetune_type = data.get('last_finetune_type')
    if last_finetune_type == 'both':
        return TEXTURE_PATTERNS['combined'][1]
    elif last_finetune_type == 'safe':
        return TEXTURE_PATTERNS['safe'][1]
    elif last_finetune_type == 'help':
        return TEXTURE_PATTERNS['help'][1]
    else:
        return ''  # Source model or unknown: solid


def get_arrow_linestyle(src: Dict[str, Any], dest: Dict[str, Any]) -> str:
    """Determine arrow linestyle based on the added finetune stage.

    Returns 'dotted' for combined, 'dashed' for safety, 'solid' for helpfulness.
    """
    src_stages = src.get('training_stages', [])
    dest_stages = dest.get('training_stages', [])

    # Find the new stage (dest should have one more stage than src)
    if len(dest_stages) <= len(src_stages):
        return 'solid'  # No new stage or invalid

    new_stage = dest_stages[len(src_stages)]
    new_dataset = new_stage[0]

    if 'both' in new_dataset:
        return DOT_LINESTYLE["pattern"]
    elif 'safe' in new_dataset:
        return DASH_LINESTYLE["pattern"]
    else:
        return SOLID_LINESTYLE["pattern"]


def compute_arrow_perp_angle(
    src: Dict[str, Any],
    dest: Dict[str, Any],
    all_data: List[Dict[str, Any]],
    x_field: str = 'helpfulness',
    y_field: str = 'safety'
) -> float:
    """Compute perpendicular angle for arrow bending, away from the data mean.

    Returns the angle (in radians) perpendicular to the arrow direction, chosen to
    bend away from the mean center of all data points. This handles horizontal,
    vertical, and diagonal arrows uniformly.
    """
    dx = dest[x_field] - src[x_field]
    dy = dest[y_field] - src[y_field]
    vec_angle = math.atan2(dy, dx)

    # Compute means from all data points
    all_x = [d[x_field] for d in all_data]
    all_y = [d[y_field] for d in all_data]
    mean_x = statistics.mean(all_x)
    mean_y = statistics.mean(all_y)

    # Arrow midpoint
    avg_x = (src[x_field] + dest[x_field]) / 2
    avg_y = (src[y_field] + dest[y_field]) / 2

    # Direction away from mean center
    away_x = avg_x - mean_x
    away_y = avg_y - mean_y

    # Two perpendicular directions: vec_angle ± 90°
    perp_plus = vec_angle + math.pi / 2

    # Choose the perpendicular that points more "away" from median
    # Dot product of perp_plus direction with away direction
    dot_plus = math.cos(perp_plus) * away_x + math.sin(perp_plus) * away_y

    if dot_plus >= 0:
        return perp_plus
    else:
        return vec_angle - math.pi / 2


def find_finetune_arrows(
    data: List[Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Find arrows between models where one is a direct finetune of another.

    Returns list of (parent, child) tuples where child has exactly one more
    training stage than parent, and all prior stages match exactly (including betas).
    """
    if len(data) < 2:
        return []

    arrows = []
    for parent in data:
        for child in data:
            if parent is child:
                continue
            # Direct finetune = ancestor with exactly one more stage
            if _is_ancestor(parent, child):
                parent_stages = parent.get('training_stages', [])
                child_stages = child.get('training_stages', [])
                if len(child_stages) == len(parent_stages) + 1:
                    arrows.append((parent, child))

    return arrows


def draw_line_of_best_fit(
    ax,
    data: List[Dict[str, Any]],
    x_field: str,
    y_field: str,
    plot_name: str,
) -> None:
    """Draw a line of best fit and print correlation statistics.

    Uses scipy.stats.linregress to compute the regression line and prints
    r-value and p-value to stdout.
    """
    x_vals = [d[x_field] for d in data]
    y_vals = [d[y_field] for d in data]

    if len(x_vals) < 2:
        print(f"  {plot_name}: Not enough data points for line of best fit")
        return

    # Compute linear regression
    result = stats.linregress(x_vals, y_vals)

    # Print statistics
    print(f"  {plot_name}: r={result.rvalue:.3f}, p={result.pvalue:.3e}, "
          f"slope={result.slope:.3f}, intercept={result.intercept:.3f}")

    # Draw the line across the current axis limits
    x_min, x_max = ax.get_xlim()
    line_x = [x_min, x_max]
    line_y = [result.slope * x + result.intercept for x in line_x]

    ax.plot(line_x, line_y, color='tab:red', linestyle='-', linewidth=1.2, alpha=0.7, zorder=1)


def _draw_arrows(
    ax,
    arrows: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    data: List[Dict[str, Any]],
    x_field: str,
    y_field: str,
) -> List:
    """Draw curved arrows between finetune pairs. Returns list of arrow annotations."""
    annotations = []
    for src, dest in arrows:
        dx = dest[x_field] - src[x_field]
        dy = dest[y_field] - src[y_field]
        distance = (dx**2 + dy**2) ** 0.5

        start_x, start_y = src[x_field], src[y_field]
        end_x, end_y = dest[x_field], dest[y_field]

        # Arrow styling based on training type
        linestyle = get_arrow_linestyle(src, dest)
        if linestyle == SOLID_LINESTYLE["pattern"]:
            lw = SOLID_LINESTYLE["width"]
        elif linestyle == DOT_LINESTYLE["pattern"]:
            lw = DOT_LINESTYLE["width"]
        else:
            lw = DASH_LINESTYLE["width"]

        arrowprops = dict(
            arrowstyle='-|>,head_length=0.6,head_width=0.4',
            color='black',
            lw=lw,
            linestyle=linestyle,
            shrinkA=12,
            shrinkB=12,
        )

        # Compute perpendicular angle that bends away from data center
        perp_angle = compute_arrow_perp_angle(src, dest, data, x_field, y_field)
        vec_angle = math.atan2(dy, dx)

        # ignore short distance case for now
        if False: # distance < 0.2:
            # Very close points: use arc with offset endpoints
            offset = 0.05
            offset_x = offset * math.cos(perp_angle)
            offset_y = offset * math.sin(perp_angle)
            start_x = src[x_field] + offset_x
            start_y = src[y_field] + offset_y
            end_x = dest[x_field] + offset_x
            end_y = dest[y_field] + offset_y

            exit_angle = math.degrees(perp_angle)
            arm = 40
            arrowprops['connectionstyle'] = f'arc,angleA={exit_angle:.0f},angleB={exit_angle:.0f},armA={arm:.0f},armB={arm:.0f},rad=3'
            arrowprops.pop('shrinkA', None)
            arrowprops.pop('shrinkB', None)
        else:
            # Add noise to avoid overlapping arrows
            noise_magnitude = 0
            random.seed(hash((src.get('model_name'), dest.get('model_name')))) # same bendiness for same arrow in all plots
            noise = random.uniform(-noise_magnitude, noise_magnitude)
            base_rad = (0.6 + noise) / (distance**(.68))
            rad = max(0.5, base_rad)

            angle_diff = perp_angle - vec_angle
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            if angle_diff > 0:
                rad = -rad
            arrowprops['connectionstyle'] = f'arc3,rad={rad:.2f}'

        ann = ax.annotate(
            '',
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=arrowprops,
            zorder=4
        )
        annotations.append(ann)

    return annotations


def _build_legend(
    ax,
    data: List[Dict[str, Any]],
    styles: Dict[str, Dict[str, Any]],
    arrows: bool,
    texture: bool,
    legend_loc: str,
    legend_marker_size: int,
    legend_fontsize: float = 16,
) -> None:
    """Build and add legend to the plot."""
    present = get_present_types(data)
    present_source_models = present['source_models']
    present_finetune_types = present['finetune_types']
    has_source_model_points = present['has_source_model_points']

    legend_handles = []
    legend_labels = []

    # Section: Source Model (only if multiple distinct source models)
    if len(present_source_models) > 1:
        legend_handles.append(Line2D([], [], color='none'))
        legend_labels.append(r'$\bf{Source\ Model}$')

        for source_model in sorted(present_source_models, key=model_sort_key):
            style = styles[source_model]
            handle = ax.scatter([], [], c=[style['color']], marker=style['marker'],
                               s=legend_marker_size, edgecolors='black', linewidths=0.5)
            legend_handles.append(handle)
            legend_labels.append(clean_model_name(source_model, format="long"))

    # Section: Last finetune (only if texture mode and types present)
    if texture and (present_finetune_types or has_source_model_points):
        legend_handles.append(Line2D([], [], color='none'))
        legend_labels.append('$\\bf{Last\\ DPO}$\n$\\bf{metric}$')

        if has_source_model_points:
            source_color = NEUTRAL_STYLE['color'] if not present_source_models else 'lightgray'
            handle = ax.scatter([], [], facecolors=source_color, edgecolors='black',
                               marker='o', s=legend_marker_size, linewidths=2.0)
            legend_handles.append(handle)
            legend_labels.append('Source model')

        for key in ['help', 'safe', 'combined']:
            if key in present_finetune_types:
                label, hatch = TEXTURE_PATTERNS[key]
                hatch_color = NEUTRAL_STYLE['color'] if not present_source_models else '#444444'
                handle = draw_finetuned_marker(ax, [], [], 'o', legend_marker_size, hatch, hatch_color)
                legend_handles.append(handle)
                legend_labels.append(label)

    # Section: Training metric (arrow line styles)
    if arrows and present_finetune_types:
        legend_handles.append(Line2D([], [], color='none'))
        legend_labels.append(r'$\bf{Training\ metric}$')
        if 'help' in present_finetune_types:
            legend_handles.append(FancyArrowPatch((0, 0), (1, 0), color='black',
                                                  linestyle=SOLID_LINESTYLE["pattern"],
                                                  arrowstyle='-|>,head_length=0.4,head_width=0.35',
                                                  mutation_scale=10, linewidth=SOLID_LINESTYLE["width"]))
            legend_labels.append('Helpfulness')
        if 'safe' in present_finetune_types:
            legend_handles.append(FancyArrowPatch((0, 0), (1, 0), color='black',
                                                  linestyle=DASH_LINESTYLE["pattern"],
                                                  arrowstyle='-|>,head_length=0.4,head_width=0.35',
                                                  mutation_scale=10, linewidth=DASH_LINESTYLE["width"]))
            legend_labels.append('Safety')
        if 'combined' in present_finetune_types:
            legend_handles.append(FancyArrowPatch((0, 0), (1, 0), color='black',
                                                  linestyle=DOT_LINESTYLE["pattern"],
                                                  arrowstyle='-|>,head_length=0.4,head_width=0.35',
                                                  mutation_scale=10, linewidth=DOT_LINESTYLE["width"]))
            legend_labels.append('S&H')

    ax.legend(handles=legend_handles, labels=legend_labels, loc=legend_loc,
              labelspacing=1.1, borderpad=0.8, fontsize=legend_fontsize,
              handler_map={FancyArrowPatch: HandlerFancyArrow()})


def generate_scatter_plot(
    data: List[Dict[str, Any]],
    output_dir: str,
    name: str,
    styles: Dict[str, Dict[str, Any]],
    figsize: tuple = (10, 8),
    dpi: int = 150,
    with_arrows: bool = False,
    hide_labels: bool = False,
    texture: bool = False,
    marker_size: int = 100,
    source_model_marker_size: int = 350,
    x_field: str = 'helpfulness',
    y_field: str = 'safety',
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    show_origin_lines: bool = False,
    x_limits: tuple = None,
    y_limits: tuple = None,
    adjust_text_factor: float = 1,
    legend_loc: str = 'best',
    include_x_axis: bool = True,
    include_y_axis: bool = True,
    include_line_of_best_fit: bool = False,
    legend_marker_size: int = BIG_MARKER_SIZE,
    legend_size_factor: float = 1,
) -> None:
    """Generate and save a scatter plot."""
    if not data:
        raise ValueError("No data to plot")

    output_path = os.path.join(output_dir, f'{name}.pdf')
    arrows = find_finetune_arrows(data) if with_arrows else None

    # Set thicker hatch lines for texture mode
    old_hatch_linewidth = plt.rcParams.get('hatch.linewidth', 1.0)
    if texture:
        plt.rcParams['hatch.linewidth'] = 2.0

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each point
    for d in data:
        # Use neutral style when source_model is None (aggregated over models)
        style = NEUTRAL_STYLE if d.get('source_model') is None else styles[d['source_model']]
        # Check if source model: either no training_stages or last_finetune_type is None
        is_source = d.get('last_finetune_type') is None
        size = source_model_marker_size if is_source else marker_size
        if style['marker'] in TRIANGLE_MARKERS:
            size *= TRIANGLE_SIZE_FACTOR
        zorder = 2 if is_source else 3  # source models drawn behind finetuned models

        if texture:
            hatch_pattern = get_point_hatch(d)
            if is_source:
                # Source models: solid filled with color
                ax.scatter(
                    d[x_field],
                    d[y_field],
                    c=[style['color']],
                    marker=style['marker'],
                    s=size,
                    edgecolors='black',
                    linewidths=2.0,
                    zorder=zorder
                )
            else:
                # Finetuned models: hollow with colored hatch pattern
                draw_finetuned_marker(ax, d[x_field], d[y_field], style['marker'],
                                      size, hatch_pattern, style['color'], zorder)
        else:
            # Original filled markers
            ax.scatter(
                d[x_field],
                d[y_field],
                c=[style['color']],
                marker=style['marker'],
                s=size,
                edgecolors='black',
                linewidths=2.0 if is_source else 0.5,
                zorder=zorder
            )

    # Draw arrows between finetune pairs
    arrow_annotations = _draw_arrows(ax, arrows, data, x_field, y_field) if arrows else []

    # Set axis limits early so transforms work correctly for adjust_text
    # Priority: explicit limits > standard plot limits > auto-computed
    is_standard_plot = (x_field == 'helpfulness' and y_field == 'safety')
    if x_limits:
        ax.set_xlim(*x_limits)
    elif is_standard_plot:
        ax.set_xlim(-0.1, 3.1)
    else:
        # Auto-compute limits with some padding
        all_x = [d[x_field] for d in data]
        x_range = max(all_x) - min(all_x) if all_x else 1
        x_pad = max(0.1, x_range * 0.1)
        ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)

    if y_limits:
        ax.set_ylim(*y_limits)
    elif is_standard_plot:
        ax.set_ylim(-0.1, 3.1)
    else:
        all_y = [d[y_field] for d in data]
        y_range = max(all_y) - min(all_y) if all_y else 1
        y_pad = max(0.1, y_range * 0.1)
        ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

    # Draw origin lines if requested (useful for delta plots)
    if show_origin_lines:
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

    # Draw line of best fit if requested
    if include_line_of_best_fit:
        draw_line_of_best_fit(ax, data, x_field, y_field, name)

    # Add data point labels using adjust_text to avoid overlaps
    texts = []
    x_points = []
    y_points = []
    if not hide_labels:
        for d in data:
            is_source = d.get('last_finetune_type') is None
            if not is_source:
                # Use 'label' field if present (for aggregated data), otherwise format_label
                label = d.get('label') or format_label(d)
                fontsize = (9 if marker_size <= SMALL_MARKER_SIZE else
                            13 if marker_size <= MEDIUM_MARKER_SIZE else
                            16 if marker_size <= BIG_MARKER_SIZE else 17)
                txt = ax.text(d[x_field], d[y_field], label,
                              fontsize=fontsize, alpha=0.8)
                texts.append(txt)
            x_points.append(d[x_field])
            y_points.append(d[y_field])

        # Adjust text positions to minimize overlaps (commented out for speed)
        # First create bounding boxes for markers manually to pass to adjust_text
        # Marker size in points^2, convert to data coordinates
        # sqrt(size) gives diameter in points, divide by 72 for inches, multiply by dpi ratio
        marker_bboxes = []
        for d in data:
            style = NEUTRAL_STYLE if d.get('source_model') is None else styles[d['source_model']]
            is_source = d.get('last_finetune_type') is None
            size = source_model_marker_size if is_source else marker_size
            if style['marker'] in TRIANGLE_MARKERS:
                size *= TRIANGLE_SIZE_FACTOR
            # Approximate radius in data coordinates (rough estimate)
            # size is in points^2, so sqrt(size) is diameter in points
            # Convert to data units using axis transform
            radius_pts = math.sqrt(size) / 2
            # Get data-to-display transform to estimate radius in data coords
            trans = ax.transData
            # Convert a small offset in display coords to data coords
            p0 = trans.inverted().transform((0, 0))
            p1 = trans.inverted().transform((radius_pts, radius_pts))
            radius_x = abs(p1[0] - p0[0])
            radius_y = abs(p1[1] - p0[1])
            bbox = Bbox.from_extents(
                d[x_field] - radius_x,
                d[y_field] - radius_y,
                d[x_field] + radius_x,
                d[y_field] + radius_y
            )
            marker_bboxes.append(BboxWrapper(bbox))
        # Combine marker bboxes and arrow annotations
        all_objects = marker_bboxes + arrow_annotations        
        adjust_text(
            texts,
            objects=all_objects if all_objects else None,
            ax=ax,
            iter_lim=800,
            expand=(1.1 * adjust_text_factor, 1.1 * adjust_text_factor),
            force_text=(30 * adjust_text_factor, 30 * adjust_text_factor),
            force_static=(30 * adjust_text_factor, 30 * adjust_text_factor),
            force_explode=(1.3 * adjust_text_factor, 1.3 * adjust_text_factor),
            arrowprops=dict(arrowstyle='-', color='darkgray', lw=0.6)
        )

    # Configure axes with labels (derive from field names if not provided)
    def format_field_label(field: str) -> str:
        """Format field name as readable label."""
        if field == 'helpfulness':
            return 'Helpfulness'
        elif field == 'safety':
            return 'Safety'
        elif field == 'helpfulness_delta':
            return 'Helpfulness Delta (post-trained - source)'
        elif field == 'safety_delta':
            return 'Safety Delta (post-trained - source)'
        else:
            return field.replace('_', ' ').title()

    axis_label_fontsize = 20 * legend_size_factor
    tick_fontsize = 15 * legend_size_factor
    if include_x_axis:
        ax.set_xlabel(x_label or format_field_label(x_field), fontsize=axis_label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_fontsize)
    else:
        ax.set_xlabel('')
        ax.set_xticks([])

    if include_y_axis:
        ax.set_ylabel(y_label or format_field_label(y_field), fontsize=axis_label_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
    else:
        ax.set_ylabel('')
        ax.set_yticks([])

    # Build and show legend if requested
    if legend_loc is not None:
        scaled_legend_marker_size = int(legend_marker_size * legend_size_factor**2)
        scaled_legend_fontsize = 16 * legend_size_factor
        _build_legend(ax, data, styles, arrows, texture, legend_loc,
                      scaled_legend_marker_size, scaled_legend_fontsize)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # Restore hatch linewidth
    plt.rcParams['hatch.linewidth'] = old_hatch_linewidth

    print(f"Plot saved to: {output_path}")


def compute_training_vs_eval_data(
    training_data: List[Dict[str, Any]],
    results_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Join training metrics with results data and compute parent-relative deltas.

    For each training config, computes the delta on the metric it was last trained on:
    - 'help' → helpfulness_delta relative to parent
    - 'safe' → safety_delta relative to parent

    Returns list of dicts with start_loss, start_acc, relevant_delta, etc.
    """
    if not training_data or not results_data:
        return []

    # Aggregate training metrics by training_stages
    agg_training = aggregate_scores(
        training_data,
        group_by=['training_stages'],
        numeric_fields=['start_loss', 'start_acc']
    )

    # Aggregate results and compute deltas relative to parent
    agg_results = aggregate_scores(
        results_data,
        group_by=['training_stages'],
    )
    deltas = compute_deltas_from_aggregated(agg_results, group_by=['training_stages'], relative_to='parent')

    # Index deltas by training_stages for lookup
    deltas_by_stages = {tuple(d['training_stages']): d for d in deltas}

    # Join training metrics with deltas
    joined = []
    for t in agg_training:
        stages = t.get('training_stages', [])
        if not stages:
            continue  # Skip source models

        stages_tuple = tuple(stages)
        if stages_tuple not in deltas_by_stages:
            continue  # No matching delta

        delta = deltas_by_stages[stages_tuple]

        # Select relevant delta based on last training stage
        last_dataset = stages[-1][0]
        if 'help' in last_dataset:
            relevant_delta = delta['helpfulness_delta']
        else:  # 'safe' or 'both'
            relevant_delta = delta['safety_delta']

        joined.append({
            **t,
            **delta,
            'relevant_delta': relevant_delta,
        })

    return joined

def generate_results_plots(
    data: List[Dict[str, Any]],
    output_dir: str,
    styles: Dict[str, Dict[str, Any]],
    name_suffix: str = '',
    figsize: tuple = (10, 8),
    dpi: int = 150,
) -> None:
    """Generate all plot types for the given data."""
    suffix = f'_{name_suffix}' if name_suffix else ''

    # Common args for all plots
    common = dict(output_dir=output_dir, figsize=figsize, dpi=dpi, styles=styles)

    # overall scatter plot, aggregate only by seeds
    group_by = ['source_model', 'training_stages']
    generate_scatter_plot(data, name=f'overall{suffix}', marker_size=170,
                          source_model_marker_size=MEDIUM_BASE_MARKER_SIZE, texture=False,
                          hide_labels=True, **common)

    # Filter by last finetune for {help, safe}, models combined
    for dataset in ['help', 'safe']:
        filtered = filter_data(data, last_finetune=dataset, include_intermediates=True,
                               disallowed_metrics=['both'], disallowed_num_stages=[3])
        aggregated = aggregate_scores(filtered, group_by=['training_stages'])
        adjust_text_factor = 2.8 if dataset == 'help' else 3.7
        generate_scatter_plot(aggregated, name=f'{dataset}_with_intermediates{suffix}',
                              texture=True, marker_size=VBIG_MARKER_SIZE, with_arrows=True, hide_labels=False,
                              source_model_marker_size=VBIG_BASE_MARKER_SIZE,
                              adjust_text_factor=adjust_text_factor, include_y_axis=(dataset=='help'),
                              legend_loc='best' if dataset == 'help' else None, **common)
        
    # Delta plot for all models combined. Omit line of best fit
    # because we're using the models-separate for that
    group_by = ['training_stages']
    aggregated = aggregate_scores(data, group_by)
    deltas = compute_deltas_from_aggregated(aggregated, group_by)
    generate_scatter_plot(deltas, name=f'delta_all{suffix}', adjust_text_factor=1.4,
                          legend_loc='upper right',
                          texture=True, marker_size=MEDIUM_MARKER_SIZE, include_y_axis=False,
                          x_field='helpfulness_delta', y_field='safety_delta', show_origin_lines=True,
                          x_limits=DELTA_X_LIMITS, y_limits=DELTA_Y_LIMITS, **common)

    # Delta plot for models separate
    group_by = ['source_model', 'training_stages']
    aggregated = aggregate_scores(data, group_by)
    deltas = compute_deltas_from_aggregated(aggregated, group_by)
    generate_scatter_plot(deltas, name=f'delta_by_model{suffix}', show_origin_lines=True,
                          texture=False, marker_size=170, legend_loc='upper right',
                          x_field='helpfulness_delta', y_field='safety_delta',
                          include_line_of_best_fit=True, hide_labels=True,
                          x_limits=DELTA_X_LIMITS, y_limits=DELTA_Y_LIMITS, **common)

def generate_all_plots(args):
    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all data 
    training_data = []
    if args.trained_models_dir:
        print(f"Loading training metrics from {args.trained_models_dir}...")
        training_data = load_training_metrics(args.trained_models_dir, args.data_dir)
        if training_data:
            print(f"Found {len(training_data)} trained models")
        else:
            print("No training metrics found")

    results_data = []
    if args.paths:
        for path in args.paths:
            data = load_results_data(path, args.data_dir)
            results_data.extend(data)
        if results_data:
            print(f"Loaded {len(results_data)} data points from {len(args.paths)} path(s)")
        else:
            print("No valid data found in the provided paths")

    if not training_data and not results_data:
        print("No data to plot")
        sys.exit(1)

    # Compute styles once for consistent colors/markers across all plots
    all_data_for_styles = training_data + results_data
    styles = assign_colors_markers(all_data_for_styles)

    # Generate training metrics plots
    if training_data:
        filtered_training = filter_data(training_data, disallowed_num_stages=[3], disallowed_metrics=['safe-most'])
        aggregated = aggregate_scores(filtered_training, group_by=['training_stages'],
                                      numeric_fields=['start_loss', 'start_acc'])
        generate_scatter_plot(aggregated, output_dir=args.output_dir, name='training_metrics',
                              styles=styles, figsize=figsize, dpi=args.dpi, x_field='start_loss',
                              y_field='start_acc', x_label='Start Loss', y_label='Start Accuracy',
                              texture=True, marker_size=BIG_MARKER_SIZE)

    # Generate training vs eval plot (specifically, start loss vs delta of relevant metric)
    if training_data and results_data:
        filtered_training = filter_data(training_data, disallowed_num_stages=[3], disallowed_metrics=['both'])
        filtered_results = filter_data(results_data, disallowed_num_stages=[3], disallowed_metrics=['both'])
        joined_data = compute_training_vs_eval_data(filtered_training, filtered_results)
        generate_scatter_plot(joined_data, output_dir=args.output_dir, name='loss_vs_delta',
                              styles=styles, figsize=figsize, dpi=args.dpi,
                              x_field='start_loss', y_field='relevant_delta',
                              x_label='Initial Loss', y_label='Delta of Last DPO Metric (vs Parent)',
                              texture=True, marker_size=BIG_MARKER_SIZE)
    if not results_data:
        print(f"Plots saved to {args.output_dir}/")
        return

    all_data = results_data

    # Plots averaged over evaluators
    print("Generating plots averaged over evaluators...")
    # aggregate over evaluators and seeds
    agg_data = aggregate_scores(all_data, group_by=['source_model', 'training_stages']) 
    generate_results_plots(agg_data, args.output_dir, styles, name_suffix='', figsize=figsize, dpi=args.dpi)

    # Per-evaluator plots
    print("Generating plots per evaluator...")
    evaluators = sorted(set(d['evaluator_model'] for d in all_data))
    for evaluator in evaluators:
        eval_suffix = get_evaluator_short_name(evaluator)
        eval_data = [d for d in all_data if d['evaluator_model'] == evaluator]
        if eval_data:
            print(f"Generating plots for {eval_suffix}...")
            # still aggregate over seeds
            agg_data = aggregate_scores(eval_data, group_by=['source_model', 'training_stages', 'evaluator_model'])
            generate_results_plots(agg_data, args.output_dir, styles, name_suffix=eval_suffix,
                                   figsize=figsize, dpi=args.dpi)

    print(f"Plots saved to {args.output_dir}/")

def main():
    parser = argparse.ArgumentParser(
        description="Generate scatter plots from ToolEmu evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'paths',
        nargs='*',
        type=str,
        help='Either a single directory (scans all reports) or one or more report JSON files'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='figs',
        help='Output directory for plots (default: figs)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/dpo_data',
        help='Directory containing DPO dataset files for inferring dataset types (default: data/dpo_data)'
    )
    parser.add_argument(
        '--figsize',
        type=str,
        default='10,8',
        help='Figure size as width,height (default: 10,8)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Output DPI (default: 150)'
    )
    parser.add_argument(
        '-t', '--trained-models-dir',
        type=str,
        default=None,
        help='Directory containing trained model subdirectories (for training metrics plot)'
    )

    generate_all_plots(parser.parse_args())

if __name__ == '__main__':
    main()
