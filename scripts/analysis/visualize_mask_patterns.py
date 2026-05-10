#!/usr/bin/env python3
"""Visualize all 7 mask strategies (M1-M7) from universal_mask.py.

Generates two figures:
1. mask_patterns_m1_m7.png — 7x3 heatmap grid + statistics table
2. mask_strategy_weights.png — horizontal bar chart of strategy weights
"""

import sys
import os

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from hftrainer.datasets.motion.motionhub.transforms.universal_mask import (
    m1_random_cell,
    m2_random_block,
    m3_temporal_contiguous,
    m4_joint_contiguous,
    m5_full_mask,
    m6_keyframe_sparse,
    m7_scattered_joint,
    NUM_JOINT_GROUPS,
    DEFAULT_STRATEGY_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T = 120
J = 23
SEEDS = [42, 123, 7]  # 3 example seeds per strategy
N_STATS_SAMPLES = 1000

STRATEGIES = [
    ('M1', 'm1_random_cell', m1_random_cell),
    ('M2', 'm2_random_block', m2_random_block),
    ('M3', 'm3_temporal_contiguous', m3_temporal_contiguous),
    ('M4', 'm4_joint_contiguous', m4_joint_contiguous),
    ('M5', 'm5_full_mask', m5_full_mask),
    ('M6', 'm6_keyframe_sparse', m6_keyframe_sparse),
    ('M7', 'm7_scattered_joint', m7_scattered_joint),
]

STRATEGY_DISPLAY = {
    'M1': 'random_cell',
    'M2': 'random_block',
    'M3': 'temporal_contiguous',
    'M4': 'joint_contiguous',
    'M5': 'full_mask',
    'M6': 'keyframe_sparse',
    'M7': 'scattered_joint',
}

# Joint labels: col 0 = "T" (translation), cols 1-22 = joint indices
JOINT_LABELS = ['T'] + [str(i) for i in range(1, 23)]

OUTDIR = os.path.join(PROJECT_ROOT, 'docs', 'figures')
os.makedirs(OUTDIR, exist_ok=True)


def generate_grid(strategy_fn, seed):
    """Generate a (T, J) binary mask grid for a given strategy and seed."""
    rng = np.random.RandomState(seed)
    grid = np.zeros((T, J), dtype=np.float32)
    strategy_fn(T, grid, rng)
    return grid


def compute_statistics():
    """Compute min/mean/max ratio and translation mask probability over N samples."""
    stats = {}
    for label, key, fn in STRATEGIES:
        ratios = []
        transl_mask_count = 0
        for i in range(N_STATS_SAMPLES):
            rng = np.random.RandomState(i)
            grid = np.zeros((T, J), dtype=np.float32)
            fn(T, grid, rng)
            ratio = grid.sum() / grid.size
            ratios.append(ratio)
            if grid[:, 0].sum() > 0:
                transl_mask_count += 1
        ratios = np.array(ratios)
        stats[label] = {
            'weight': DEFAULT_STRATEGY_WEIGHTS[key],
            'min': ratios.min(),
            'mean': ratios.mean(),
            'max': ratios.max(),
            'transl_prob': transl_mask_count / N_STATS_SAMPLES,
        }
    return stats


def make_main_figure(stats):
    """Create the 7x3 heatmap figure with statistics table."""
    # Custom colormap: white (0) -> dark blue (1)
    cmap = plt.cm.Blues

    n_rows = len(STRATEGIES)
    n_cols = len(SEEDS)

    fig = plt.figure(figsize=(18, 28))

    # Use gridspec: top part for heatmaps, bottom for stats table
    outer_gs = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[7, 1.2], hspace=0.08
    )

    # Heatmap grid
    heatmap_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=outer_gs[0], hspace=0.35, wspace=0.15
    )

    for row_idx, (label, key, fn) in enumerate(STRATEGIES):
        for col_idx, seed in enumerate(SEEDS):
            grid = generate_grid(fn, seed)
            ratio = grid.sum() / grid.size

            ax = fig.add_subplot(heatmap_gs[row_idx, col_idx])
            im = ax.imshow(
                grid, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                interpolation='nearest', origin='upper'
            )

            # Ratio annotation
            ax.text(
                0.98, 0.02, f'ratio={ratio:.2f}',
                transform=ax.transAxes, fontsize=8,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
            )

            # X-axis: joint group labels (show every label)
            ax.set_xticks(range(J))
            if row_idx == n_rows - 1:
                ax.set_xticklabels(JOINT_LABELS, fontsize=6, rotation=0)
                ax.set_xlabel('Joint groups', fontsize=8)
            else:
                ax.set_xticklabels([])

            # Y-axis: time frames (sparse ticks)
            yticks = [0, 29, 59, 89, 119]
            ax.set_yticks(yticks)
            if col_idx == 0:
                ax.set_yticklabels([str(y) for y in yticks], fontsize=7)
                ax.set_ylabel('Time frame', fontsize=8)
            else:
                ax.set_yticklabels([])

            # Title for top row
            if row_idx == 0:
                ax.set_title(f'Seed {seed}', fontsize=10, fontweight='bold')

            # Strategy label on left side
            if col_idx == 0:
                display_name = STRATEGY_DISPLAY[label]
                ax.annotate(
                    f'{label}: {display_name}',
                    xy=(-0.35, 0.5), xycoords='axes fraction',
                    fontsize=10, fontweight='bold',
                    ha='right', va='center', rotation=0,
                )

    # -----------------------------------------------------------------------
    # Statistics table at the bottom
    # -----------------------------------------------------------------------
    table_ax = fig.add_subplot(outer_gs[1])
    table_ax.axis('off')

    col_labels = ['Strategy', 'Weight', 'Min Ratio', 'Mean Ratio', 'Max Ratio', 'Transl Mask Prob']
    table_data = []
    for label, key, _ in STRATEGIES:
        s = stats[label]
        table_data.append([
            f'{label}: {STRATEGY_DISPLAY[label]}',
            f'{s["weight"]:.2f}',
            f'{s["min"]:.3f}',
            f'{s["mean"]:.3f}',
            f'{s["max"]:.3f}',
            f'{s["transl_prob"]:.2f}',
        ])

    table = table_ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(len(table_data)):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    table_ax.set_title(
        f'Summary Statistics (over {N_STATS_SAMPLES} samples, T={T}, J={J})',
        fontsize=11, fontweight='bold', pad=10,
    )

    # Main title
    fig.suptitle(
        f'M1\u2013M7 Mask Pattern Coverage (T={T}, J={J})',
        fontsize=16, fontweight='bold', y=0.995,
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)),
        cax=cbar_ax,
    )
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0 (keep)', '1 (mask)'])
    cbar.ax.tick_params(labelsize=9)

    outpath = os.path.join(OUTDIR, 'mask_patterns_m1_m7.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def make_weights_figure():
    """Create horizontal bar chart of strategy weights."""
    labels = []
    weights = []
    for label, key, _ in STRATEGIES:
        labels.append(f'{label}: {STRATEGY_DISPLAY[label]}')
        weights.append(DEFAULT_STRATEGY_WEIGHTS[key])

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(labels))
    colors = plt.cm.Blues(np.linspace(0.3, 0.85, len(labels)))

    bars = ax.barh(y_pos, weights, color=colors, edgecolor='#333333', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Sampling Weight', fontsize=12)
    ax.set_title('M1\u2013M7 Strategy Sampling Weights', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, w in zip(bars, weights):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{w:.0%}', va='center', ha='left', fontsize=10, fontweight='bold',
        )

    ax.set_xlim(0, max(weights) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    outpath = os.path.join(OUTDIR, 'mask_strategy_weights.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def main():
    print(f'Computing statistics over {N_STATS_SAMPLES} samples...')
    stats = compute_statistics()

    print('Generating main heatmap figure...')
    make_main_figure(stats)

    print('Generating weights bar chart...')
    make_weights_figure()

    print('Done!')


if __name__ == '__main__':
    main()
