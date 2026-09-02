"""Render the system diagram used at the top of the README.

A static schematic of what was built: data to features to the sparse disjoint-window
Mamba encoder to the two prediction heads, over a band naming the reproducible
walk-forward harness the model is measured with.

    Output: output/architecture.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ssm.config import OUTPUT

FIG_PATH = OUTPUT / 'architecture.png'

INK    = '#111827'
SUB    = '#4b5563'
GREY   = '#d1d5db'
PANEL  = '#f3f4f6'
BLUE   = '#4C72B0'
BLUE_F = '#e8eef7'
RED    = '#C44E52'
RED_F  = '#f7ecec'


def _box(ax, x, y, w, h, title, lines, *, face=PANEL, edge=GREY, tcol=INK, scol=SUB):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.014',
                                linewidth=1.2, edgecolor=edge, facecolor=face, zorder=3))
    ax.text(x + w / 2, y + h - 0.055, title, ha='center', va='top',
            fontsize=10.5, fontweight='bold', color=tcol, zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.115 - i * 0.058, ln, ha='center', va='top',
                fontsize=8.7, color=scol, zorder=4)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=13,
                                 linewidth=1.5, color='#9ca3af', zorder=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.text(0.035, 0.955, 'A selective state-space forecaster for BTC, built from scratch',
            fontsize=15, fontweight='bold', color=INK)
    ax.text(0.035, 0.895,
            'from-scratch Mamba SSM · sparse disjoint-window encoder designed against '
            'lag-copying · reproducible walk-forward harness', fontsize=9.5, color=SUB)

    y, h = 0.5, 0.26
    _box(ax, 0.035, y, 0.145, h, 'Data',
         ['CoinMetrics OHLCV', 'DXY · on-chain', '~4,000 daily rows'])
    _box(ax, 0.205, y, 0.165, h, 'Detrend + features',
         ['log-trend removed', '8 channels: vol, DXY,', '%R, halving, residual'])
    _box(ax, 0.395, y, 0.155, h, 'Sparse sampler',
         ['7 non-contiguous', '7-day windows', 'per anchor date t'])
    _box(ax, 0.575, y, 0.165, h, 'Mamba encoder',
         ['shared selective SSM', 'logcumsumexp scan', 'per-window states'],
         face=BLUE_F, edge=BLUE, tcol='#1f3b63', scol='#3b5a82')

    _box(ax, 0.775, 0.615, 0.20, 0.15, 'Level head', ['residual · 7 days · MSE'])
    _box(ax, 0.775, 0.40, 0.20, 0.15, 'Direction head', ['up / down · 14 days · BCE'],
         face=RED_F, edge=RED, tcol='#7a2e31', scol='#8a3a3d')

    ym = y + h / 2
    _arrow(ax, 0.180, ym, 0.203, ym)
    _arrow(ax, 0.370, ym, 0.393, ym)
    _arrow(ax, 0.550, ym, 0.573, ym)
    _arrow(ax, 0.740, ym, 0.773, 0.69)
    _arrow(ax, 0.740, ym, 0.773, 0.49)

    ax.add_patch(FancyBboxPatch((0.035, 0.145), 0.94, 0.235,
                                boxstyle='round,pad=0.008,rounding_size=0.014',
                                linewidth=1.2, edgecolor=GREY, facecolor='#fbfbfc', zorder=1))
    ax.text(0.055, 0.34, 'Reproducible walk-forward harness (2019–2026)',
            fontsize=10.5, fontweight='bold', color=INK)
    ax.text(0.055, 0.283,
            'calendar-year folds · 14-day embargo · per-fold refit & normalisation · '
            'moving-block bootstrap 95% intervals', fontsize=8.7, color=SUB)
    ax.text(0.055, 0.225,
            'nested baselines from 0-parameter counting to logistic regression · '
            'leak-free re-implementation of the winrate surfaces', fontsize=8.7, color=SUB)

    ax.add_patch(FancyBboxPatch((0.035, 0.03), 0.94, 0.08,
                                boxstyle='round,pad=0.006,rounding_size=0.02',
                                linewidth=0, facecolor=INK, zorder=3))
    ax.text(0.5, 0.07,
            'Design goal — make lag-copying unlearnable:  '
            'detrend · isolate each window · expose AR error',
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#ffffff', zorder=4)

    fig.savefig(FIG_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {FIG_PATH}')


if __name__ == '__main__':
    main()
