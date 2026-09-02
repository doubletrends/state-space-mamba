"""Render the one figure that summarises both experiments.

Reads the scored summaries written by ``exp_1_control.py`` and
``exp_2_direction.py`` and plots the headline number -- log-loss **skill** against
the base rate -- for the representative model of each family, with its moving-block
bootstrap 95% interval, at every display horizon.

The base rate is the horizontal line at skill = 0: knowing only the historical
up-rate and nothing else. A bar above it is a model that beat that; on or below it
is one that did not. The other variants in each family (``nb_temp``, ``logit_z``,
``mamba_dir``) sit in the JSON and tell the same story.

    Input : output/exp1_summary.json, output/exp2_summary.json
    Output: output/exp_skill_summary.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from ssm.config import OUTPUT

EXP1_JSON = OUTPUT / 'exp1_summary.json'
EXP2_JSON = OUTPUT / 'exp2_summary.json'
FIG_PATH  = OUTPUT / 'exp_skill_summary.png'

# (json file, model key, label, colour) -- one representative per family.
SERIES = [
    (EXP1_JSON, 'nb',        'winrate Naive Bayes (0 learned params)', '#4C72B0'),
    (EXP1_JSON, 'logit_raw', 'logistic regression on raw features',    '#55A868'),
    (EXP2_JSON, 'mamba_cal', 'MambaSSM directional (calibrated)',      '#C44E52'),
]


def _load_pooled(path):
    return json.loads(path.read_text(encoding='utf-8'))['pooled']


def main() -> None:
    pooled = {EXP1_JSON: _load_pooled(EXP1_JSON), EXP2_JSON: _load_pooled(EXP2_JSON)}

    horizons = [3, 7, 14]
    hkeys    = [f'+{h}d' for h in horizons]
    x        = np.arange(len(horizons))
    width    = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (path, key, label, colour) in enumerate(SERIES):
        skills, lo, hi = [], [], []
        for hk in hkeys:
            rec = pooled[path][hk][key]
            s   = rec['skill']
            ci  = rec.get('skill_ci95') or [s, s]
            skills.append(s)
            lo.append(s - ci[0])
            hi.append(ci[1] - s)
        off = (i - 1) * width
        ax.bar(x + off, skills, width, color=colour, label=label, zorder=3)
        ax.errorbar(x + off, skills, yerr=[lo, hi], fmt='none',
                    ecolor='#333333', elinewidth=1.1, capsize=3, zorder=4)

    ax.axhline(0.0, color='#333333', linewidth=1.2, zorder=2)
    ax.text(-0.35, 0.003, 'base rate — historical up-rate only',
            ha='left', va='bottom', fontsize=9, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([f'+{h} days' for h in horizons])
    ax.set_ylabel('log-loss skill vs base rate')
    ax.set_title('Measured out-of-sample directional signal — walk-forward 2019–2026\n'
                 'skill over the historical up-rate, from 0-parameter counting to the SSM',
                 fontsize=11)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.margins(x=0.05)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)
    print(f'Saved -> {FIG_PATH}')


if __name__ == '__main__':
    main()
