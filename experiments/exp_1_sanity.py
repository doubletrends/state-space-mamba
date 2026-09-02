"""Sanity checks for the experiment-1 encoder, run before trusting its verdict.

Experiment 1 reports that Naive Bayes over the winrate surfaces has *negative*
out-of-sample skill. That is a strong claim, and it has two boring explanations
that must be ruled out first:

  A. the encoding is simply wrong (a sign flip, a bad interpolation, a horizon
     misalignment) -- in which case the in-sample fit would also be unimpressive
  B. the port does not reproduce the sibling project -- in which case the surfaces
     being scored are not the ones winrate-matrix actually found

Check A refits on the full history and scores on that same history. In-sample,
Naive Bayes must show clearly positive skill that grows with the horizon: the
deviations were counted from these very rows, so the ranking has to work here even
if it is weak. A flat or negative column would indicate a bug. The gap between
this number and the (negative) walk-forward number is the size of the overfit.

Check B fits on full history and prints the current combined estimate per family,
which is directly comparable to ``python run.py --probe`` in the sibling repo.
"""

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ssm.config import OUTPUT, ensure_dirs
from ssm.winrate import Node, WinrateEncoder, labels, logodds
from ssm.winrate.data import load_panel

NODE_SPEC   = Path(__file__).parent / 'winrate_nodes_btc.json'
SANITY_LOG  = OUTPUT / 'exp1_sanity.log'
PIVOT       = 7


class _Tee:
    """Write to several streams at once, so the run is both shown and saved."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> None:
        for st in self.streams:
            st.write(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def main() -> None:
    spec     = json.loads(NODE_SPEC.read_text(encoding='utf-8'))
    horizons = spec['horizons']
    display  = spec['display_horizons']
    nodes    = [Node.from_dict(n) for n in spec['nodes']]

    panel = load_panel(spec['start_date'])
    enc   = WinrateEncoder(nodes, horizons).fit(panel, panel.index)   # full history, as the sibling does
    sel   = enc.best_per_family(PIVOT)
    ids   = list(sel.values())
    delta, z = enc.transform(panel, ids)
    y_all = labels(panel['close'], horizons)

    print('=' * 78)
    print('CHECK A -- in-sample skill (fit and scored on the same full history)')
    print('=' * 78)
    print('  If the encoding is correct these must be clearly positive: the bins were')
    print('  counted from these rows. A flat or negative number here means a bug.\n')
    print(f'    {"horizon":<9} {"n":>6} {"base ll":>9} {"nb ll":>9} {"skill":>9} {"AUC":>8}')
    print(f'    {"-" * 9} {"-" * 6} {"-" * 9} {"-" * 9} {"-" * 9} {"-" * 8}')

    for h in display:
        p_nb = enc.naive_bayes(z, h)
        y    = y_all[h]
        ok   = y.notna() & p_nb.notna()
        yv   = y[ok].astype(int).to_numpy()
        pv   = p_nb[ok].to_numpy()
        p0   = float(enc.base_rate_.loc[h]) / 100.0
        pb   = np.full(len(yv), p0)

        ll_b = log_loss(yv, pb, labels=[0, 1])
        ll_m = log_loss(yv, pv, labels=[0, 1])
        auc  = roc_auc_score(yv, pv)
        print(f'    +{h:<8d} {len(yv):>6} {ll_b:>9.5f} {ll_m:>9.5f} '
              f'{1 - ll_m / ll_b:>+9.4f} {auc:>8.4f}')

    print('\n' + '=' * 78)
    print(f'CHECK B -- current combined estimate  (compare to: run.py --probe)')
    print('=' * 78)
    last = panel.index[-1]
    print(f'  as of {last.date()}, one node per family, full-history surfaces\n')
    print(f'    {"family":<13} {"node":<26} {"current x":>11} '
          + ''.join(f'{"+" + str(h) + "d":>9}' for h in display))
    print(f'    {"-" * 13} {"-" * 26} {"-" * 11} ' + ''.join(f'{"-" * 9}' for _ in display))

    for fam, nid in sel.items():
        x  = float(enc.surfaces_[nid]['feature'].loc[last])
        dv = [delta.loc[last, (nid, h)] for h in display]
        cells = ''.join(f'{v:>+8.1f}pp' if pd.notna(v) else f'{"n/a":>9}' for v in dv)
        print(f'    {fam:<13} {nid:<26} {x:>11.4g} {cells}')

    print(f'\n    {"":<13} {"combined":<26} {"":>11} ', end='')
    for h in display:
        p = float(enc.naive_bayes(z, h).loc[last]) * 100
        print(f'{p:>8.1f}%', end='')
    print()
    print(f'    {"":<13} {"base rate":<26} {"":>11} ', end='')
    for h in display:
        print(f'{float(enc.base_rate_.loc[h]):>8.1f}%', end='')
    print('\n')


if __name__ == '__main__':
    ensure_dirs()
    with open(SANITY_LOG, 'w', encoding='utf-8') as fh, redirect_stdout(_Tee(sys.stdout, fh)):
        main()
    print(f'\nSaved -> {SANITY_LOG}')
