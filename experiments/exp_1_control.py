"""Experiment 1 -- the control: is a learned combiner better than counting?

Before spending any capacity on a state-space model, establish whether *any*
learned model beats the winrate-matrix Naive Bayes combiner out of sample. If a
logistic regression on the same conditional-deviation vector cannot beat plain
counting, no sequence model will either, and the honest conclusion is that the
sibling project already extracted the available structure.

The ladder of capacity, each model nested in the next:

    base       P(up) = p0(h)                     constant, 0 learned parameters
    nb         logit p0 + sum_i z_i              winrate-matrix combine(), 0 learned
    nb_temp    a + alpha * sum_i z_i             2 learned -- one global temperature
    logit_z    a + sum_i beta_i z_i              13 learned -- per-family weights
    logit_raw  a + sum_i beta_i x_i              13 learned -- raw features, no surface

``nb_temp`` matters because Naive Bayes is known to be overconfident when its
inputs are correlated -- and they are: the sibling project's own findings flag the
RSI / Williams %R / stochastic families as near-duplicates. A fitted alpha < 1 is
the cheapest possible correction for that, and it is the smallest step that a
learned model can take beyond counting.

``logit_raw`` is the counterfactual for the deviation encoding itself: the same
learner on the same features without the winrate surface in between. The gap
between it and ``logit_z`` is what the surfaces are worth.

Protocol -- everything is refitted inside the walk-forward loop:

  * folds are calendar years; the model for fold Y is fitted only on rows < Y
  * the last ``max(horizons)`` training rows are embargoed, since their outcomes
    fall inside the test period
  * threshold grids, conditional counts, base rates, per-family node selection
    and regression coefficients are all refitted per fold
  * the regularisation strength is chosen on an inner chronological split of the
    training window, never on test data

Predictions from all folds are pooled into one out-of-sample set and scored.
Because horizons overlap, the pooled errors are strongly autocorrelated, so the
skill score is reported with a moving-block bootstrap interval rather than a
naive standard error.

    Output: output/exp1_predictions.csv, output/exp1_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from _scoring import edge_buckets, per_fold_skill, score_table
from ssm.config import OUTPUT, ensure_dirs
from ssm.winrate import Node, WinrateEncoder, labels, logodds

NODE_SPEC     = Path(__file__).parent / 'winrate_nodes_btc.json'
PRED_CSV      = OUTPUT / 'exp1_predictions.csv'
SUMMARY_JSON  = OUTPUT / 'exp1_summary.json'

FIRST_TEST_YEAR = 2019     # ~4 years of training history before the first fold
PIVOT_HORIZON   = 7        # the horizon per-family node selection is made at
C_GRID          = [0.01, 0.1, 1.0, 10.0, 100.0, 1e4]
INNER_VAL_FRAC  = 0.20     # tail of the training window used to pick C
SEED            = 42

MODELS          = ['base', 'nb', 'nb_temp', 'logit_z', 'logit_raw']


# -- models --------------------------------------------------------------------

def _fit_logistic(X_tr, y_tr, embargo: int, seed: int = SEED):
    """L2 logistic with C picked on an inner chronological tail of the training window.

    The inner split carries its own embargo for the same reason the outer one does:
    the last rows of the inner-train block have outcomes inside the inner-val block.
    """
    n_val = max(60, int(len(X_tr) * INNER_VAL_FRAC))
    if len(X_tr) - n_val - embargo < 100:
        best_c = 1.0
    else:
        cut = len(X_tr) - n_val
        Xa, ya = X_tr[:cut - embargo], y_tr[:cut - embargo]
        Xb, yb = X_tr[cut:], y_tr[cut:]
        best_c, best_ll = 1.0, np.inf
        if len(np.unique(ya)) > 1 and len(np.unique(yb)) > 1:
            for c in C_GRID:
                m = LogisticRegression(C=c, max_iter=2000, random_state=seed).fit(Xa, ya)
                ll = log_loss(yb, m.predict_proba(Xb)[:, 1], labels=[0, 1])
                if ll < best_ll:
                    best_c, best_ll = c, ll
    model = LogisticRegression(C=best_c, max_iter=2000, random_state=seed).fit(X_tr, y_tr)
    return model, best_c


def _predict_fold(enc, z_tr, z_te, x_tr, x_te, y_tr, horizon, embargo):
    """All five models for one horizon of one fold -> {name: test probabilities}."""
    h_lo = logodds(enc.base_rate_.to_numpy(dtype=float))[enc.horizons.index(horizon)]
    p0   = float(enc.base_rate_.loc[horizon]) / 100.0

    sum_tr = z_tr.sum(axis=1).to_numpy()[:, None]
    sum_te = z_te.sum(axis=1).to_numpy()[:, None]

    out   = {'base': np.full(len(z_te), p0)}
    diag  = {}

    out['nb'] = 1.0 / (1.0 + np.exp(-(h_lo + sum_te.ravel())))

    if len(np.unique(y_tr)) > 1:
        m_t, c_t = _fit_logistic(sum_tr, y_tr, embargo)
        out['nb_temp'] = m_t.predict_proba(sum_te)[:, 1]
        diag['nb_temp_alpha'] = float(m_t.coef_[0][0])
        diag['nb_temp_C']     = c_t

        m_z, c_z = _fit_logistic(z_tr.to_numpy(), y_tr, embargo)
        out['logit_z'] = m_z.predict_proba(z_te.to_numpy())[:, 1]
        diag['logit_z_C']     = c_z
        diag['logit_z_coef']  = {c[0]: float(v) for c, v in zip(z_tr.columns, m_z.coef_[0])}

        m_r, c_r = _fit_logistic(x_tr, y_tr, embargo)
        out['logit_raw'] = m_r.predict_proba(x_te)[:, 1]
        diag['logit_raw_C'] = c_r
    else:
        for k in ('nb_temp', 'logit_z', 'logit_raw'):
            out[k] = out['base'].copy()
    return out, diag


# -- walk-forward --------------------------------------------------------------

def run_walk_forward(panel: pd.DataFrame, nodes: list[Node], horizons: list[int]
                     ) -> tuple[pd.DataFrame, list[dict]]:
    embargo = max(horizons)
    y_all   = labels(panel['close'], horizons)
    years   = sorted({d.year for d in panel.index if d.year >= FIRST_TEST_YEAR})

    rows, fold_log = [], []
    for year in years:
        te_idx = panel.index[panel.index.year == year]
        tr_idx = panel.index[panel.index < te_idx[0]]
        if len(tr_idx) <= embargo + 400 or len(te_idx) == 0:
            continue
        tr_idx = tr_idx[:-embargo]                      # outcomes of the tail fall in test

        enc = WinrateEncoder(nodes, horizons).fit(panel, tr_idx)
        sel = enc.best_per_family(PIVOT_HORIZON)        # selection is part of the fit
        ids = list(sel.values())

        delta, z = enc.transform(panel, ids)
        raw = pd.DataFrame(
            {nid: enc.surfaces_[nid]['feature'].reindex(panel.index) for nid in ids},
            index=panel.index,
        )
        med  = raw.loc[tr_idx].median()
        std  = raw.loc[tr_idx].std().replace(0, 1.0)
        rawz = ((raw.fillna(med) - med) / std).to_numpy()

        pos_tr = panel.index.get_indexer(tr_idx)
        pos_te = panel.index.get_indexer(te_idx)

        fold = {'year': year, 'n_train': len(tr_idx), 'n_test': len(te_idx),
                'selected': sel, 'horizons': {}}

        for h in horizons:
            yv     = y_all[h].to_numpy(dtype=float)
            ok_tr  = pos_tr[~np.isnan(yv[pos_tr])]
            ok_te  = pos_te[~np.isnan(yv[pos_te])]
            if len(ok_tr) < 200 or len(ok_te) == 0:
                continue

            z_h   = z.xs(h, axis=1, level='horizon')
            preds, diag = _predict_fold(
                enc,
                z_h.iloc[ok_tr], z_h.iloc[ok_te],
                rawz[ok_tr], rawz[ok_te],
                yv[ok_tr].astype(int), h, embargo,
            )
            fold['horizons'][h] = diag

            for k, name in enumerate(panel.index[ok_te]):
                rows.append({
                    'date': name, 'year': year, 'horizon': h,
                    'y': int(yv[ok_te][k]),
                    **{m: float(p[k]) for m, p in preds.items()},
                })
        fold_log.append(fold)
        print(f'  fold {year}: train={len(tr_idx):5d}  test={len(te_idx):4d}  '
              f'nodes={len(ids)}  [{", ".join(sorted(sel.values()))}]')

    return pd.DataFrame(rows), fold_log


# -- entry point ---------------------------------------------------------------

def main() -> None:
    ensure_dirs()
    spec     = json.loads(NODE_SPEC.read_text(encoding='utf-8'))
    horizons = spec['horizons']
    display  = spec['display_horizons']
    nodes    = [Node.from_dict(n) for n in spec['nodes']]

    from ssm.winrate.data import load_panel
    panel = load_panel(spec['start_date'])
    print(f'Panel      : {len(panel)} rows  {panel.index[0].date()} -> {panel.index[-1].date()}')
    print(f'Nodes      : {len(nodes)} candidates over {len({n.family for n in nodes})} families')
    print(f'Horizons   : +{horizons[0]}d .. +{horizons[-1]}d   pivot=+{PIVOT_HORIZON}d')
    print(f'Protocol   : walk-forward by calendar year from {FIRST_TEST_YEAR}, '
          f'embargo={max(horizons)} rows\n')

    df, folds = run_walk_forward(panel, nodes, horizons)
    if df.empty:
        print('No predictions produced.')
        return

    df.to_csv(PRED_CSV, index=False)
    summary = score_table(df, MODELS, display, seed=SEED)

    for h in display:
        edge_buckets(df, h, 'nb')
        edge_buckets(df, h, 'logit_z')

    per_fold = per_fold_skill(df, MODELS, PIVOT_HORIZON)

    alphas = {f['year']: f['horizons'].get(PIVOT_HORIZON, {}).get('nb_temp_alpha')
              for f in folds if PIVOT_HORIZON in f.get('horizons', {})}
    print(f'\n  Fitted Naive Bayes temperature alpha @ +{PIVOT_HORIZON}d by fold '
          f'(1.0 = trust Naive Bayes as-is):')
    print('    ' + '  '.join(f'{y}={a:+.3f}' for y, a in alphas.items() if a is not None))

    SUMMARY_JSON.write_text(json.dumps({
        'protocol': {
            'first_test_year': FIRST_TEST_YEAR, 'pivot_horizon': PIVOT_HORIZON,
            'embargo': max(horizons), 'n_predictions': int(len(df)),
            'test_start': str(df['date'].min().date()), 'test_end': str(df['date'].max().date()),
        },
        'pooled': summary,
        'per_fold_skill': per_fold,
        'nb_temperature': {str(k): v for k, v in alphas.items()},
        'fold_selection': {str(f['year']): f['selected'] for f in folds},
    }, indent=2, default=float), encoding='utf-8')

    print(f'\nPredictions -> {PRED_CSV}')
    print(f'Summary     -> {SUMMARY_JSON}')


if __name__ == '__main__':
    main()
