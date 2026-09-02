"""Shared out-of-sample scoring for the walk-forward experiments.

Both experiments emit predictions in the same long format -- one row per
(date, horizon) with a ``y`` label, a ``base`` reference probability and one
column per model -- so they are scored by this single implementation and their
numbers are directly comparable.

The headline metric is the **log-loss skill score** against the constant base
rate that the training window observed:

    skill = 1 - logloss(model) / logloss(base)

Zero means the model is worth exactly as much as knowing the historical up-rate
and nothing else; negative means it is worse than that. Accuracy is reported too
but is close to useless here -- with a base rate near 55% a model that always
says "up" scores 55% while carrying no information at all.

Because a 14-day horizon means 14 consecutive rows share almost all of their
outcome, pooled errors are heavily autocorrelated and a naive standard error
would be far too tight. Intervals come from a moving-block bootstrap over
calendar dates instead.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

BOOTSTRAP_BLOCK = 30      # days per block -- longer than the longest horizon
BOOTSTRAP_REPS  = 2000
EPS             = 1e-15


def skill(y, p, p_base) -> float:
    """Log-loss skill score vs the base rate: 1 = perfect, 0 = no better, < 0 = worse."""
    return 1.0 - log_loss(y, p, labels=[0, 1]) / log_loss(y, p_base, labels=[0, 1])


def _row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def block_bootstrap_skill(df: pd.DataFrame, model: str,
                          rng: np.random.Generator) -> tuple[float, float]:
    """Moving-block bootstrap 95% interval for the skill score.

    Within one horizon there is exactly one row per date, so a block of dates is a
    contiguous slice of rows. Resampling then reduces to summing precomputed
    per-row log-loss contributions over sampled slices -- no resampled frame is
    ever materialised, which is what makes 2000 reps tractable.
    """
    sub  = df.sort_values('date')
    y    = sub['y'].to_numpy(dtype=float)
    ll_m = _row_log_loss(y, sub[model].to_numpy(dtype=float))
    ll_b = _row_log_loss(y, sub['base'].to_numpy(dtype=float))

    n = len(y)
    if n < BOOTSTRAP_BLOCK * 2:
        return np.nan, np.nan

    cs_m = np.concatenate([[0.0], np.cumsum(ll_m)])
    cs_b = np.concatenate([[0.0], np.cumsum(ll_b)])
    n_blk  = max(1, n // BOOTSTRAP_BLOCK)
    starts = rng.integers(0, n - BOOTSTRAP_BLOCK + 1, size=(BOOTSTRAP_REPS, n_blk))
    tot_m  = (cs_m[starts + BOOTSTRAP_BLOCK] - cs_m[starts]).sum(axis=1)
    tot_b  = (cs_b[starts + BOOTSTRAP_BLOCK] - cs_b[starts]).sum(axis=1)

    vals = 1.0 - tot_m / tot_b
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score_table(df: pd.DataFrame, models: list[str], display: list[int],
                seed: int = 42, title: str = 'Pooled out-of-sample scores') -> dict:
    """Print and return the pooled score table for each display horizon."""
    rng     = np.random.default_rng(seed)
    summary = {}

    print('\n' + '=' * 96)
    print(f'{title}  (N={len(df)} predictions, '
          f'{pd.Timestamp(df["date"].min()).date()} -> {pd.Timestamp(df["date"].max()).date()})')
    print('=' * 96)

    for h in display:
        sub = df[df['horizon'] == h]
        if sub.empty:
            continue
        print(f'\n  horizon +{h}d   n={len(sub)}   realized up-rate={sub["y"].mean() * 100:.1f}%')
        print(f'    {"model":<12} {"log loss":>9} {"skill":>8} {"95% CI":>19} '
              f'{"Brier":>8} {"AUC":>7} {"acc":>7}')
        print(f'    {"-" * 12} {"-" * 9} {"-" * 8} {"-" * 19} {"-" * 8} {"-" * 7} {"-" * 7}')
        summary[f'+{h}d'] = {}
        for m in models:
            if m not in sub.columns:
                continue
            ll  = log_loss(sub['y'], sub[m], labels=[0, 1])
            sk  = skill(sub['y'], sub[m], sub['base'])
            br  = brier_score_loss(sub['y'], sub[m])
            auc = (roc_auc_score(sub['y'], sub[m])
                   if sub['y'].nunique() > 1 and sub[m].nunique() > 1 else np.nan)
            acc = ((sub[m] > 0.5).astype(int) == sub['y']).mean()
            ci  = (np.nan, np.nan) if m == 'base' else block_bootstrap_skill(sub, m, rng)
            ci_s = '        --       ' if np.isnan(ci[0]) else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'
            print(f'    {m:<12} {ll:>9.5f} {sk:>+8.4f} {ci_s:>19} '
                  f'{br:>8.5f} {auc:>7.4f} {acc:>7.3f}')
            summary[f'+{h}d'][m] = {
                'log_loss': ll, 'skill': sk, 'skill_ci95': list(ci),
                'brier': br, 'auc': float(auc), 'accuracy': float(acc),
            }
    return summary


def edge_buckets(df: pd.DataFrame, horizon: int, model: str) -> None:
    """Directional accuracy bucketed by the model's own edge -- winrate-matrix's table.

    A model with real signal shows realized up-rate rising monotonically across the
    buckets. A flat or scrambled column means the edge it reports is not information.
    """
    sub = df[df['horizon'] == horizon].copy()
    if sub.empty or model not in sub.columns:
        return
    sub['edge'] = (sub[model] - sub['base']) * 100
    bins = [-200, -10, -5, 0, 5, 10, 200]
    labs = ['< -10pp', '-10 to -5', '-5 to 0', '0 to +5', '+5 to +10', '> +10pp']
    sub['bucket'] = pd.cut(sub['edge'], bins=bins, labels=labs)
    base = sub['base'].mean() * 100

    print(f'\n  {model} @ +{horizon}d, bucketed by model edge  (mean base rate {base:.1f}%):')
    print(f'    {"bucket":<12} {"n":>5} {"avg edge":>10} {"realized up":>13} {"vs base":>10}')
    print(f'    {"-" * 12} {"-" * 5} {"-" * 10} {"-" * 13} {"-" * 10}')
    for lab in labs:
        b = sub[sub['bucket'] == lab]
        if b.empty:
            continue
        realized = b['y'].mean() * 100
        print(f'    {lab:<12} {len(b):>5} {b["edge"].mean():>+9.1f}pp '
              f'{realized:>12.1f}% {realized - base:>+9.1f}pp')


def per_fold_skill(df: pd.DataFrame, models: list[str], horizon: int) -> dict:
    """Skill per calendar-year fold -- shows whether a pooled number is stable or driven by one year."""
    sub, out = df[df['horizon'] == horizon], {}
    print(f'\n  Per-fold skill at +{horizon}d:')
    for year, g in sub.groupby('year'):
        if g['y'].nunique() < 2:
            continue
        row = {m: skill(g['y'], g[m], g['base']) for m in models if m != 'base' and m in g.columns}
        out[int(year)] = row
        print(f'    {int(year)}  n={len(g):>3}  ' + '  '.join(f'{m}={v:+.4f}' for m, v in row.items()))
    return out
