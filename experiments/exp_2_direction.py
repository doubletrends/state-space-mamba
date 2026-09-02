"""Experiment 2 -- refit MambaSSM as a directional classifier and score it honestly.

Two changes to the model, and nothing else:

  1. TARGET. The production pipeline predicts the log-price residual *level* at
     t+1..t+7 under MSE. Its variance (0.407 over the full history) is ~330x the
     variance of a one-day change (0.0012), so nearly all of that loss is the model
     carrying the level forward rather than forecasting anything. Here the target is
     ``close[t+h] > close[t]`` -- bounded, roughly balanced, and exactly the quantity
     the companion winrate-matrix project estimates.

     The level target is also why the production run needed a 30-day input embargo:
     with the level as the target, copying the newest input is a strong solution, so
     the input had to be blinded to suppress it. Under a direction target, copying
     predicts nothing, so the embargo is no longer needed -- and dropping it matters,
     because ``corr(R[t+1], R[t-30]) = 0.947`` while ``var(R[t] - R[t-30])`` is 33x
     ``var(R[t] - R[t-1])``: the embargo was discarding most of the available signal.

  2. HORIZON. The production model predicts t+1..t+7 and is evaluated at t+1 -- the
     horizon where winrate-matrix finds the weakest conditional structure (MVRV:
     +12.6pp at +3d against +34.3pp at +14d). Here the head emits one logit per
     horizon over +1..+14d and every horizon is scored.

  The anchors change from ``[365, 180, 90, 75, 60, 45, 30]`` to
  ``[365, 180, 90, 60, 30, 14, 0]``: same window count, but the nearest window now
  ends on the anchor day itself instead of 30 days before it.

Everything else is held fixed against experiment 1 -- same walk-forward folds, same
embargo, same base rate, same scoring -- so the numbers sit in the same table as the
Naive Bayes and logistic baselines rather than in a separate one.

Feature normalisation is refitted per fold on training rows only. Stage 3 of the
production pipeline z-scores on the full dataset (``fit_mask`` is all-True), which
leaks the holdout into the input scale; that is corrected here.

    Output: output/exp2_predictions.csv, output/exp2_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch import nn

from _scoring import edge_buckets, per_fold_skill, score_table
from ssm.arch.mamba import create_direction_model
from ssm.config import OUTPUT, DWD_CSV, ensure_dirs
from ssm.data.loader import build_direction_windows
from ssm.winrate.data import load_panel

PRED_CSV     = OUTPUT / 'exp2_predictions.csv'
SUMMARY_JSON = OUTPUT / 'exp2_summary.json'
NODE_SPEC    = Path(__file__).parent / 'winrate_nodes_btc.json'

# The stage-3 feature set, taken from the DWD *before* stage 3's full-dataset z-score.
FEATURES = [
    'years_since_halving',
    'realized_vol_30',
    'dxy_ret_30',
    'dxy_ret_100',
    'short_percent_r',
    'long_percent_r',
    'wr_composite',
    'log_price_residual',
]

WINDOW_ANCHORS = [365, 180, 90, 60, 30, 14, 0]   # nearest window now ends on t itself
WINDOW_LEN     = 7

FIRST_TEST_YEAR = 2019
PIVOT_HORIZON   = 7
MODELS          = ['base', 'mamba_dir', 'mamba_cal']

EPOCHS         = 200
PATIENCE       = 20
LR             = 3e-4
BATCH_SIZE     = 32
WEIGHT_DECAY   = 1e-3
D_MODEL        = 32
N_LAYER        = 3
D_STATE        = 16
DROPOUT        = 0.1
INNER_VAL_FRAC = 0.20
SEED           = 42


# -- data ----------------------------------------------------------------------

def load_inputs(horizons: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame and close series on one shared index."""
    dwd   = pd.read_csv(DWD_CSV, index_col='Date', parse_dates=True)[FEATURES]
    panel = load_panel('2015-01-01')

    idx  = dwd.dropna().index.intersection(panel.index)
    return dwd.loc[idx], panel.loc[idx, 'close']


# -- training ------------------------------------------------------------------

def _masked_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE over the horizons whose outcome is realised; NaN horizons contribute nothing."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def train_fold(X_tr, y_tr, X_val, y_val, n_feat, n_horizons, device, seed=SEED):
    """Train one fold with early stopping on an inner chronological validation tail."""
    torch.manual_seed(seed)
    model     = create_direction_model(n_feat, n_horizons, device, len(WINDOW_ANCHORS),
                                       D_MODEL, N_LAYER, D_STATE, DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xt = torch.tensor(X_tr,  device=device)
    yt = torch.tensor(y_tr,  device=device)
    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, device=device)

    best_state, best_loss, patience, best_epoch = None, np.inf, 0, 0
    gen = torch.Generator().manual_seed(seed)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen).to(device)
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = _masked_bce(model(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(_masked_bce(model(Xv), yv))

        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, patience = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss, best_epoch


@torch.no_grad()
def predict_logits(model, X, device, batch: int = 512) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        chunk = torch.tensor(X[i:i + batch], device=device)
        out.append(model(chunk).cpu().numpy())
    return np.vstack(out)


def platt_scale(logit_val: np.ndarray, y_val: np.ndarray, logit_te: np.ndarray) -> np.ndarray:
    """Per-horizon Platt scaling fitted on the inner validation split.

    A probabilistic head that is confidently wrong scores worse than one that
    abstains, so an uncalibrated log loss conflates two different failures:
    "the ranking is wrong" and "the ranking is fine but the confidence is not".
    Refitting ``sigmoid(a * logit + b)`` on held-out rows removes the second, and
    what survives is attributable to the ranking alone. If the model has no signal,
    calibration shrinks it toward the base rate and skill approaches zero from below;
    if it has signal, calibration should push skill positive.
    """
    out = np.empty_like(logit_te)
    for k in range(logit_te.shape[1]):
        ok = ~np.isnan(y_val[:, k])
        yk = y_val[ok, k]
        if ok.sum() < 50 or len(np.unique(yk)) < 2:
            out[:, k] = 1.0 / (1.0 + np.exp(-logit_te[:, k]))
            continue
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit_val[ok, k][:, None], yk)
        out[:, k] = lr.predict_proba(logit_te[:, k][:, None])[:, 1]
    return out


# -- walk-forward --------------------------------------------------------------

def run_walk_forward(feats: pd.DataFrame, close: pd.Series, horizons: list[int], device
                     ) -> tuple[pd.DataFrame, list[dict]]:
    embargo = max(horizons)
    years   = sorted({d.year for d in feats.index if d.year >= FIRST_TEST_YEAR})

    rows, fold_log = [], []
    for year in years:
        te_dates = feats.index[feats.index.year == year]
        tr_dates = feats.index[feats.index < te_dates[0]]
        if len(tr_dates) <= embargo + 500 or len(te_dates) == 0:
            continue
        tr_dates = tr_dates[:-embargo]

        # Normalisation fitted on training rows only.
        mu, sd = feats.loc[tr_dates].mean(), feats.loc[tr_dates].std().replace(0, 1.0)
        arr    = ((feats - mu) / sd).to_numpy(dtype=np.float32)

        X, y, ts = build_direction_windows(arr, close.to_numpy(dtype=float),
                                           WINDOW_ANCHORS, WINDOW_LEN, horizons)
        sample_dates = feats.index[ts]

        is_tr = sample_dates.isin(tr_dates)
        is_te = sample_dates.isin(te_dates)
        if is_tr.sum() < 400 or is_te.sum() == 0:
            continue

        # Inner chronological validation tail, embargoed like the outer split.
        tr_pos  = np.flatnonzero(is_tr)
        cut     = len(tr_pos) - max(120, int(len(tr_pos) * INNER_VAL_FRAC))
        inner_a = tr_pos[:max(cut - embargo, 1)]
        inner_b = tr_pos[cut:]
        te_pos  = np.flatnonzero(is_te)

        # The model is fitted on the inner-train slice only (inner_b picks the early-stop
        # epoch and, below, fits the calibrator); it is not refitted on the full training
        # window afterwards. Deliberate: it keeps one clean held-out block per fold.
        model, val_loss, best_epoch = train_fold(
            X[inner_a], y[inner_a], X[inner_b], y[inner_b],
            arr.shape[1], len(horizons), device,
        )
        logit_te  = predict_logits(model, X[te_pos],  device)
        logit_val = predict_logits(model, X[inner_b], device)
        p_te      = 1.0 / (1.0 + np.exp(-logit_te))
        p_cal     = platt_scale(logit_val, y[inner_b], logit_te)

        # Base rate: the training window's realised up-rate, exactly as in experiment 1.
        base = np.nanmean(y[tr_pos], axis=0)

        for k, h in enumerate(horizons):
            for j, pos in enumerate(te_pos):
                if np.isnan(y[pos, k]):
                    continue
                rows.append({
                    'date': sample_dates[pos], 'year': year, 'horizon': h,
                    'y': int(y[pos, k]),
                    'base': float(base[k]),
                    'mamba_dir': float(p_te[j, k]),
                    'mamba_cal': float(p_cal[j, k]),
                })

        fold_log.append({'year': year, 'n_train': int(is_tr.sum()), 'n_test': int(is_te.sum()),
                         'inner_val_loss': val_loss, 'best_epoch': best_epoch})
        print(f'  fold {year}: train={int(is_tr.sum()):5d}  test={int(is_te.sum()):4d}  '
              f'best_epoch={best_epoch:3d}  inner_val_bce={val_loss:.5f}')

    return pd.DataFrame(rows), fold_log


# -- entry point ---------------------------------------------------------------

def main() -> None:
    ensure_dirs()
    spec     = json.loads(NODE_SPEC.read_text(encoding='utf-8'))
    horizons = spec['horizons']
    display  = spec['display_horizons']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feats, close = load_inputs(horizons)

    print(f'Device     : {device}')
    print(f'Data       : {len(feats)} rows  {feats.index[0].date()} -> {feats.index[-1].date()}')
    print(f'Features   : {len(FEATURES)} channels (train-only standardised per fold)')
    print(f'Windows    : {WINDOW_ANCHORS} x {WINDOW_LEN}   nearest window ends on t')
    print(f'Target     : direction at +{horizons[0]}d .. +{horizons[-1]}d, BCE-with-logits')
    print(f'Protocol   : walk-forward by calendar year from {FIRST_TEST_YEAR}, '
          f'embargo={max(horizons)} rows\n')

    df, folds = run_walk_forward(feats, close, horizons, device)
    if df.empty:
        print('No predictions produced.')
        return

    df.to_csv(PRED_CSV, index=False)
    summary  = score_table(df, MODELS, display, seed=SEED,
                           title='MambaSSM directional -- pooled out-of-sample scores')
    for h in display:
        edge_buckets(df, h, 'mamba_cal')
    per_fold = per_fold_skill(df, MODELS, PIVOT_HORIZON)

    spread = df.groupby('horizon')['mamba_dir'].agg(['min', 'max', 'std'])
    print('\n  Prediction spread by horizon (a collapsed model outputs a near-constant):')
    print(f'    {"horizon":<9} {"min":>8} {"max":>8} {"std":>8}')
    print(f'    {"-" * 9} {"-" * 8} {"-" * 8} {"-" * 8}')
    for h in display:
        r = spread.loc[h]
        print(f'    +{h:<8d} {r["min"]:>8.4f} {r["max"]:>8.4f} {r["std"]:>8.4f}')

    SUMMARY_JSON.write_text(json.dumps({
        'protocol': {
            'first_test_year': FIRST_TEST_YEAR, 'pivot_horizon': PIVOT_HORIZON,
            'embargo': max(horizons), 'window_anchors': WINDOW_ANCHORS,
            'window_len': WINDOW_LEN, 'features': FEATURES,
            'n_predictions': int(len(df)),
            'test_start': str(df['date'].min().date()), 'test_end': str(df['date'].max().date()),
        },
        'model': {'d_model': D_MODEL, 'n_layer': N_LAYER, 'd_state': D_STATE,
                  'dropout': DROPOUT, 'lr': LR, 'batch_size': BATCH_SIZE,
                  'weight_decay': WEIGHT_DECAY, 'max_epochs': EPOCHS, 'patience': PATIENCE},
        'pooled': summary,
        'per_fold_skill': per_fold,
        'folds': folds,
        'prediction_spread': {f'+{h}d': spread.loc[h].to_dict() for h in display},
    }, indent=2, default=float), encoding='utf-8')

    print(f'\nPredictions -> {PRED_CSV}')
    print(f'Summary     -> {SUMMARY_JSON}')


if __name__ == '__main__':
    main()
