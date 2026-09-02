"""Stage 5 — synthetic future rollout beyond the observed data window.

The model is trained on the full observed DM dataset with a random train/val split.
Stage 5 then rolls forward for ``FORECAST_DAYS`` beyond the final observed date:

    - ``years_since_halving`` is computed deterministically
    - six exogenous inputs are projected by separate cycle-aligned linear regressions
    - ``log_price_residual`` is forecast autoregressively and fed back as input

    Input : config.DM_CSV, config.TRAIN_META_JSON, config.CHECKPOINT,
            config.TREND_PARAMS_JSON
    Output: step5_fig_evaluation.png, step5_rollout.csv
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from ssm.config import (
    OUTPUT, DM_CSV, TRAIN_META_JSON, TREND_PARAMS_JSON, CHECKPOINT,
    FORECAST_DAYS, HALVING_DATES,
)
from ssm.arch.mamba import create_model
from ssm.artifacts import load_checkpoint_state, load_train_meta

INPUT_FEATURE_COLUMNS = [
    'years_since_halving',
    'realized_vol_30',
    'dxy_ret_30',
    'dxy_ret_100',
    'short_percent_r',
    'long_percent_r',
    'wr_composite',
    'log_price_residual',
]
PROJECTED_FEATURE_COLUMNS = [
    'realized_vol_30',
    'dxy_ret_30',
    'dxy_ret_100',
    'short_percent_r',
    'long_percent_r',
    'wr_composite',
]
TARGET_COLUMN = 'log_price_residual_target'

DEEP_BLUE = 'steelblue'
PALE_BLUE = 'lightsteelblue'
PROJECTION_WARMUP_DAYS = 30


def load_artifacts():
    data = pd.read_csv(DM_CSV, index_col='Date', parse_dates=True)
    meta = load_train_meta(TRAIN_META_JSON)
    return data, meta


def load_trend_params() -> dict:
    with open(TREND_PARAMS_JSON) as f:
        return json.load(f)


def require_meta(meta, *keys):
    missing = [k for k in keys if k not in meta]
    if missing:
        raise KeyError(f'train_meta.json missing: {", ".join(missing)}. Re-run training.')
    return [meta[k] for k in keys]


def cycle_info(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return cycle id and day-since-halving for each date."""
    rows = []
    for dt in pd.to_datetime(index):
        eligible = HALVING_DATES[HALVING_DATES <= dt]
        if len(eligible):
            last_halving = eligible.max()
            cycle_id = int(np.where(HALVING_DATES == last_halving)[0][-1])
        else:
            last_halving = HALVING_DATES.min()
            cycle_id = 0
        rows.append({
            'date': dt,
            'cycle_id': cycle_id,
            'cycle_day': int((dt - last_halving).days),
            'years_since_halving': max(0.0, (dt - last_halving).days / 365.25),
        })
    return pd.DataFrame(rows).set_index('date')


def project_feature_by_cycle_day(history: pd.DataFrame, future_info: pd.DataFrame, column: str) -> pd.Series:
    """Project one exogenous feature using separate cycle-aligned regressions per future day."""
    history_info = cycle_info(history.index)
    joined = history_info.join(history[[column]], how='inner').dropna()

    preds = []
    for dt, row in future_info.iterrows():
        candidates = joined[(joined['cycle_day'] == row['cycle_day']) & (joined['cycle_id'] < row['cycle_id'])]
        if candidates.empty:
            nearest = joined.assign(day_gap=(joined['cycle_day'] - row['cycle_day']).abs())
            candidates = nearest.sort_values(['day_gap', 'cycle_id']).groupby('cycle_id', as_index=False).head(1)

        x = candidates['cycle_id'].to_numpy(dtype=float)
        y = candidates[column].to_numpy(dtype=float)
        target_cycle = float(row['cycle_id'])

        if len(y) == 0:
            pred = float(history[column].dropna().iloc[-1])
        elif len(y) == 1 or np.allclose(x, x[0]):
            pred = float(y[-1])
        else:
            slope, intercept = np.polyfit(x, y, deg=1)
            pred = float(slope * target_cycle + intercept)
        preds.append(pred)

    return pd.Series(preds, index=future_info.index, name=column)


def blend_from_last_observation(history_last: float, projected: pd.Series, transition_days: int) -> pd.Series:
    """Blend a projected series from the last observed value into its synthetic path.

    Day 1 of the rollout stays anchored to the last observed feature value; the
    synthetic cycle-based projection is phased in over ``transition_days``.
    """
    if projected.empty or transition_days <= 0:
        return projected

    steps = min(len(projected), transition_days)
    weights = np.linspace(0.0, 1.0, num=steps, endpoint=True)
    eased = 3.0 * weights ** 2 - 2.0 * weights ** 3

    blended = projected.copy()
    raw = projected.iloc[:steps].to_numpy(dtype=float)
    blended.iloc[:steps] = (1.0 - eased) * float(history_last) + eased * raw
    return blended


def project_future_inputs(history: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
    """Build synthetic future exogenous inputs before residual autoregression."""
    future_index = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    future_info = cycle_info(future_index)
    future = pd.DataFrame(index=future_index, columns=INPUT_FEATURE_COLUMNS, dtype=float)
    future['years_since_halving'] = future_info['years_since_halving']

    for column in PROJECTED_FEATURE_COLUMNS:
        projected = project_feature_by_cycle_day(history, future_info, column)
        future[column] = blend_from_last_observation(
            float(history[column].dropna().iloc[-1]), projected, PROJECTION_WARMUP_DAYS,
        )

    future['log_price_residual'] = np.nan
    return future


def autoregressive_rollout(model, history: pd.DataFrame, meta: dict, device, forecast_days: int) -> pd.DataFrame:
    """Roll the model forward day by day beyond the observed history."""
    window_anchors, window_len = require_meta(meta, 'window_anchors', 'window_len')

    history_inputs = history[INPUT_FEATURE_COLUMNS].copy()
    future_inputs = project_future_inputs(history_inputs, forecast_days)
    working_inputs = pd.concat([history_inputs, future_inputs])

    predictions = []
    model.eval()
    with torch.no_grad():
        for forecast_date in future_inputs.index:
            origin_idx = working_inputs.index.get_loc(forecast_date) - 1
            windows = [[working_inputs.iloc[origin_idx - a - d].values for d in range(window_len - 1, -1, -1)]
                       for a in window_anchors]
            x = torch.tensor(np.array([windows], dtype=np.float32)).to(device)
            pred = float(model(x).reshape(-1)[0].cpu())
            working_inputs.loc[forecast_date, 'log_price_residual'] = pred

            record = {'date': forecast_date, 'predicted_residual': pred}
            for column in INPUT_FEATURE_COLUMNS:
                record[column] = float(working_inputs.loc[forecast_date, column])
            predictions.append(record)

    return pd.DataFrame(predictions).set_index('date')


def reconstruct_prices(index: pd.DatetimeIndex, residuals: np.ndarray, trend_params: dict) -> np.ndarray:
    a, b, c = trend_params['a'], trend_params['b'], trend_params['c']
    ref_date = pd.Timestamp(trend_params['ref_date'])
    x_days = (index - ref_date).days.to_numpy(dtype=float)
    trend = a * np.log(x_days + c) + b
    return np.exp(residuals + trend)


def plot_evaluation(history: pd.DataFrame, forecast: pd.DataFrame, trend_params: dict) -> None:
    context_days = min(180, len(history))
    history_tail = history.iloc[-context_days:].copy()
    history_tail['actual_price'] = reconstruct_prices(
        history_tail.index, history_tail['log_price_residual'].to_numpy(dtype=float), trend_params
    )

    fig, (ax_res, ax_px) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Synthetic future rollout — projected exogenous inputs + autoregressive residual', fontsize=13)

    ax_res.plot(history_tail.index, history_tail['log_price_residual'], color=DEEP_BLUE, linewidth=1.0, label='history')
    ax_res.plot(forecast.index, forecast['predicted_residual'], color='tomato', linewidth=1.1, linestyle='--',
                label='forecast')
    ax_res.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax_res.set_ylabel('log-price residual')
    ax_res.legend(fontsize=9)
    ax_res.grid(linestyle=':', linewidth=0.5)

    ax_px.plot(history_tail.index, history_tail['actual_price'], color=PALE_BLUE, linewidth=1.4, label='history price')
    ax_px.plot(forecast.index, forecast['predicted_price'], color='tomato', linewidth=1.1, linestyle='--',
               label='forecast price')
    ax_px.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_px.set_ylabel('BTC price (USD)')
    ax_px.set_xlabel('date')
    ax_px.legend(fontsize=9)
    ax_px.grid(linestyle=':', linewidth=0.5)

    plt.tight_layout()
    path = OUTPUT / 'step5_fig_evaluation.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved → {path.name}')


def hindcast_rollout(model, data: pd.DataFrame, meta: dict, device, n_days: int = 365) -> pd.DataFrame:
    """Slide the model over the last n_days of observed data using real inputs.

    At each anchor t the windows are built from actual observed features
    (teacher-forced, no autoregression). Only the t+1 step prediction is
    recorded and compared to the actual next-day residual.
    """
    window_anchors, window_len = require_meta(meta, 'window_anchors', 'window_len')
    inputs = data[INPUT_FEATURE_COLUMNS]
    max_offset = max(window_anchors) + window_len - 1
    end_idx   = len(data) - 1          # last anchor with a known t+1 actual
    start_idx = max(max_offset, end_idx - n_days)

    records = []
    model.eval()
    with torch.no_grad():
        for t in range(start_idx, end_idx):
            windows = [[inputs.iloc[t - a - d].values for d in range(window_len - 1, -1, -1)]
                       for a in window_anchors]
            x    = torch.tensor(np.array([windows], dtype=np.float32)).to(device)
            pred = float(model(x).reshape(-1)[0].cpu())
            records.append({
                'date':                  data.index[t],
                'predicted_residual_t1': pred,
                'actual_residual_t1':    float(data['log_price_residual'].iloc[t + 1]),
            })

    return pd.DataFrame(records).set_index('date')


def plot_hindcast(hindcast: pd.DataFrame, trend_params: dict) -> None:
    pred_prices   = reconstruct_prices(
        hindcast.index + pd.Timedelta(days=1),
        hindcast['predicted_residual_t1'].to_numpy(dtype=float), trend_params,
    )
    actual_prices = reconstruct_prices(
        hindcast.index + pd.Timedelta(days=1),
        hindcast['actual_residual_t1'].to_numpy(dtype=float), trend_params,
    )

    fig, (ax_res, ax_px) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Hindcast — model predictions vs actual (last year, teacher-forced)', fontsize=13)

    ax_res.plot(hindcast.index, hindcast['actual_residual_t1'],    color=DEEP_BLUE, linewidth=1.0, label='actual')
    ax_res.plot(hindcast.index, hindcast['predicted_residual_t1'], color='tomato',  linewidth=1.0, linestyle='--', label='predicted (t+1)')
    ax_res.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax_res.set_ylabel('log-price residual')
    ax_res.legend(fontsize=9)
    ax_res.grid(linestyle=':', linewidth=0.5)

    ax_px.plot(hindcast.index, actual_prices, color=PALE_BLUE, linewidth=1.4, label='actual price')
    ax_px.plot(hindcast.index, pred_prices,   color='tomato',  linewidth=1.0, linestyle='--', label='predicted price (t+1)')
    ax_px.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_px.set_ylabel('BTC price (USD)')
    ax_px.set_xlabel('date')
    ax_px.legend(fontsize=9)
    ax_px.grid(linestyle=':', linewidth=0.5)

    plt.tight_layout()
    path = OUTPUT / 'step5_fig_hindcast.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved → {path.name}')


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    data, meta = load_artifacts()
    window_anchors, window_len, predict_window, d_model, n_layer, d_state, dropout = require_meta(
        meta, 'window_anchors', 'window_len', 'predict_window', 'd_model', 'n_layer', 'd_state', 'dropout'
    )

    model = create_model(data, predict_window, device, n_windows=len(window_anchors),
                         d_model=d_model, n_layer=n_layer, d_state=d_state, dropout=dropout)
    model.load_state_dict(load_checkpoint_state(CHECKPOINT, device))
    model.eval()

    print(f'Data end       : {data.index[-1].date()}')
    print(f'Forecast days  : {FORECAST_DAYS}')
    print(f'Windows        : {window_anchors}×{window_len}')
    print('Running synthetic future rollout (projected exogenous inputs + autoregressive residual)...')

    forecast = autoregressive_rollout(model, data, meta, device, FORECAST_DAYS)
    if forecast.empty:
        print('No predictions.')
        return

    trend_params = load_trend_params()
    forecast['predicted_price'] = reconstruct_prices(
        forecast.index, forecast['predicted_residual'].to_numpy(dtype=float), trend_params
    )
    plot_evaluation(data, forecast, trend_params)

    csv_path = OUTPUT / 'step5_rollout.csv'
    forecast.to_csv(csv_path, float_format='%.6f')
    print(f'Saved → {csv_path.name}')

    print('\nRunning hindcast (last 365 days, teacher-forced)...')
    hindcast = hindcast_rollout(model, data, meta, device, n_days=365)
    plot_hindcast(hindcast, trend_params)


if __name__ == '__main__':
    evaluate()
