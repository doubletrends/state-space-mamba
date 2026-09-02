"""Feature functions ported from the companion winrate-matrix project.

Every function here is a faithful port of ``data/features.py`` (cross-asset) and
``workspaces/btc_daily_14days/plugin.py`` (BTC on-chain + halving cycle) in
https://github.com/chengmarc/winrate-matrix, so that a deviation surface computed
here reproduces the one in that project's node spreadsheets.

They are ported rather than imported because this project needs to *refit* the
surfaces on arbitrary training sub-windows (see ``encoder.py``); the sibling
project only ever fits on full history and persists the result to xlsx.

All functions are strictly backward-looking — ``f(t)`` depends only on rows
``<= t`` — so computing them over the full panel leaks nothing across a later
train/test split. Only the *conditional counting* in ``encoder.py`` needs a fit
mask.
"""

from typing import Callable

import numpy as np
import pandas as pd

_registry: dict[str, Callable] = {}


def register(name: str, fn: Callable) -> None:
    _registry[name] = fn


def compute(data: pd.DataFrame, feature: str, params: dict) -> pd.Series:
    if feature not in _registry:
        raise ValueError(f"Unknown feature: '{feature}'. Available: {sorted(_registry)}")
    return _registry[feature](data, params)


def available() -> list[str]:
    return sorted(_registry)


# -- oscillators ---------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    rs    = up.rolling(period).mean() / down.rolling(period).mean()
    return 100.0 - (100.0 / (1.0 + rs))


def _rsi_spread(close: pd.Series, fast: int, slow: int) -> pd.Series:
    return _rsi(close, fast) - _rsi(close, slow)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return ((close - hh) / (hh - ll) + 1.0)   # [0, 1]; 1 = oversold, 0 = overbought


def _wr_spread(high: pd.Series, low: pd.Series, close: pd.Series, fast: int, slow: int) -> pd.Series:
    return _williams_r(high, low, close, fast) - _williams_r(high, low, close, slow)


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int) -> pd.Series:
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    return 100.0 * (close - ll) / (hh - ll)


def _stoch_d(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int = 3) -> pd.Series:
    return _stoch_k(high, low, close, k_period).rolling(d_period).mean()


# -- trend / momentum ----------------------------------------------------------

def _macd(close: pd.Series, fast: int, slow: int) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow) / close   # normalized by price


def _macd_histogram(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    line = _macd(close, fast, slow)
    return line - line.ewm(span=signal, adjust=False).mean()


def _ma_ratio(close: pd.Series, period: int) -> pd.Series:
    return close / close.rolling(period).mean() - 1.0


def _ma_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    return close.rolling(fast).mean() / close.rolling(slow).mean() - 1.0


def _roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period)


def _roc_spread(close: pd.Series, fast: int, slow: int) -> pd.Series:
    return close.pct_change(fast) - close.pct_change(slow)


# -- volatility ----------------------------------------------------------------

def _realized_vol(close: pd.Series, period: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252) * 100


def _vol_ratio(close: pd.Series, fast: int, slow: int) -> pd.Series:
    log_ret  = np.log(close / close.shift(1))
    fast_vol = log_ret.rolling(fast).std()
    slow_vol = log_ret.rolling(slow).std()
    return fast_vol / slow_vol.replace(0, np.nan)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean() / close   # normalized by price


def _bb_pct(close: pd.Series, period: int, std_dev: float = 2.0) -> pd.Series:
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - (ma - std_dev * std)) / (2 * std_dev * std)


def _bb_width(close: pd.Series, period: int, std_dev: float = 2.0) -> pd.Series:
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std_dev * std) / ma


# -- structure / flow ----------------------------------------------------------

def _volume_ratio(volume: pd.Series, period: int) -> pd.Series:
    return volume / volume.rolling(period).mean()


def _drawdown(close: pd.Series, period: int) -> pd.Series:
    return close / close.rolling(period).max() - 1.0


def _drawdown_recovery(close: pd.Series, short: int, long: int) -> pd.Series:
    return _drawdown(close, short) - _drawdown(close, long)


# -- macro ---------------------------------------------------------------------

def _dxy_ret(dxy: pd.Series, period: int) -> pd.Series:
    return dxy.pct_change(period)


def _dxy_ma_ratio(dxy: pd.Series, period: int) -> pd.Series:
    dxy = dxy.ffill()   # fill weekend gaps so the rolling window sees consecutive values
    return dxy / dxy.rolling(period).mean() - 1.0


# -- on-chain (BTC) ------------------------------------------------------------

def _hash_rate_ma_ratio(data: pd.DataFrame, period: int) -> pd.Series:
    hr = np.log(data['hash_rate'].replace(0, np.nan))
    return hr / hr.rolling(period).mean() - 1.0


def _adr_act_ma_ratio(data: pd.DataFrame, period: int) -> pd.Series:
    aa = np.log(data['adr_act_cnt'].replace(0, np.nan))
    return aa / aa.rolling(period).mean() - 1.0


# -- halving cycle (BTC) -------------------------------------------------------
# Kept local rather than imported from ssm.config: these surfaces must reproduce
# winrate-matrix, whose plugin carries its own halving table.

_HALVING_DATES = pd.to_datetime([
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-20',
])
_NEXT_HALVING_EST = pd.Timestamp('2028-04-20')
_ALL_HALVINGS     = list(_HALVING_DATES) + [_NEXT_HALVING_EST]


def _cycle_phase(data: pd.DataFrame) -> pd.Series:
    """Fractional position in the current halving cycle: 0 = just after, 1 = just before next."""
    phases = []
    for dt in pd.to_datetime(data.index):
        past   = [h for h in _ALL_HALVINGS if h <= dt]
        future = [h for h in _ALL_HALVINGS if h > dt]
        last   = max(past)   if past   else _ALL_HALVINGS[0]
        nxt    = min(future) if future else _NEXT_HALVING_EST
        phases.append((dt - last).days / max((nxt - last).days, 1))
    return pd.Series(phases, index=data.index, dtype=float)


def _days_to_halving(data: pd.DataFrame) -> pd.Series:
    vals = []
    for dt in pd.to_datetime(data.index):
        future = [h for h in _ALL_HALVINGS if h > dt]
        nxt    = min(future) if future else _NEXT_HALVING_EST
        vals.append(max(0, (nxt - dt).days))
    return pd.Series(vals, index=data.index, dtype=float)


def _days_since_halving(data: pd.DataFrame) -> pd.Series:
    days = []
    for dt in pd.to_datetime(data.index):
        eligible = _HALVING_DATES[_HALVING_DATES <= dt]
        last     = eligible.max() if len(eligible) else _HALVING_DATES.min()
        days.append(max(0, (dt - last).days))
    return pd.Series(days, index=data.index, dtype=float)


# -- registrations -------------------------------------------------------------

register('rsi',                lambda d, p: _rsi(d['close'], p['period']))
register('rsi_spread',         lambda d, p: _rsi_spread(d['close'], p['fast'], p['slow']))
register('stoch_k',            lambda d, p: _stoch_k(d['high'], d['low'], d['close'], p['k_period']))
register('stoch_d',            lambda d, p: _stoch_d(d['high'], d['low'], d['close'], p['k_period'], p.get('d_period', 3)))
register('williams_r',         lambda d, p: _williams_r(d['high'], d['low'], d['close'], p['period']))
register('wr_spread',          lambda d, p: _wr_spread(d['high'], d['low'], d['close'], p['fast'], p['slow']))
register('macd',               lambda d, p: _macd(d['close'], p['fast'], p['slow']))
register('macd_histogram',     lambda d, p: _macd_histogram(d['close'], p['fast'], p['slow'], p.get('signal', 9)))
register('ma_ratio',           lambda d, p: _ma_ratio(d['close'], p['period']))
register('ma_cross',           lambda d, p: _ma_cross(d['close'], p['fast'], p['slow']))
register('realized_vol',       lambda d, p: _realized_vol(d['close'], p['period']))
register('vol_ratio',          lambda d, p: _vol_ratio(d['close'], p['fast'], p['slow']))
register('atr',                lambda d, p: _atr(d['high'], d['low'], d['close'], p['period']))
register('bb_pct',             lambda d, p: _bb_pct(d['close'], p['period'], p.get('std_dev', 2.0)))
register('bb_width',           lambda d, p: _bb_width(d['close'], p['period'], p.get('std_dev', 2.0)))
register('volume_ratio',       lambda d, p: _volume_ratio(d['volume'], p['period']))
register('drawdown',           lambda d, p: _drawdown(d['close'], p['period']))
register('drawdown_recovery',  lambda d, p: _drawdown_recovery(d['close'], p['short'], p['long']))
register('roc',                lambda d, p: _roc(d['close'], p['period']))
register('roc_spread',         lambda d, p: _roc_spread(d['close'], p['fast'], p['slow']))
register('dxy_ret',            lambda d, p: _dxy_ret(d['dxy'], p['period']))
register('dxy_ma_ratio',       lambda d, p: _dxy_ma_ratio(d['dxy'], p['period']))
register('mvrv',               lambda d, p: d['mvrv'])
register('hash_rate_ma_ratio', lambda d, p: _hash_rate_ma_ratio(d, p['period']))
register('adr_act_ma_ratio',   lambda d, p: _adr_act_ma_ratio(d, p['period']))
register('cycle_phase',        lambda d, p: _cycle_phase(d))
register('days_to_halving',    lambda d, p: _days_to_halving(d))
register('days_since_halving', lambda d, p: _days_since_halving(d))
