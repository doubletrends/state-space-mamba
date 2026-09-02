"""Directional windowing and the winrate deviation encoder.

The two guards that matter here are alignment and leakage:

  * ``build_direction_windows`` must place the anchor-0 window so it *ends* on t,
    and must label from ``close[t+h]`` -- an off-by-one either way silently turns
    the task into something else that still trains
  * ``WinrateEncoder`` must count only over its fit index, or the surfaces it
    produces carry the test period's outcomes into the model's inputs
"""

import numpy as np
import pandas as pd
import torch

from ssm.arch.mamba import create_direction_model
from ssm.data.loader import build_direction_windows
from ssm.winrate import Node, WinrateEncoder

ANCHORS  = [90, 30, 14, 0]
WIN_LEN  = 7
HORIZONS = [1, 3, 7]
N_FEAT   = 5


def _synthetic(n=400, seed=0):
    rng   = np.random.default_rng(seed)
    feats = rng.standard_normal((n, N_FEAT)).astype(np.float32)
    close = np.cumsum(rng.standard_normal(n)) + 100.0
    return feats, close


def test_window_alignment_includes_anchor_day():
    feats, close = _synthetic()
    X, _, ts = build_direction_windows(feats, close, ANCHORS, WIN_LEN, HORIZONS)
    t = ts[0]
    # anchor 0 spans t-6 .. t inclusive, oldest first -- the embargo is gone
    assert np.allclose(X[0, -1, -1], feats[t])
    assert np.allclose(X[0, -1, 0], feats[t - (WIN_LEN - 1)])
    # the oldest anchor still ends exactly 90 rows back
    assert np.allclose(X[0, 0, -1], feats[t - 90])


def test_shapes_and_feature_count():
    feats, close = _synthetic()
    X, y, ts = build_direction_windows(feats, close, ANCHORS, WIN_LEN, HORIZONS)
    max_offset = max(ANCHORS) + WIN_LEN - 1
    assert X.shape == (len(close) - max_offset, len(ANCHORS), WIN_LEN, N_FEAT)
    assert y.shape == (len(ts), len(HORIZONS))
    # every column is an input for the directional task -- no target column is held out
    assert X.shape[-1] == N_FEAT


def test_labels_are_forward_direction():
    feats, close = _synthetic()
    _, y, ts = build_direction_windows(feats, close, ANCHORS, WIN_LEN, HORIZONS)
    for i in range(0, len(ts), 37):
        t = ts[i]
        for k, h in enumerate(HORIZONS):
            if t + h < len(close):
                assert y[i, k] == float(close[t + h] > close[t])
            else:
                assert np.isnan(y[i, k])


def test_unrealised_horizons_are_nan():
    feats, close = _synthetic()
    _, y, ts = build_direction_windows(feats, close, ANCHORS, WIN_LEN, HORIZONS)
    assert np.isnan(y[-1, -1])                      # +7d off the end of the data
    assert not np.isnan(y[: -max(HORIZONS)]).any()  # everything with an outcome is labelled


def test_direction_model_emits_one_logit_per_horizon():
    feats, close = _synthetic()
    X, _, _ = build_direction_windows(feats, close, ANCHORS, WIN_LEN, HORIZONS)
    model = create_direction_model(N_FEAT, len(HORIZONS), torch.device('cpu'),
                                   len(ANCHORS), d_model=8, n_layer=1, d_state=4)
    out = model(torch.tensor(X[:3]))
    assert out.shape == (3, len(HORIZONS))
    assert torch.isfinite(out).all()


def _price_frame(n=900, seed=1):
    rng   = np.random.default_rng(seed)
    close = pd.Series(np.cumprod(1 + rng.normal(0, 0.02, n)) * 100.0,
                      index=pd.date_range('2015-01-01', periods=n, freq='D'))
    return pd.DataFrame({'close': close, 'high': close * 1.01,
                         'low': close * 0.99, 'volume': 1.0}, index=close.index)


def test_encoder_counts_only_over_the_fit_index():
    """A surface fitted on the first half must not move when the second half changes."""
    data  = _price_frame()
    nodes = [Node(id='rsi_14', family='rsi', feature='rsi', params={'period': 14})]
    fit   = data.index[:450]

    a = WinrateEncoder(nodes, [1, 3, 7]).fit(data, fit)

    tampered = data.copy()
    tampered.iloc[450:, tampered.columns.get_loc('close')] *= 3.0
    b = WinrateEncoder(nodes, [1, 3, 7]).fit(tampered, fit)

    assert np.allclose(a.base_rate_.to_numpy(), b.base_rate_.to_numpy(), equal_nan=True)
    assert np.allclose(a.surfaces_['rsi_14']['devs'],
                       b.surfaces_['rsi_14']['devs'], equal_nan=True)


def test_naive_bayes_reduces_to_base_rate_without_evidence():
    """With every contribution zeroed the combination must return the base rate."""
    data  = _price_frame()
    nodes = [Node(id='rsi_14', family='rsi', feature='rsi', params={'period': 14})]
    enc   = WinrateEncoder(nodes, [1, 3, 7]).fit(data, data.index)
    _, z  = enc.transform(data)

    p = enc.naive_bayes(z * 0.0, horizon=7)
    assert np.allclose(p.to_numpy(), float(enc.base_rate_.loc[7]) / 100.0, atol=1e-9)
