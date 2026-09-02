"""Evaluation must rely on the saved training manifest, not config fallbacks."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "step_5", Path(__file__).resolve().parents[1] / "pipeline" / "step_5_evaluate.py"
)
step_5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step_5)


def test_require_meta_returns_requested_values():
    meta = {'window_anchors': [365, 45, 30], 'window_len': 7, 'predict_window': 7, 'd_model': 32}
    assert step_5.require_meta(meta, 'window_anchors', 'window_len', 'predict_window', 'd_model') == [
        [365, 45, 30], 7, 7, 32
    ]


def test_require_meta_raises_on_missing_fields():
    with pytest.raises(KeyError, match='train_meta.json missing'):
        step_5.require_meta({'window_anchors': [365, 45, 30]}, 'window_anchors', 'window_len', 'predict_window')


def test_cycle_info_computes_non_negative_cycle_day():
    idx = step_5.pd.to_datetime(['2024-04-20', '2024-04-21'])
    info = step_5.cycle_info(idx)

    assert list(info['cycle_day']) == [0, 1]
    assert info.iloc[0]['years_since_halving'] == pytest.approx(0.0)


def test_project_feature_by_cycle_day_falls_back_to_last_value_when_no_history():
    idx = step_5.pd.to_datetime(['2011-01-01', '2011-01-02'])
    history = step_5.pd.DataFrame({'realized_vol_30': [1.0, 2.0]}, index=idx)
    future_info = step_5.cycle_info(step_5.pd.to_datetime(['2011-01-03']))

    projected = step_5.project_feature_by_cycle_day(history, future_info, 'realized_vol_30')

    assert projected.iloc[0] == pytest.approx(2.0)


def test_blend_from_last_observation_anchors_rollout_start():
    projected = step_5.pd.Series([10.0, 20.0, 30.0], index=step_5.pd.date_range('2026-01-01', periods=3))

    blended = step_5.blend_from_last_observation(5.0, projected, transition_days=3)

    assert blended.iloc[0] == pytest.approx(5.0)
    assert blended.iloc[-1] == pytest.approx(30.0)
    assert 5.0 < blended.iloc[1] < 20.0
