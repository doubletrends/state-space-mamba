"""Leak-free reimplementation of the winrate-matrix conditional-deviation surfaces.

The sibling project (https://github.com/chengmarc/winrate-matrix) fits every node
on full history and persists the result to xlsx. That is fine for describing the
past, but a surface fitted on all 4,000 rows cannot be used as a *feature* in a
model evaluated out of sample -- the outcome of the test rows is baked into the
bin it is then scored against.

This module rebuilds the same three quantities with an explicit ``fit_index``:

    base rate      p0(h)                       = P(close[t+h] > close[t])
    CDF surface    P(up | X > x) - p0(h)       over a threshold grid
    PDF surface    P(up | X ~= x) - p0(h)      recovered by finite-differencing the CDF

Fitted on a training sub-window, they can be applied to later rows honestly.

Naming, the threshold grid (30 points spanning the 2nd-98th percentile), the
n < 20 undefined floor, the longest-horizon observation count, the finite
difference and the shrinkage weights all follow the sibling project exactly, so
a surface fitted here on full history reproduces its spreadsheets.

Two encodings are produced per node and horizon:

    delta   d_i(t, h)  -- the raw deviation in percentage points, interpolated at x_t
    z       z_i(t, h)  -- the Naive Bayes log-odds contribution of that deviation,

                z_i = w(n_i) * [ logit(p0_h + d_i) - logit(p0_h) ]

``z`` is the useful one for modelling: summing it and adding ``logit p0`` *is*
the sibling project's ``combine()``. So a logistic regression on ``z`` with an
intercept is exactly Naive Bayes with learned weights instead of weights pinned
at 1.0 -- which makes the two directly nested, and the comparison meaningful.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import features

MIN_N         = 20    # a slice with fewer observations is undefined (matches winrate-matrix)
SHRINK_FLOOR  = 30    # bins below the CLT floor contribute zero weight
SHRINK_N0     = 50    # excess observations needed to reach half weight above the floor
N_THRESHOLDS  = 30
PCT_LO, PCT_HI = 2.0, 98.0


# -- node specification --------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """One feature evaluated at one parameter set -- the unit of work."""
    id:      str
    family:  str
    feature: str
    params:  dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> 'Node':
        return cls(id=d['id'], family=d['family'], feature=d['feature'], params=d.get('params') or {})


# -- primitives ----------------------------------------------------------------

def shrink_weight(n) -> float:
    """Two-part shrinkage weight: 0 below the CLT floor, 0.5 at floor+N0, -> 1 as n -> inf."""
    n = int(n) if pd.notna(n) else 0
    if n < SHRINK_FLOOR:
        return 0.0
    excess = n - SHRINK_FLOOR
    return excess / (excess + SHRINK_N0)


def logodds(p_pct) -> np.ndarray:
    """Log-odds of a probability given in percent, clamped away from 0 and 100."""
    p = np.clip(np.asarray(p_pct, dtype=float) / 100.0, 0.001, 0.999)
    return np.log(p / (1.0 - p))


def price_up(close: pd.Series, h: int) -> pd.Series:
    """True/False/NaN -- did close rise h bars later? NaN for the last h rows."""
    shifted = close.shift(-h)
    return (shifted > close).where(shifted.notna())


def usable_masks(fit_mask: np.ndarray, horizons: list[int]) -> dict[int, np.ndarray]:
    """Per-horizon mask of rows that may be counted during a fit.

    A training row is only usable at horizon ``h`` if its *outcome* row ``t+h`` also
    falls inside the fit window. Without this, the last ``h`` rows of a training
    window resolve against prices from the test period, and the surface quietly
    carries the future it is supposed to be predicting.

    Enforcing it here rather than asking callers to trim their index keeps the
    guarantee with the code that can violate it, and is exact per horizon instead of
    trimming every horizon by the longest one.
    """
    n = len(fit_mask)
    out = {}
    for h in horizons:
        shifted        = np.zeros(n, dtype=bool)
        shifted[:n - h] = fit_mask[h:]
        out[h] = fit_mask & shifted
    return out


def compute_base_rate(close: pd.Series, horizons: list[int], fit_mask: np.ndarray) -> pd.Series:
    """Unconditional win rate per horizon, in percent, counted over ``fit_mask`` only."""
    usable = usable_masks(fit_mask, horizons)
    out = {}
    for h in horizons:
        fu    = price_up(close, h).to_numpy(dtype=float)
        valid = usable[h] & ~np.isnan(fu)
        n     = int(valid.sum())
        out[h] = float(np.nanmean(fu[valid]) * 100) if n >= MIN_N else np.nan
    return pd.Series(out, name='base_rate')


# -- surfaces ------------------------------------------------------------------

def _cdf_above(feat: np.ndarray, ups: dict[int, np.ndarray], thresholds: np.ndarray,
               horizons: list[int], fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(up | X > x) in percent for each (threshold, horizon), plus the longest-horizon count.

    Returns ``(rates, counts)`` where ``rates`` is ``(n_thresholds, n_horizons)`` and
    ``counts`` is ``(n_thresholds,)`` -- the count at the *longest* horizon, the most
    conservative choice because longer horizons lose more tail rows to unrealized
    outcomes, so n decreases monotonically with h.
    """
    rates  = np.full((len(thresholds), len(horizons)), np.nan)
    counts = np.zeros(len(thresholds), dtype=int)
    h_last = horizons[-1]
    usable = usable_masks(fit_mask, horizons)

    for i, x in enumerate(thresholds):
        sel = feat > x
        for j, h in enumerate(horizons):
            fu    = ups[h]
            valid = sel & usable[h] & ~np.isnan(fu)
            n     = int(valid.sum())
            if h == h_last:
                counts[i] = n
            if n >= MIN_N:
                rates[i, j] = float(fu[valid].mean() * 100)
    return rates, counts


def _pdf_from_cdf(rates: np.ndarray, counts: np.ndarray, thresholds: np.ndarray,
                  base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the local (PDF) deviation by finite-differencing the CDF-above.

    Rows are ordered by ascending threshold, so row i covers the larger population
    (X > lower threshold) and row i+1 the smaller one. Wins inside the slice between
    two adjacent thresholds are ``n_hi * P_hi - n_lo * P_lo``; dividing by the slice
    count gives the local win rate for observations whose feature value falls *within*
    the interval. Deviations are returned in percentage points from the base rate.

    Returns ``(midpoints, deviations, slice_counts)``.
    """
    mids, devs, ns = [], [], []
    for i in range(len(thresholds) - 1):
        n_hi, n_lo = int(counts[i]), int(counts[i + 1])
        n_slice    = n_hi - n_lo
        if n_slice < 1:
            continue
        v_hi, v_lo = rates[i], rates[i + 1]
        wins       = n_hi * v_hi / 100.0 - n_lo * v_lo / 100.0
        row        = np.where(
            np.isnan(v_hi) | np.isnan(v_lo),
            np.nan,
            wins / n_slice * 100.0 - base,
        )
        mids.append((thresholds[i] + thresholds[i + 1]) / 2.0)
        devs.append(row)
        ns.append(n_slice)

    if not mids:
        n_h = rates.shape[1]
        return np.empty(0), np.empty((0, n_h)), np.empty(0, dtype=int)
    return np.asarray(mids), np.vstack(devs), np.asarray(ns, dtype=int)


def _interp_clamped(xs: np.ndarray, table: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``table`` rows at ``query``, clamping outside the grid.

    NaN cells are carried through rather than filled: a bin the training window
    never populated has no estimate, and inventing one would manufacture edge.
    """
    out = np.full((len(query), table.shape[1]), np.nan)
    if len(xs) == 0:
        return out
    idx = np.clip(np.searchsorted(xs, query) - 1, 0, len(xs) - 2) if len(xs) > 1 else np.zeros(len(query), int)
    for k, q in enumerate(query):
        if np.isnan(q):
            continue
        if len(xs) == 1 or q <= xs[0]:
            out[k] = table[0]
            continue
        if q >= xs[-1]:
            out[k] = table[-1]
            continue
        i  = idx[k]
        lo, hi = xs[i], xs[i + 1]
        t  = (q - lo) / (hi - lo) if hi != lo else 0.5
        out[k] = table[i] * (1.0 - t) + table[i + 1] * t
    return out


def _interp_counts(xs: np.ndarray, ns: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Nearest-bin observation count for each query value (clamped at the grid edges)."""
    out = np.zeros(len(query))
    if len(xs) == 0:
        return out
    for k, q in enumerate(query):
        if np.isnan(q):
            continue
        out[k] = ns[int(np.argmin(np.abs(xs - q)))]
    return out


# -- the encoder ---------------------------------------------------------------

class WinrateEncoder:
    """Fits one deviation surface per node, then encodes any row as delta / z vectors.

    ``fit`` counts only over ``fit_index``, and only over rows whose *outcome* also
    falls inside it (see :func:`usable_masks`); ``transform`` may then be called on
    any rows, including later ones. Features themselves are computed over the whole
    panel -- they are backward-looking, so that leaks nothing.
    """

    def __init__(self, nodes: list[Node], horizons: list[int],
                 n_thresholds: int = N_THRESHOLDS):
        self.nodes        = list(nodes)
        self.horizons     = list(horizons)
        self.n_thresholds = n_thresholds
        self.base_rate_: pd.Series | None = None
        self.surfaces_: dict[str, dict] = {}
        self.dropped_: dict[str, str]   = {}

    # -- fit --

    def fit(self, data: pd.DataFrame, fit_index: pd.DatetimeIndex) -> 'WinrateEncoder':
        close    = data['close']
        fit_mask = data.index.isin(fit_index)
        self.base_rate_ = compute_base_rate(close, self.horizons, fit_mask)
        base_arr = self.base_rate_.to_numpy(dtype=float)

        ups = {h: price_up(close, h).to_numpy(dtype=float) for h in self.horizons}

        self.surfaces_.clear()
        self.dropped_.clear()
        for node in self.nodes:
            try:
                feat = features.compute(data, node.feature, node.params).reindex(data.index)
            except Exception as exc:                     # a node whose source column is absent
                self.dropped_[node.id] = f'{type(exc).__name__}: {exc}'
                continue

            fv       = feat.to_numpy(dtype=float)
            fit_vals = fv[fit_mask & ~np.isnan(fv)]
            if len(fit_vals) < 100 or np.nanstd(fit_vals) == 0:
                self.dropped_[node.id] = f'only {len(fit_vals)} usable training values'
                continue

            lo, hi = np.percentile(fit_vals, [PCT_LO, PCT_HI])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                self.dropped_[node.id] = 'degenerate threshold range'
                continue

            thresholds    = np.linspace(float(lo), float(hi), self.n_thresholds)
            rates, counts = _cdf_above(fv, ups, thresholds, self.horizons, fit_mask)
            mids, devs, ns = _pdf_from_cdf(rates, counts, thresholds, base_arr)
            if len(mids) == 0:
                self.dropped_[node.id] = 'no populated slices'
                continue

            self.surfaces_[node.id] = {
                'node': node, 'feature': feat, 'mids': mids, 'devs': devs, 'ns': ns,
            }
        return self

    # -- inspect --

    @property
    def fitted_ids(self) -> list[str]:
        return [n.id for n in self.nodes if n.id in self.surfaces_]

    def peak_signal(self, node_id: str, horizon: int, min_n: int = SHRINK_FLOOR) -> float:
        """Max |deviation| at ``horizon`` over slices with at least ``min_n`` observations."""
        s = self.surfaces_.get(node_id)
        if s is None:
            return 0.0
        j    = self.horizons.index(horizon)
        vals = s['devs'][s['ns'] >= min_n, j]
        vals = vals[~np.isnan(vals)]
        return float(np.abs(vals).max()) if len(vals) else 0.0

    def best_per_family(self, horizon: int, min_n: int = SHRINK_FLOOR) -> dict[str, str]:
        """Highest-peak node per family, selected only from what this fit could see."""
        best: dict[str, tuple[str, float]] = {}
        for nid in self.fitted_ids:
            fam = self.surfaces_[nid]['node'].family
            pk  = self.peak_signal(nid, horizon, min_n)
            if fam not in best or pk > best[fam][1]:
                best[fam] = (nid, pk)
        return {fam: nid for fam, (nid, _) in sorted(best.items())}

    # -- transform --

    def transform(self, data: pd.DataFrame, node_ids: list[str] | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Encode every row of ``data`` as ``(delta, z)`` frames.

        Both are indexed by date with MultiIndex columns ``(node_id, horizon)``.
        ``delta`` is in percentage points; ``z`` is the shrinkage-weighted log-odds
        contribution, so ``logit p0 + z.sum(axis=1)`` reproduces Naive Bayes.
        """
        ids      = node_ids if node_ids is not None else self.fitted_ids
        base_arr = self.base_rate_.to_numpy(dtype=float)
        base_lo  = logodds(base_arr)

        d_cols, z_cols, keys = [], [], []
        for nid in ids:
            s     = self.surfaces_[nid]
            query = s['feature'].reindex(data.index).to_numpy(dtype=float)
            dev   = _interp_clamped(s['mids'], s['devs'], query)          # (rows, horizons), pp
            cnt   = _interp_counts(s['mids'], s['ns'], query)             # (rows,)
            w     = np.array([shrink_weight(c) for c in cnt])[:, None]

            contrib = w * (logodds(base_arr[None, :] + dev) - base_lo[None, :])
            contrib = np.where(np.isnan(dev), 0.0, contrib)   # an undefined bin adds no evidence

            d_cols.append(dev)
            z_cols.append(contrib)
            keys.extend((nid, h) for h in self.horizons)

        cols = pd.MultiIndex.from_tuples(keys, names=['node', 'horizon'])
        delta = pd.DataFrame(np.hstack(d_cols), index=data.index, columns=cols)
        z     = pd.DataFrame(np.hstack(z_cols), index=data.index, columns=cols)
        return delta, z

    # -- the incumbent --

    def naive_bayes(self, z: pd.DataFrame, horizon: int) -> pd.Series:
        """The sibling project's ``combine()``: base log-odds plus every contribution, unweighted."""
        j  = self.horizons.index(horizon)
        lo = logodds(self.base_rate_.to_numpy(dtype=float))[j]
        s  = z.xs(horizon, axis=1, level='horizon').sum(axis=1)
        return pd.Series(1.0 / (1.0 + np.exp(-(lo + s.to_numpy()))), index=z.index)


def labels(close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Binary up/down outcome per horizon; NaN where the outcome is not yet realized."""
    return pd.DataFrame({h: price_up(close, h) for h in horizons}, index=close.index)
