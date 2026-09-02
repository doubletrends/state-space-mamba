"""Conditional-winrate encoding, ported from the companion winrate-matrix project.

The sibling project established that BTC daily returns carry real conditional
structure -- P(up at t+h | X_t in condition) deviates from its unconditional base
rate by 10-35pp for valuation, volatility and overextension features. This package
makes that structure usable *as model input* rather than only as a report:

    features.py  feature functions, ported so surfaces reproduce the sibling project
    encoder.py   fit-mask-aware surfaces + the delta / z encodings + Naive Bayes
    data.py      the OHLCV + DXY + on-chain panel the nodes are computed over
"""

from .encoder import Node, WinrateEncoder, labels, logodds, shrink_weight

__all__ = ['Node', 'WinrateEncoder', 'labels', 'logodds', 'shrink_weight']
