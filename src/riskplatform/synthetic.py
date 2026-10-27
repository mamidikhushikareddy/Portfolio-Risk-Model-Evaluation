"""Deterministic SYNTHETIC price generator for tests and demos.

These are simulated random walks, NOT market data. They carry no information about
real ETFs. Any output derived from them must be labelled synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_PARAMS = {  # annual drift, annual vol (arbitrary illustrative values)
    "VTI": (0.09, 0.18), "EFA": (0.06, 0.20), "AGG": (0.03, 0.04), "TLT": (0.03, 0.14),
    "GLD": (0.05, 0.16), "EEM": (0.06, 0.24), "XLE": (0.06, 0.26), "SPY": (0.09, 0.18),
}
_CORR_GROUPS = {"VTI": 0, "SPY": 0, "EFA": 0, "EEM": 0, "XLE": 0, "AGG": 1, "TLT": 1, "GLD": 2}


def make_synthetic_prices(start: str = "2005-01-03", end: str = "2024-12-31", seed: int = 42,
                          tickers: list[str] | None = None, crisis_windows: bool = True) -> pd.DataFrame:
    """Business-day prices with correlated shocks, fixed holidays removed, and stress regimes."""
    tickers = tickers or list(_PARAMS)
    idx = pd.bdate_range(start, end)
    idx = idx[~((idx.month == 12) & (idx.day == 25)) & ~((idx.month == 1) & (idx.day == 1))
              & ~((idx.month == 7) & (idx.day == 4))]
    rng = np.random.default_rng(seed)
    n, k = len(idx), len(tickers)
    corr = np.full((k, k), 0.1)
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if _CORR_GROUPS[a] == _CORR_GROUPS[b]:
                corr[i, j] = 0.85 if {a, b} & {"SPY", "VTI"} else 0.7
    corr[np.arange(k), np.arange(k)] = 1.0
    z = rng.standard_normal((n, k)) @ np.linalg.cholesky(corr).T
    mu = np.array([_PARAMS[t][0] for t in tickers]) / 252
    sig = np.array([_PARAMS[t][1] for t in tickers]) / np.sqrt(252)
    r = mu + sig * z
    if crisis_windows:
        eq = np.array([_CORR_GROUPS[t] == 0 for t in tickers])
        for s, e in [("2008-09-15", "2009-03-09"), ("2020-02-20", "2020-03-23")]:
            m = (idx >= s) & (idx <= e)
            r[np.ix_(m, eq)] = r[np.ix_(m, eq)] * 2.5 - 0.004
    prices = 100 * np.cumprod(1 + r, axis=0)
    return pd.DataFrame(prices, index=idx, columns=tickers).rename_axis("date")


def make_synthetic_yields(index: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    """Synthetic DGS10/DGS2-like level series (percent) for demos only."""
    rng = np.random.default_rng(seed)
    d10 = 3.0 + np.cumsum(rng.normal(0, 0.03, len(index)))
    d2 = 2.0 + np.cumsum(rng.normal(0, 0.035, len(index)))
    return pd.DataFrame({"DGS10": np.clip(d10, 0.2, None), "DGS2": np.clip(d2, 0.1, None)}, index=index)
