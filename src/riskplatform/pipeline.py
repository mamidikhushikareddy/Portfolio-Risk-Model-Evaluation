"""Orchestration: turn prices + weights into one AnalysisResult used by CLI, app and Excel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from . import analytics as an
from . import backtesting as bt
from . import risk_measures as rm
from . import stress as st
from .config import (BENCHMARK, CONFIDENCE_LEVELS, DEFAULT_BACKTEST_WINDOW, HYPOTHETICAL_SHOCKS, STRESS_EVENTS,
                     TRADING_DAYS)


@dataclass
class AnalysisResult:
    weights: pd.Series
    prices: pd.DataFrame
    returns: pd.DataFrame
    portfolio_returns: pd.Series
    sample_info: dict
    asset_stats: pd.DataFrame
    portfolio_stats: dict
    corr: pd.DataFrame
    cov_annual: pd.DataFrame
    risk_contrib: pd.DataFrame
    var_table: pd.DataFrame
    backtest_summary: pd.DataFrame
    forecasts: dict = field(default_factory=dict)
    stress_replays: list = field(default_factory=list)
    stress_hypothetical: list = field(default_factory=list)
    yield_context: dict = field(default_factory=dict)
    drawdown: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)


def run_analysis(prices: pd.DataFrame, weights: Mapping[str, float] | pd.Series, benchmark: str = BENCHMARK,
                 rf_annual: float = 0.0, window: int = DEFAULT_BACKTEST_WINDOW,
                 confidences: tuple[float, ...] = CONFIDENCE_LEVELS, yields: pd.DataFrame | None = None,
                 events: dict | None = None, shocks: dict | None = None, run_backtests: bool = True,
                 quality: dict | None = None, metadata: dict | None = None) -> AnalysisResult:
    """Compute every metric for a validated set of fractional weights."""
    w = an.validate_weights(weights)
    assets = list(w.index)
    absent = [c for c in assets + [benchmark] if c not in prices.columns]
    if absent:
        raise KeyError(f"Prices unavailable for: {absent}")
    held = list(w[w > 0].index)
    if not held:
        raise ValueError("At least one asset needs a positive weight")
    all_cols = assets + ([benchmark] if benchmark not in assets else [])
    rets_all = an.simple_returns(prices[all_cols])
    sample, info = an.align_returns(rets_all, held + ([benchmark] if benchmark not in held else []))
    wh = w[held]
    port = an.portfolio_returns(sample, wh)
    bench = sample[benchmark]
    cov = an.covariance_matrix(sample[held])
    corr = an.correlation_matrix(sample[held])
    rc = an.risk_contributions(cov, wh).reindex(assets).fillna({"weight": 0.0})
    rc["weight"] = w
    rc[["marginal_contribution", "component_contribution", "pct_contribution"]] = \
        rc[["marginal_contribution", "component_contribution", "pct_contribution"]].fillna(0.0)

    stats_rows = {}
    for a in all_cols:
        s = an.series_stats(rets_all[a], rets_all[benchmark], rf_annual, prices=prices[a])
        stats_rows[a] = s
    pstats = an.series_stats(port, bench, rf_annual)
    pstats.update({k: v for k, v in an.beta_ols(port, bench).items() if k in ("beta_ci95_low", "beta_ci95_high", "r_squared")})
    pstats["ann_volatility_cov_check"] = float(np.sqrt(wh.values @ cov.values @ wh.values))
    asset_stats = pd.DataFrame(stats_rows).T
    dd = an.drawdown_stats(port)

    var_tab = rm.risk_table(port, confidences)
    forecasts: dict = {}
    rows = []
    if run_backtests and len(port) > window:
        for model in rm.MODELS:
            for c in confidences:
                f = bt.rolling_var_forecasts(port, window=window, confidence=c, model=model)
                forecasts[(model, c)] = f
                rows.append(bt.backtest_summary(f))
    bt_summary = pd.DataFrame(rows)

    px_held = prices[held]
    replays = [st.replay_event(px_held, wh, n, s, e) | {"description": d}
               for n, (s, e, d) in (events or STRESS_EVENTS).items()]
    hyp = [st.hypothetical_shock(wh, sh, n) for n, sh in (shocks or HYPOTHETICAL_SHOCKS).items()
           if all(h in sh for h in held)]
    yc = {}
    if yields is not None and not yields.empty:
        for n, (s, e, _) in (events or STRESS_EVENTS).items():
            yc[n] = st.yield_context(yields, s, e)

    assumptions = {"trading_days": TRADING_DAYS, "rf_annual": rf_annual, "benchmark": benchmark,
                   "backtest_window": window, "confidence_levels": list(confidences),
                   "portfolio": "constant weights, rebalanced daily, no transaction costs",
                   "var_sign": "positive loss thresholds, 1-day horizon",
                   "risk_contribution": "Euler decomposition of annualised volatility"}
    return AnalysisResult(w, prices, rets_all, port, info, asset_stats, pstats, corr, cov, rc, var_tab, bt_summary,
                          forecasts, replays, hyp, yc, dd, quality or {}, metadata or {}, assumptions)
