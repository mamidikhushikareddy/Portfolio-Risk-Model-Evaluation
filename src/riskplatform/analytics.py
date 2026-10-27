"""Return and portfolio analytics. Each function is independently testable.

Conventions
-----------
* Returns are decimals (0.01 = 1%). Prices -> returns use p_t / p_{t-1} - 1 on the
  shared date index; if either price is missing the return is NaN (never zero).
* Annualisation: arithmetic mean * 252, volatility * sqrt(252). Risk-free rate is an
  annual decimal converted to a daily rate by dividing by 252.
* Portfolio returns assume constant weights rebalanced daily.
* Risk contributions use the Euler decomposition of annualised portfolio volatility.
"""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

from .config import TRADING_DAYS

log = logging.getLogger(__name__)


def _clean(returns: pd.Series | np.ndarray, min_obs: int = 1) -> pd.Series:
    """Drop NaN explicitly (logged); never zero-fill."""
    s = pd.Series(returns, dtype=float)
    n = int(s.isna().sum())
    if n:
        log.debug("Dropping %d missing returns", n)
        s = s.dropna()
    if len(s) < min_obs:
        raise ValueError(f"Need at least {min_obs} valid return observations, got {len(s)}")
    return s


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Daily simple returns p_t/p_{t-1}-1; NaN where either price is missing."""
    if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else (prices <= 0).any():
        raise ValueError("Non-positive prices found; returns are undefined")
    return prices / prices.shift(1) - 1


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Daily log returns ln(p_t/p_{t-1})."""
    if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else (prices <= 0).any():
        raise ValueError("Non-positive prices found; log returns are undefined")
    return np.log(prices / prices.shift(1))


def align_returns(returns: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Keep only dates where every requested column has a return (complete-case)."""
    missing = [c for c in columns if c not in returns.columns]
    if missing:
        raise KeyError(f"Columns not in returns: {missing}")
    sub = returns[columns]
    out = sub.dropna(how="any")
    if out.empty:
        raise ValueError("No overlapping dates with complete returns for: " + ", ".join(columns))
    info = {"rows_before": int(len(sub)), "rows_after": int(len(out)),
            "rows_dropped_missing": int(len(sub) - len(out)),
            "start": out.index[0], "end": out.index[-1]}
    return out, info


def annualized_mean_return(returns, periods: int = TRADING_DAYS) -> float:
    """Arithmetic annualised mean = mean daily return * periods."""
    return float(_clean(returns, 1).mean() * periods)


def annualized_volatility(returns, periods: int = TRADING_DAYS) -> float:
    """Sample (ddof=1) daily standard deviation * sqrt(periods)."""
    return float(_clean(returns, 2).std(ddof=1) * np.sqrt(periods))


def cumulative_growth(returns: pd.Series, initial: float = 10_000.0) -> pd.Series:
    """Value of a hypothetical investment. Raises if any return is missing."""
    if returns.isna().any():
        raise ValueError("cumulative_growth requires complete returns (no NaN)")
    return initial * (1 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from running peak (peak includes the initial value 1.0). Values <= 0."""
    if returns.isna().any():
        raise ValueError("drawdown_series requires complete returns (no NaN)")
    wealth = (1 + returns).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1


def drawdown_stats(returns: pd.Series) -> dict:
    """Maximum drawdown with peak/trough/recovery dates and durations (trading days)."""
    dd = drawdown_series(returns)
    if dd.empty:
        raise ValueError("No returns supplied")
    wealth = (1 + returns).cumprod()
    peak_level = np.maximum(wealth.cummax(), 1.0)
    trough_pos = int(np.argmin(dd.values))
    trough_date = dd.index[trough_pos]
    max_dd = float(dd.iloc[trough_pos])
    under = dd < 0
    run_len = under.groupby((~under).cumsum()).cumsum()
    out = {"max_drawdown": max_dd, "trough_date": trough_date, "peak_date": None, "recovery_date": None,
           "peak_to_trough_days": None, "peak_to_recovery_days": None,
           "longest_underwater_days": int(run_len.max()), "current_drawdown": float(dd.iloc[-1])}
    if max_dd == 0:
        return out
    level = float(peak_level.iloc[trough_pos])
    before = wealth.iloc[: trough_pos + 1]
    hits = np.where(before.values >= level - 1e-12)[0]
    peak_pos = int(hits[-1]) if len(hits) else -1  # -1 = the initial value before the sample
    out["peak_date"] = dd.index[peak_pos] if peak_pos >= 0 else None
    out["peak_to_trough_days"] = trough_pos - peak_pos
    after = wealth.iloc[trough_pos + 1:]
    rec = np.where(after.values >= level - 1e-12)[0]
    if len(rec):
        rec_pos = trough_pos + 1 + int(rec[0])
        out["recovery_date"] = dd.index[rec_pos]
        out["peak_to_recovery_days"] = rec_pos - peak_pos
    return out


def beta(asset: pd.Series, benchmark: pd.Series) -> float:
    """OLS beta = cov(asset, benchmark) / var(benchmark) on dates where both exist."""
    both = pd.concat([asset, benchmark], axis=1, join="inner").dropna()
    if len(both) < 3:
        raise ValueError("Need at least 3 overlapping observations to estimate beta")
    var_b = both.iloc[:, 1].var(ddof=1)
    if var_b == 0:
        raise ValueError("Benchmark variance is zero; beta undefined")
    return float(both.iloc[:, 0].cov(both.iloc[:, 1]) / var_b)


def beta_ols(asset: pd.Series, benchmark: pd.Series, maxlags: int = 5) -> dict:
    """OLS beta with Newey-West (HAC) standard error and 95% interval (statsmodels).

    The point estimate equals ``beta`` (covariance formula); the interval shows how
    uncertain it is. HAC errors allow for autocorrelation/heteroskedasticity in daily returns.
    """
    import statsmodels.api as sm

    both = pd.concat([asset, benchmark], axis=1, join="inner").dropna()
    if len(both) < 30:
        raise ValueError("Need at least 30 overlapping observations for an OLS beta interval")
    fit = sm.OLS(both.iloc[:, 0].values, sm.add_constant(both.iloc[:, 1].values)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    lo, hi = fit.conf_int(alpha=0.05)[1]
    return {"beta": float(fit.params[1]), "alpha_daily": float(fit.params[0]), "beta_se_hac": float(fit.bse[1]),
            "beta_ci95_low": float(lo), "beta_ci95_high": float(hi), "r_squared": float(fit.rsquared)}


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation on complete-case rows."""
    clean = returns.dropna(how="any")
    if len(clean) < 3:
        raise ValueError("Need at least 3 complete rows for a correlation matrix")
    return clean.corr()


def covariance_matrix(returns: pd.DataFrame, annualize: bool = True, periods: int = TRADING_DAYS) -> pd.DataFrame:
    """Sample covariance (ddof=1) on complete-case rows; annualised by * periods."""
    clean = returns.dropna(how="any")
    if len(clean) < 3:
        raise ValueError("Need at least 3 complete rows for a covariance matrix")
    cov = clean.cov()
    return cov * periods if annualize else cov


def downside_deviation(returns, mar_annual: float = 0.0, periods: int = TRADING_DAYS, annualize: bool = True) -> float:
    """sqrt(mean(min(r - MAR_daily, 0)^2)) over ALL observations; * sqrt(periods) if annualised."""
    r = _clean(returns, 2)
    shortfall = np.minimum(r.values - mar_annual / periods, 0.0)
    dd = float(np.sqrt(np.mean(shortfall ** 2)))
    return dd * np.sqrt(periods) if annualize else dd


def sharpe_ratio(returns, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe: mean(excess)/std(excess)*sqrt(periods); daily rf = rf_annual/periods."""
    ex = _clean(returns, 2) - rf_annual / periods
    sd = ex.std(ddof=1)
    return float("nan") if sd == 0 else float(ex.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised excess return over annualised downside deviation (MAR = risk-free rate)."""
    r = _clean(returns, 2)
    dd = downside_deviation(r, rf_annual, periods, annualize=True)
    if dd == 0:
        return float("nan")
    return float((r.mean() - rf_annual / periods) * periods / dd)


def weights_from_percent(percent: Mapping[str, float]) -> pd.Series:
    return pd.Series({k: v / 100.0 for k, v in percent.items()}, dtype=float)


def validate_weights(weights: Mapping[str, float] | pd.Series, tickers: list[str] | None = None,
                     tol: float = 1e-6, allow_short: bool = False) -> pd.Series:
    """Validate fractional weights: finite, long-only (default), summing to 100%."""
    w = pd.Series(weights, dtype=float)
    if tickers is not None:
        missing, extra = set(tickers) - set(w.index), set(w.index) - set(tickers)
        if missing or extra:
            raise ValueError(f"Weights/tickers mismatch. Missing: {sorted(missing)}; unexpected: {sorted(extra)}")
        w = w.reindex(tickers)
    if w.empty:
        raise ValueError("No weights supplied")
    if not np.isfinite(w.values).all():
        raise ValueError("Weights must be finite numbers")
    if not allow_short and (w < 0).any():
        raise ValueError(f"Negative weights not allowed (long-only): {w[w < 0].to_dict()}")
    total = float(w.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights sum to {total:.4%}, must sum to 100%")
    return w


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted sum of asset returns (daily rebalanced). Raises if any needed return is NaN."""
    cols = list(weights.index)
    missing = [c for c in cols if c not in returns.columns]
    if missing:
        raise KeyError(f"Returns missing for: {missing}")
    sub = returns[cols]
    if sub.isna().any().any():
        raise ValueError(f"{int(sub.isna().any(axis=1).sum())} dates have missing returns; "
                         "align returns first (missing values are never treated as zero)")
    return pd.Series(sub.values @ weights.values, index=sub.index, name="portfolio")


def risk_contributions(cov_annual: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility.

    marginal_i = (Sigma w)_i / sigma_p ; component_i = w_i * marginal_i ;
    pct_i = component_i / sigma_p. Components sum to sigma_p; pct sums to 100%.
    """
    w = weights.reindex(cov_annual.index)
    if w.isna().any():
        raise ValueError("Weights missing for covariance assets: " + str(list(w.index[w.isna()])))
    sigma_sq = float(w.values @ cov_annual.values @ w.values)
    if sigma_sq <= 0:
        raise ValueError("Portfolio variance is not positive; risk contributions undefined")
    sigma = np.sqrt(sigma_sq)
    marginal = pd.Series(cov_annual.values @ w.values / sigma, index=w.index)
    component = w * marginal
    return pd.DataFrame({"weight": w, "marginal_contribution": marginal,
                         "component_contribution": component, "pct_contribution": component / sigma})


def max_drawdown_from_prices(prices: pd.Series) -> float:
    """Maximum drawdown from observed prices (no returns are bridged across data gaps)."""
    p = prices.dropna()
    if len(p) < 2 or (p <= 0).any():
        raise ValueError("Need at least 2 positive prices for a drawdown")
    return float((p / p.cummax() - 1).min())


def series_stats(returns: pd.Series, benchmark: pd.Series | None = None, rf_annual: float = 0.0,
                 prices: pd.Series | None = None) -> dict:
    """Summary statistics for one return series (uses its own valid observations).

    If ``prices`` is given, max drawdown is computed from prices, which stays correct when
    the return series has gaps (a missing price removes two returns).
    """
    r = returns.dropna()
    if len(r) < 3:
        raise ValueError("Need at least 3 observations for summary statistics")
    out = {"start": r.index[0], "end": r.index[-1], "n_obs": int(len(r)),
           "ann_return_arith": annualized_mean_return(r), "ann_volatility": annualized_volatility(r),
           "sharpe": sharpe_ratio(r, rf_annual), "sortino": sortino_ratio(r, rf_annual),
           "downside_deviation": downside_deviation(r, rf_annual),
           "max_drawdown": max_drawdown_from_prices(prices) if prices is not None else drawdown_stats(r)["max_drawdown"],
           "worst_day": float(r.min()), "best_day": float(r.max())}
    if benchmark is not None:
        both = pd.concat([r, benchmark], axis=1, join="inner").dropna()
        out["beta_vs_benchmark"] = beta(both.iloc[:, 0], both.iloc[:, 1]) if len(both) >= 3 else float("nan")
        out["corr_vs_benchmark"] = float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) >= 3 else float("nan")
    return out
