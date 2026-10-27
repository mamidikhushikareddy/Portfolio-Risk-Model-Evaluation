"""Rolling out-of-sample one-day-ahead VaR forecasts and breach testing.

Timing: the forecast stored on row ``t`` (realised date) uses ONLY returns dated
strictly before ``t`` (window ending at ``forecast_date``, the prior trading day).
A breach occurs when the realised loss on ``t`` strictly exceeds the forecast VaR.

A low breach count does not prove a model is reliable: with 99% VaR and a few
thousand days the expected breach count is small, so tests have low power, and a
model can pass unconditional coverage while breaches cluster or ES is badly wrong.
ES here is reported descriptively only; a passed VaR test does NOT validate ES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from .risk_measures import var_es


def rolling_var_forecasts(returns: pd.Series, window: int = 500, confidence: float = 0.99,
                          model: str = "historical", use_mean: bool = True) -> pd.DataFrame:
    """Return DataFrame indexed by realised date with forecast VaR/ES and breach flags."""
    r = pd.Series(returns, dtype=float)
    if r.isna().any():
        raise ValueError("Backtest returns must be complete (no NaN)")
    if not r.index.is_monotonic_increasing or r.index.has_duplicates:
        raise ValueError("Backtest returns need a sorted, duplicate-free date index")
    if window < 30:
        raise ValueError("window must be at least 30 observations")
    if len(r) <= window:
        raise ValueError(f"Need more than window={window} observations; have {len(r)}")
    vals = r.values
    rows = []
    for i in range(window, len(r)):
        history = vals[i - window:i]  # strictly before index i
        v, e = var_es(history, confidence, model, use_mean)
        rows.append((r.index[i], r.index[i - 1], v, e, -vals[i]))
    out = pd.DataFrame(rows, columns=["realised_date", "forecast_date", "var_forecast", "es_forecast",
                                      "realised_loss"]).set_index("realised_date")
    out["breach"] = out["realised_loss"] > out["var_forecast"]
    out.attrs.update(model=model, confidence=confidence, window=window)
    return out


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval for a proportion."""
    lo = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    return float(lo), float(hi)


def kupiec_pof(breaches: int, n: int, confidence: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR test (unconditional coverage). Returns (LR, p-value)."""
    if n <= 0 or not 0 <= breaches <= n:
        raise ValueError("Invalid breach count / sample size")
    p = 1 - confidence
    phat = breaches / n
    ll0 = xlogy(n - breaches, 1 - p) + xlogy(breaches, p)
    ll1 = xlogy(n - breaches, 1 - phat) + xlogy(breaches, phat)
    lr = float(max(-2 * (ll0 - ll1), 0.0))
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence LR test on a 0/1 breach sequence. NaN if undefined."""
    h = np.asarray(hits, dtype=int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return float("nan"), float("nan")
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = float(max(-2 * (ll0 - ll1), 0.0))
    return lr, float(stats.chi2.sf(lr, 1))


def backtest_summary(forecasts: pd.DataFrame, confidence: float | None = None, model: str | None = None) -> dict:
    """Breach statistics, exact CI, Kupiec/Christoffersen tests and descriptive ES check."""
    conf = confidence if confidence is not None else forecasts.attrs.get("confidence")
    mod = model or forecasts.attrs.get("model")
    n = int(len(forecasts))
    x = int(forecasts["breach"].sum())
    expected_rate = 1 - conf
    lo, hi = clopper_pearson(x, n)
    lr, p = kupiec_pof(x, n, conf)
    ind_lr, ind_p = christoffersen_independence(forecasts["breach"].values)
    notes = []
    if n * expected_rate < 5:
        notes.append("Expected breaches < 5: coverage test has very low power.")
    notes.append("Passing a VaR test does not validate Expected Shortfall.")
    breach_rows = forecasts[forecasts["breach"]]
    return {"model": mod, "confidence": conf, "n_forecasts": n, "breaches": x,
            "breach_rate": x / n, "expected_breach_rate": expected_rate,
            "expected_breaches": n * expected_rate, "breach_rate_ci95_low": lo, "breach_rate_ci95_high": hi,
            "expected_rate_inside_ci": bool(lo <= expected_rate <= hi),
            "kupiec_lr": lr, "kupiec_p_value": p, "kupiec_reject_5pct": bool(p < 0.05),
            "christoffersen_ind_lr": ind_lr, "christoffersen_ind_p_value": ind_p,
            "mean_forecast_var": float(forecasts["var_forecast"].mean()),
            "mean_realised_loss_on_breach": float(breach_rows["realised_loss"].mean()) if x else float("nan"),
            "mean_forecast_es_on_breach_days": float(breach_rows["es_forecast"].mean()) if x else float("nan"),
            "es_check": "descriptive only (not a statistical ES backtest)", "notes": " ".join(notes)}
