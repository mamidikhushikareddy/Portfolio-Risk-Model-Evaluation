"""Value at Risk and Expected Shortfall.

Sign convention: VaR and ES are reported as POSITIVE LOSSES (fraction of portfolio
value). A return of -2% is a loss of +0.02. Inputs are signed returns.

* Historical VaR(c)  = empirical c-quantile of losses (numpy 'linear' interpolation).
* Historical ES(c)   = mean of losses >= VaR(c) (tail average, inclusive).
* Parametric (normal): VaR = z_c * sigma - mu ; ES = sigma * phi(z_c)/(1-c) - mu,
  with daily mu and sigma (ddof=1). Normality is a strong assumption; real returns
  have fatter tails, so parametric normal numbers typically understate extreme losses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MODELS = ("historical", "parametric")


def _losses(returns) -> np.ndarray:
    r = np.asarray(pd.Series(returns, dtype=float).values)
    if np.isnan(r).any():
        raise ValueError("Returns contain NaN; VaR/ES do not impute missing values")
    if r.size < 2:
        raise ValueError("Need at least 2 return observations")
    return -r


def _check_conf(c: float) -> None:
    if not 0.5 < c < 1.0:
        raise ValueError(f"confidence must be in (0.5, 1), got {c}")


def historical_var(returns, confidence: float) -> float:
    _check_conf(confidence)
    return float(np.quantile(_losses(returns), confidence))


def historical_es(returns, confidence: float) -> float:
    _check_conf(confidence)
    losses = _losses(returns)
    var = np.quantile(losses, confidence)
    return float(losses[losses >= var].mean())


def parametric_var(returns, confidence: float, use_mean: bool = True) -> float:
    _check_conf(confidence)
    r = -_losses(returns)
    mu = r.mean() if use_mean else 0.0
    return float(stats.norm.ppf(confidence) * r.std(ddof=1) - mu)


def parametric_es(returns, confidence: float, use_mean: bool = True) -> float:
    _check_conf(confidence)
    r = -_losses(returns)
    mu = r.mean() if use_mean else 0.0
    z = stats.norm.ppf(confidence)
    return float(r.std(ddof=1) * stats.norm.pdf(z) / (1 - confidence) - mu)


def var_es(returns, confidence: float, model: str, use_mean: bool = True) -> tuple[float, float]:
    """(VaR, ES) as positive losses for the chosen model."""
    if model == "historical":
        return historical_var(returns, confidence), historical_es(returns, confidence)
    if model == "parametric":
        return parametric_var(returns, confidence, use_mean), parametric_es(returns, confidence, use_mean)
    raise ValueError(f"Unknown model '{model}'. Choose from {MODELS}")


def risk_table(returns, confidences=(0.95, 0.99), horizon_note: str = "1-day") -> pd.DataFrame:
    """Full-sample VaR/ES table (in-sample, descriptive: NOT a forecast evaluation)."""
    rows = []
    n = len(pd.Series(returns).dropna())
    for model in MODELS:
        for c in confidences:
            v, e = var_es(returns, c, model)
            rows.append({"model": model, "confidence": c, "var": v, "es": e,
                         "tail_observations": int(round(n * (1 - c))), "horizon": horizon_note})
    return pd.DataFrame(rows)
