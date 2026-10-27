"""Model-risk demonstration on SYNTHETIC data (seeded, reproducible).

Question: if real returns have fat tails, does a normal-distribution VaR look reliable
under a conventional out-of-sample backtest?
Setup: 5,000 daily returns from a Student-t (3 d.o.f.) scaled to 1% daily volatility,
a 500-day rolling window and 99% one-day VaR. The data-generating process is known, so
any model failure is attributable to the model, not to data noise.

Run:  python examples/model_risk_demo.py
"""
import numpy as np
import pandas as pd

from riskplatform.backtesting import backtest_summary, rolling_var_forecasts

DOF, N, WINDOW, CONF, SEED = 3, 5000, 500, 0.99, 11


def make_fat_tailed_returns(n: int = N, dof: int = DOF, vol: float = 0.01, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.standard_t(dof, n) * vol / np.sqrt(dof / (dof - 2))  # rescale to exact target volatility
    return pd.Series(r, index=pd.bdate_range("2000-01-03", periods=n))


def run() -> pd.DataFrame:
    r = make_fat_tailed_returns()
    rows = []
    for model in ("historical", "parametric"):
        s = backtest_summary(rolling_var_forecasts(r, WINDOW, CONF, model))
        rows.append({"model": f"{model} VaR", "breaches": s["breaches"], "expected": round(s["expected_breaches"], 1),
                     "breach rate": f"{s['breach_rate']:.2%}",
                     "95% CI": f"{s['breach_rate_ci95_low']:.2%}-{s['breach_rate_ci95_high']:.2%}",
                     "Kupiec p": round(s["kupiec_p_value"], 4),
                     "loss on breach / forecast ES": round(s["mean_realised_loss_on_breach"] / s["mean_forecast_es_on_breach_days"], 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"Synthetic Student-t({DOF}) returns, {N:,} days, {WINDOW}-day window, {CONF:.0%} 1-day VaR\n")
    print(run().to_string(index=False))
