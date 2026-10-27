import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskplatform import backtesting as bt
from riskplatform import risk_measures as rm


@pytest.fixture()
def ramp():
    return pd.Series(-np.arange(1, 101) / 1000.0)  # losses 0.001 .. 0.100


def test_historical_var_es_known_sample(ramp):
    assert rm.historical_var(ramp, 0.95) == pytest.approx(0.09505)
    assert rm.historical_es(ramp, 0.95) == pytest.approx(np.mean([0.096, 0.097, 0.098, 0.099, 0.100]))
    assert rm.historical_es(ramp, 0.95) >= rm.historical_var(ramp, 0.95)


def test_sign_convention_gains_are_negative_loss():
    r = pd.Series(np.linspace(0.01, 0.05, 50))  # all gains
    assert rm.historical_var(r, 0.95) < 0  # negative loss threshold = positive return


def test_parametric_known_values():
    r = pd.Series([0.01, -0.01] * 100)
    sd = r.std(ddof=1)
    z = stats.norm.ppf(0.99)
    assert rm.parametric_var(r, 0.99) == pytest.approx(z * sd)  # mean is zero
    assert rm.parametric_es(r, 0.99) == pytest.approx(sd * stats.norm.pdf(z) / 0.01)
    assert rm.parametric_es(r, 0.99) > rm.parametric_var(r, 0.99)


def test_var_input_validation():
    with pytest.raises(ValueError):
        rm.historical_var(pd.Series([0.01, np.nan, 0.02]), 0.95)
    with pytest.raises(ValueError):
        rm.historical_var(pd.Series([0.01, 0.02]), 1.5)
    with pytest.raises(ValueError):
        rm.var_es(pd.Series([0.01, 0.02]), 0.95, "bogus")


def _series(n=700, seed=3):
    r = np.random.default_rng(seed).standard_t(5, n) * 0.01
    return pd.Series(r, index=pd.bdate_range("2018-01-01", periods=n))


def test_forecast_timing_and_no_lookahead():
    r = _series()
    f = bt.rolling_var_forecasts(r, window=250, confidence=0.95, model="historical")
    first = f.index[0]
    assert first == r.index[250] and f["forecast_date"].iloc[0] == r.index[249]
    assert (f["forecast_date"] < f.index).all()
    # Recompute manually: forecast for realised date i uses only r[i-250:i]
    manual = rm.historical_var(r.iloc[0:250], 0.95)
    assert f["var_forecast"].iloc[0] == pytest.approx(manual)
    # Lookahead check: changing returns at/after date t must not alter forecasts made earlier
    t = 400
    r2 = r.copy()
    r2.iloc[t:] = r2.iloc[t:] * 10 - 0.05
    f2 = bt.rolling_var_forecasts(r2, window=250, confidence=0.95, model="historical")
    same = f.loc[: r.index[t]].index
    assert f.loc[same, "var_forecast"].equals(f2.loc[same, "var_forecast"])
    assert not f.loc[r.index[t + 1:], "var_forecast"].equals(f2.loc[r.index[t + 1:], "var_forecast"])


def test_breach_counting():
    n = 100
    idx = pd.bdate_range("2020-01-01", periods=n + 40)
    r = pd.Series(0.001 * np.where(np.arange(n + 40) % 2 == 0, 1.0, -1.0), index=idx)  # losses never exceed 0.001
    r.iloc[60] = -0.5
    r.iloc[90] = -0.6
    f = bt.rolling_var_forecasts(r, window=40, confidence=0.95)
    assert f["breach"].sum() == 2
    assert f.loc[idx[60], "breach"] and f.loc[idx[90], "breach"]
    assert f.loc[idx[60], "realised_loss"] == pytest.approx(0.5)
    s = bt.backtest_summary(f)
    assert s["breaches"] == 2 and s["n_forecasts"] == len(f) and s["breach_rate"] == pytest.approx(2 / len(f))


def test_forecast_input_guards():
    r = _series(100)
    with pytest.raises(ValueError):
        bt.rolling_var_forecasts(r, window=100)
    with pytest.raises(ValueError):
        bt.rolling_var_forecasts(r.where(r.index != r.index[5]), window=30)
    dup = pd.concat([r, r.iloc[[0]]])
    with pytest.raises(ValueError):
        bt.rolling_var_forecasts(dup, window=30)


def test_kupiec_known_values():
    lr, p = bt.kupiec_pof(5, 100, 0.95)  # exactly expected -> LR 0
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    lr, p = bt.kupiec_pof(15, 100, 0.95)
    expected = -2 * ((85 * np.log(0.95) + 15 * np.log(0.05)) - (85 * np.log(0.85) + 15 * np.log(0.15)))
    assert lr == pytest.approx(expected) and p < 0.01
    lr0, p0 = bt.kupiec_pof(0, 250, 0.99)  # zero breaches handled without log(0)
    assert np.isfinite(lr0) and 0 < p0 < 1


def test_clopper_pearson_and_summary_fields():
    lo, hi = bt.clopper_pearson(0, 100)
    assert lo == 0.0 and 0.03 < hi < 0.05
    f = bt.rolling_var_forecasts(_series(900), window=300, confidence=0.99, model="parametric")
    s = bt.backtest_summary(f)
    for k in ("kupiec_p_value", "breach_rate_ci95_low", "expected_breaches", "notes", "es_check"):
        assert k in s
    assert s["breach_rate_ci95_low"] <= s["breach_rate"] <= s["breach_rate_ci95_high"]
    assert "Expected Shortfall" in s["notes"]


def test_christoffersen_detects_clustering():
    clustered = np.zeros(500, dtype=int)
    clustered[100:110] = 1
    clustered[300:310] = 1
    _, p_c = bt.christoffersen_independence(clustered)
    spread = np.zeros(500, dtype=int)
    spread[::25] = 1
    _, p_s = bt.christoffersen_independence(spread)
    assert p_c < 0.01 < p_s
    assert np.isnan(bt.christoffersen_independence(np.zeros(50))[0])
