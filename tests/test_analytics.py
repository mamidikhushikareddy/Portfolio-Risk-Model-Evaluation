import numpy as np
import pandas as pd
import pytest

from riskplatform import analytics as an


def test_simple_and_log_returns(small_prices):
    r = an.simple_returns(small_prices["A"])
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10) and r.iloc[2] == pytest.approx(-0.10)
    assert an.log_returns(small_prices["A"]).iloc[1] == pytest.approx(np.log(1.1))


def test_missing_price_gives_nan_return_not_zero():
    s = pd.Series([100.0, np.nan, 110.0], index=pd.bdate_range("2020-01-01", periods=3))
    r = an.simple_returns(s)
    assert r.isna().tolist() == [True, True, True]  # return after a gap is NOT bridged or zero-filled


def test_non_positive_prices_rejected():
    with pytest.raises(ValueError):
        an.simple_returns(pd.Series([100.0, 0.0, 5.0]))
    with pytest.raises(ValueError):
        an.log_returns(pd.Series([100.0, -1.0]))


def test_volatility_annualisation():
    r = pd.Series([0.01, -0.01] * 50)
    daily = r.std(ddof=1)
    assert an.annualized_volatility(r) == pytest.approx(daily * np.sqrt(252))
    assert an.annualized_mean_return(pd.Series([0.001] * 10)) == pytest.approx(0.252)
    with pytest.raises(ValueError):
        an.annualized_volatility(pd.Series([0.01]))


def test_cumulative_growth_and_nan_guard():
    g = an.cumulative_growth(pd.Series([0.1, -0.1]), initial=100)
    assert g.tolist() == pytest.approx([110.0, 99.0])
    with pytest.raises(ValueError):
        an.cumulative_growth(pd.Series([0.1, np.nan]))


def test_drawdown_known_path():
    idx = pd.bdate_range("2020-01-01", periods=6)
    r = pd.Series([0.10, -0.20, -0.10, 0.50, 0.10, 0.0], index=idx)
    s = an.drawdown_stats(r)
    # wealth: 1.1, 0.88, 0.792, 1.188, 1.3068 ; max dd = 0.792/1.1-1 = -0.28
    assert s["max_drawdown"] == pytest.approx(-0.28)
    assert s["peak_date"] == idx[0] and s["trough_date"] == idx[2] and s["recovery_date"] == idx[3]
    assert s["peak_to_trough_days"] == 2 and s["peak_to_recovery_days"] == 3
    assert s["longest_underwater_days"] == 2


def test_drawdown_unrecovered_and_initial_peak():
    idx = pd.bdate_range("2020-01-01", periods=3)
    s = an.drawdown_stats(pd.Series([-0.1, -0.1, 0.05], index=idx))
    assert s["peak_date"] is None and s["recovery_date"] is None
    assert s["max_drawdown"] == pytest.approx(0.9 * 0.9 - 1)
    assert s["current_drawdown"] < 0


def test_beta_and_correlation_known():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0, 0.01, 500))
    a = 1.5 * b
    assert an.beta(a, b) == pytest.approx(1.5)
    assert an.correlation_matrix(pd.DataFrame({"a": a, "b": b})).loc["a", "b"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        an.beta(pd.Series([0.1, 0.2, 0.3]), pd.Series([0.0, 0.0, 0.0]))


def test_covariance_annualised():
    df = pd.DataFrame({"a": [0.01, -0.02, 0.03, 0.0], "b": [0.0, 0.01, -0.01, 0.02]})
    assert an.covariance_matrix(df).values == pytest.approx(df.cov().values * 252)


def test_sharpe_sortino_downside():
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0, 0.01])
    ex = r - 0.02 / 252
    assert an.sharpe_ratio(r, 0.02) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))
    dd_daily = np.sqrt(np.mean(np.minimum(r.values - 0.0, 0) ** 2))
    assert an.downside_deviation(r, 0.0, annualize=False) == pytest.approx(dd_daily)
    assert an.sortino_ratio(r, 0.0) == pytest.approx(r.mean() * 252 / (dd_daily * np.sqrt(252)))
    assert np.isnan(an.sharpe_ratio(pd.Series([0.01] * 5)))


def test_weight_validation():
    assert an.validate_weights({"A": 0.6, "B": 0.4}, ["A", "B"]).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="sum to"):
        an.validate_weights({"A": 0.6, "B": 0.3})
    with pytest.raises(ValueError, match="Negative"):
        an.validate_weights({"A": 1.2, "B": -0.2})
    with pytest.raises(ValueError, match="mismatch"):
        an.validate_weights({"A": 1.0}, ["A", "B"])
    with pytest.raises(ValueError, match="finite"):
        an.validate_weights({"A": np.nan, "B": 1.0})


def test_portfolio_returns_and_missing_guard(small_prices):
    r = an.simple_returns(small_prices)
    aligned, info = an.align_returns(r, ["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.5})
    p = an.portfolio_returns(aligned, w)
    assert p.iloc[0] == pytest.approx(0.5 * 0.10 + 0.5 * 0.0)
    assert info["rows_dropped_missing"] == 1  # first row has no return
    with pytest.raises(ValueError, match="missing"):
        an.portfolio_returns(r, w)  # contains NaN first row: never zero-filled


def test_align_edge_cases():
    r = pd.DataFrame({"A": [np.nan, 0.01], "B": [np.nan, np.nan]})
    with pytest.raises(ValueError):
        an.align_returns(r, ["A", "B"])
    with pytest.raises(KeyError):
        an.align_returns(r, ["Z"])


def test_risk_contribution_reconciliation(prices, weights):
    r = an.simple_returns(prices[list(weights.index)]).dropna()
    cov = an.covariance_matrix(r)
    rc = an.risk_contributions(cov, weights)
    sigma = np.sqrt(weights.values @ cov.values @ weights.values)
    assert rc["component_contribution"].sum() == pytest.approx(sigma)
    assert rc["pct_contribution"].sum() == pytest.approx(1.0)
    port = an.portfolio_returns(r, weights)
    assert an.annualized_volatility(port) == pytest.approx(sigma)


def test_risk_contribution_two_assets_known():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["a", "b"], columns=["a", "b"])
    rc = an.risk_contributions(cov, pd.Series({"a": 0.5, "b": 0.5}))
    # sigma = sqrt(0.25*0.04+0.25*0.01)=sqrt(0.0125); a share = 0.01/0.0125 = 80%
    assert rc.loc["a", "pct_contribution"] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        an.risk_contributions(cov, pd.Series({"a": 0.0, "b": 0.0}))
