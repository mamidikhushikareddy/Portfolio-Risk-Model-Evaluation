import io

import numpy as np
import pandas as pd
import pytest

from riskplatform import ingestion as ing
from riskplatform.validation import quality_report_markdown, validate_prices


def _px():
    idx = pd.bdate_range("2020-01-01", periods=30)
    return pd.DataFrame({"AAA": np.linspace(100, 130, 30), "BBB": np.linspace(50, 40, 30)}, index=idx)


def test_duplicate_dates_detected():
    p = _px()
    p = pd.concat([p, p.iloc[[3]]]).sort_index()
    rep = validate_prices(p)
    assert rep["duplicate_date_rows"] == 1 and rep["status"] == "FAIL"
    assert any(i["code"] == "DUPLICATE_DATES" for i in rep["issues"])


def test_missing_values_and_nonpositive_flagged_not_filled():
    p = _px()
    p.iloc[5, 0] = np.nan
    p.iloc[10, 1] = -1.0
    before = p.copy()
    rep = validate_prices(p, ["AAA", "BBB", "CCC"])
    codes = {i["code"] for i in rep["issues"]}
    assert {"MISSING_VALUES", "NON_POSITIVE_PRICE", "MISSING_COLUMN"} <= codes
    assert rep["per_ticker"]["AAA"]["n_missing_inside_span"] == 1
    pd.testing.assert_frame_equal(p, before)  # input untouched: nothing deleted/filled


def test_large_return_flag_keeps_data():
    p = _px()
    p.iloc[12, 0] = p.iloc[11, 0] * 1.5
    rep = validate_prices(p)
    assert rep["per_ticker"]["AAA"]["n_large_return_flags"] >= 1
    assert len(p) == 30


def test_late_start_short_history_and_incomplete_end():
    p = _px()
    p.loc[p.index[:10], "BBB"] = np.nan
    rep = validate_prices(p, requested_start="2019-06-01", requested_end="2021-01-01")
    codes = {(i["ticker"], i["code"]) for i in rep["issues"]}
    assert ("BBB", "LATE_START") in codes and ("AAA", "SHORT_HISTORY") in codes and ("AAA", "INCOMPLETE_END") in codes


def test_calendar_diagnostics_and_markdown():
    rep = validate_prices(_px())
    assert rep["calendar"]["common_complete_rows"] == 30
    md = quality_report_markdown(rep)
    assert "Data Quality Report" in md and "AAA" in md


def test_gap_detection():
    p = _px()
    p = p.drop(p.index[10:16])
    rep = validate_prices(p)
    assert rep["per_ticker"]["AAA"]["large_gaps"]


def test_download_with_failure_and_cache(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=5)

    def dl(t, s, e):
        if t == "BAD":
            raise RuntimeError("boom")
        return pd.Series(np.arange(5) + 100.0, index=idx, name=t)

    res = ing.download_prices(["OK", "BAD"], "2020-01-01", None, cache_dir=tmp_path, downloader=dl)
    assert list(res.prices.columns) == ["OK"]
    assert res.metadata["unavailable"][0]["ticker"] == "BAD"
    assert "retrieved_at_utc" in res.metadata["per_ticker"]["OK"]

    def offline(t, s, e):
        raise RuntimeError("no network")

    again = ing.download_prices(["OK"], "2020-01-01", None, cache_dir=tmp_path, downloader=offline)
    assert again.metadata["per_ticker"]["OK"]["origin"] == "cache"  # same request served from cache
    stale = ing.download_prices(["OK"], "2019-01-01", None, cache_dir=tmp_path, downloader=offline)
    assert stale.metadata["per_ticker"]["OK"]["origin"] == "stale_cache"
    assert "warning" in stale.metadata["per_ticker"]["OK"]


def test_all_downloads_fail_gives_empty_frame(tmp_path):
    def offline(t, s, e):
        raise RuntimeError("no network")

    res = ing.download_prices(["A", "B"], "2020-01-01", downloader=offline, cache_dir=tmp_path)
    assert res.prices.empty and len(res.metadata["unavailable"]) == 2


def test_fred_parse_keeps_missing_as_nan():
    text = "observation_date,DGS10\n2020-01-01,.\n2020-01-02,1.88\n2020-01-03,1.80\n"
    s = ing.parse_fred_csv(text, "DGS10")
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 1.88


def test_fred_failure_is_graceful(tmp_path):
    def bad(*a):
        raise OSError("blocked")

    out = ing.download_fred(["DGS10"], "2020-01-01", fetcher=bad, cache_dir=tmp_path)
    assert out.prices.empty and out.metadata["unavailable"]


def test_csv_wide_and_long_and_duplicates(tmp_path):
    wide = "date,AAA,BBB\n2020-01-02,10,20\n2020-01-03,11,\n2020-01-03,11.5,21\n"
    with pytest.raises(ValueError, match="duplicate"):
        ing.read_price_csv(io.StringIO(wide))
    d = ing.read_price_csv(io.StringIO(wide), duplicates="keep_last")
    assert d.prices.loc["2020-01-03", "AAA"] == 11.5 and d.metadata["duplicates_found"] == 1
    gap = "date,AAA,BBB\n2020-01-02,10,20\n2020-01-03,11,\n"
    assert np.isnan(ing.read_price_csv(io.StringIO(gap)).prices.loc["2020-01-03", "BBB"])  # blank stays NaN
    long = "date,ticker,adj_close\n2020-01-02,AAA,10\n2020-01-02,BBB,20\n2020-01-03,AAA,11\n"
    lp = ing.read_price_csv(io.StringIO(long)).prices
    assert lp.shape == (2, 2) and np.isnan(lp.loc["2020-01-03", "BBB"])
    with pytest.raises(ValueError, match="date"):
        ing.read_price_csv(io.StringIO("x,y\n1,2\n"))
    with pytest.raises(ValueError):
        ing.read_price_csv(io.StringIO("date,AAA\n2020-01-02,abc\n"))


def test_prepare_prices_policy_validation():
    with pytest.raises(ValueError):
        ing.prepare_prices(_px(), duplicates="average")
