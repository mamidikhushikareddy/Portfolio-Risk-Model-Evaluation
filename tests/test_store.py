import numpy as np
import pandas as pd

from riskplatform.store import load_prices, save_prices


def test_duckdb_roundtrip_keeps_missing_as_missing(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=4)
    p = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0], "B": [5.0, 6.0, 7.0, 8.0]}, index=idx)
    db = save_prices(p, {"status": "PASS"}, tmp_path / "t.duckdb")
    back = load_prices(db)
    assert np.isnan(back.loc[idx[2], "A"]) and back.loc[idx[3], "B"] == 8.0
    assert back.shape == (4, 2)
