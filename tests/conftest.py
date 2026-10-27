import pandas as pd
import pytest

from riskplatform.config import DEFAULT_WEIGHTS, TICKERS
from riskplatform.synthetic import make_synthetic_prices, make_synthetic_yields


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return make_synthetic_prices()


@pytest.fixture(scope="session")
def yields(prices) -> pd.DataFrame:
    return make_synthetic_yields(prices.index)


@pytest.fixture()
def weights() -> pd.Series:
    return pd.Series({t: DEFAULT_WEIGHTS[t] for t in TICKERS})


@pytest.fixture()
def small_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=6)
    return pd.DataFrame({"A": [100, 110, 99, 99, 108.9, 108.9], "B": [50, 50, 55, 44, 44, 48.4]}, index=idx)
