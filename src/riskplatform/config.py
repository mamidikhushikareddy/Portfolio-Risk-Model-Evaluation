"""Central configuration: universe, dates, conventions and scenario definitions."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("RISK_PLATFORM_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
IMPORT_DIR = DATA_DIR / "imports"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DUCKDB_PATH = DATA_DIR / "riskplatform.duckdb"

TRADING_DAYS = 252  # annualisation convention (documented in docs/METHODOLOGY.md)

TICKERS: dict[str, str] = {
    "VTI": "US equities",
    "EFA": "Developed international equities",
    "AGG": "US aggregate bonds",
    "TLT": "Long-duration US Treasuries",
    "GLD": "Gold",
    "EEM": "Emerging markets equities",
    "XLE": "Energy sector equities",
}
BENCHMARK = "SPY"
FRED_SERIES: dict[str, str] = {
    "DGS10": "10-Year Treasury constant maturity yield (percent)",
    "DGS2": "2-Year Treasury constant maturity yield (percent)",
}

# Approximate ETF launch dates (from memory of issuer data; VERIFY against issuer
# fact sheets before relying on them). Used only for warnings, never for data.
INCEPTION_DATES: dict[str, str] = {
    "VTI": "2001-05-24", "EFA": "2001-08-14", "AGG": "2003-09-22", "TLT": "2002-07-22",
    "GLD": "2004-11-18", "EEM": "2003-04-07", "XLE": "1998-12-16", "SPY": "1993-01-29",
}

DEFAULT_START = "2005-01-01"  # after the latest inception (GLD, Nov 2004)
DEFAULT_END: str | None = None  # None = today

# Illustrative hypothetical allocation used to populate the UI. NOT a recommendation.
DEFAULT_WEIGHTS: dict[str, float] = {
    "VTI": 0.30, "EFA": 0.10, "AGG": 0.25, "TLT": 0.10, "GLD": 0.10, "EEM": 0.10, "XLE": 0.05,
}

DEFAULT_RF_ANNUAL = 0.0
DEFAULT_BACKTEST_WINDOW = 500
CONFIDENCE_LEVELS = (0.95, 0.99)
LARGE_RETURN_THRESHOLD = 0.10  # |daily return| flag only; observations are never deleted
MIN_HISTORY_YEARS = 3.0
MAX_GAP_CALENDAR_DAYS = 5

# Historical event windows. Return convention: from the last close BEFORE `start`
# through the close on `end`. Dates are editable approximations of well-known episodes.
STRESS_EVENTS: dict[str, tuple[str, str, str]] = {
    "2008 financial crisis (equity peak to trough)": (
        "2007-10-09", "2009-03-09",
        "S&P 500 closing peak (2007-10-09) to closing trough (2009-03-09)."),
    "March 2020 market shock": (
        "2020-02-20", "2020-03-23",
        "First trading day after the 2020-02-19 S&P 500 peak to the 2020-03-23 trough."),
    "2022 interest-rate shock": (
        "2022-01-03", "2022-12-30",
        "Calendar-year 2022: rapid policy-rate increases, joint equity and bond losses."),
}

# Hypothetical shocks (NOT history). Fractions of price; user-editable in the UI.
HYPOTHETICAL_SHOCKS: dict[str, dict[str, float]] = {
    "Hypothetical equity sell-off (-20% equities, +3% Treasuries)": {
        "VTI": -0.20, "EFA": -0.22, "EEM": -0.25, "XLE": -0.20, "AGG": 0.01, "TLT": 0.03, "GLD": 0.02},
    "Hypothetical stagflation (equities and bonds fall)": {
        "VTI": -0.12, "EFA": -0.12, "EEM": -0.15, "XLE": 0.05, "AGG": -0.06, "TLT": -0.12, "GLD": 0.08},
}

# Duration approximation inputs. ASSUMPTIONS, not measured: check issuer fact sheets.
APPROX_DURATION: dict[str, float] = {"AGG": 6.0, "TLT": 17.0}
APPROX_CONVEXITY: dict[str, float] = {"AGG": 60.0, "TLT": 300.0}
