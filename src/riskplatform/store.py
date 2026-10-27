"""Optional DuckDB persistence for downloaded prices and quality reports."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from .config import DUCKDB_PATH


def save_prices(prices: pd.DataFrame, quality: dict | None = None, path: Path = DUCKDB_PATH) -> Path:
    """Persist long-format prices and the latest quality report to a local DuckDB file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    long = prices.rename_axis("date").reset_index().melt(id_vars="date", var_name="ticker", value_name="price")
    long = long.dropna(subset=["price"])
    with duckdb.connect(str(path)) as con:
        con.register("long_df", long)
        con.execute("CREATE OR REPLACE TABLE prices_long AS SELECT * FROM long_df")
        if quality is not None:
            con.execute("CREATE OR REPLACE TABLE quality_report AS SELECT ? AS report_json", [json.dumps(quality, default=str)])
    return path


def load_prices(path: Path = DUCKDB_PATH) -> pd.DataFrame:
    with duckdb.connect(str(path), read_only=True) as con:
        long = con.execute("SELECT * FROM prices_long").df()
    return long.pivot(index="date", columns="ticker", values="price").sort_index()
