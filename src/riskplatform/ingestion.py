"""Data ingestion: yfinance (unofficial), FRED public CSV, local CSV, with CSV caching.

yfinance scrapes Yahoo Finance and is NOT an official or guaranteed data source.
Prices fetched with ``auto_adjust=True`` are split/dividend adjusted closes, so
returns derived from them approximate total returns. Missing prices are NEVER
forward-filled here.
"""
from __future__ import annotations

import io
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .config import CACHE_DIR

log = logging.getLogger(__name__)

Downloader = Callable[[str, str | None, str | None], pd.Series]


@dataclass
class PriceData:
    """Wide price table (date x ticker) plus provenance metadata."""

    prices: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _yfinance_download(ticker: str, start: str | None, end: str | None) -> pd.Series:
    """Download adjusted closes for one ticker. Raises on failure/empty result."""
    import yfinance as yf  # imported lazily so offline tests never need it

    # yfinance treats `end` as exclusive; add a day so the requested end date is included.
    end_incl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else None
    df = yf.download(ticker, start=start, end=end_incl, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns MultiIndex columns
        close = close.iloc[:, 0]
    close = close.astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = ticker
    return close


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.csv", cache_dir / f"{key}.meta.json"


def _write_cache(cache_dir: Path, key: str, s: pd.Series, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _cache_paths(cache_dir, key)
    s.rename("value").rename_axis("date").to_csv(csv_path)
    meta_path.write_text(json.dumps(meta, indent=2))


def _read_cache(cache_dir: Path, key: str) -> tuple[pd.Series, dict] | None:
    csv_path, meta_path = _cache_paths(cache_dir, key)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    s = df["value"].astype(float)
    s.name = key
    return s, meta


def _is_fresh(meta: dict, start: str | None, end: str | None) -> bool:
    """A cache entry is reusable only for the same request, and (if the request reaches
    today) only if it was retrieved today, so open-ended requests never silently go stale."""
    if meta.get("requested_start") != start or meta.get("requested_end") != end:
        return False
    today = datetime.now(timezone.utc).date()
    if end is None or pd.Timestamp(end).date() >= today:
        return str(meta.get("retrieved_at_utc", ""))[:10] == today.isoformat()
    return True


def download_prices(
    tickers: Iterable[str],
    start: str | None,
    end: str | None = None,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
    downloader: Downloader | None = None,
) -> PriceData:
    """Fetch adjusted closes per ticker, falling back to cache; never raises per ticker.

    A ticker that cannot be downloaded AND has no cache is listed under
    ``metadata['unavailable']``. If every ticker is unavailable the frame is empty.
    """
    dl = downloader or _yfinance_download
    series: dict[str, pd.Series] = {}
    per_ticker: dict[str, dict] = {}
    unavailable: list[dict] = []
    for t in tickers:
        cached = _read_cache(cache_dir, t)
        if cached and _is_fresh(cached[1], start, end) and not refresh:
            series[t], per_ticker[t] = cached[0], {**cached[1], "origin": "cache"}
            continue
        try:
            s = dl(t, start, end).dropna(how="all")
            if s.empty:
                raise RuntimeError("empty series")
            meta = {"source": "Yahoo Finance via yfinance (unofficial, auto_adjust=True)",
                    "retrieved_at_utc": _utc_now(), "requested_start": start, "requested_end": end,
                    "rows": int(len(s))}
            _write_cache(cache_dir, t, s, meta)
            series[t], per_ticker[t] = s, {**meta, "origin": "download"}
        except Exception as exc:  # noqa: BLE001 - network libraries raise many types
            log.warning("Download failed for %s: %s", t, exc)
            if cached:
                series[t] = cached[0]
                per_ticker[t] = {**cached[1], "origin": "stale_cache",
                                 "warning": f"download failed ({exc}); using cached data"}
            else:
                unavailable.append({"ticker": t, "reason": str(exc)})
    prices = pd.DataFrame(series).sort_index() if series else pd.DataFrame()
    meta = {"source": "Yahoo Finance via yfinance (unofficial)", "retrieved_at_utc": _utc_now(),
            "requested_start": start, "requested_end": end, "per_ticker": per_ticker,
            "unavailable": unavailable, "synthetic": False}
    return PriceData(prices, meta)


def load_cached_prices(tickers: Iterable[str], cache_dir: Path = CACHE_DIR) -> PriceData:
    """Read prices from local cache only (no network)."""
    series, per_ticker, missing = {}, {}, []
    for t in tickers:
        c = _read_cache(cache_dir, t)
        if c:
            series[t], per_ticker[t] = c[0], {**c[1], "origin": "cache"}
        else:
            missing.append({"ticker": t, "reason": "not in local cache"})
    prices = pd.DataFrame(series).sort_index() if series else pd.DataFrame()
    return PriceData(prices, {"source": "local cache", "per_ticker": per_ticker,
                              "unavailable": missing, "synthetic": False,
                              "retrieved_at_utc": _utc_now()})


def _fred_fetch(series_id: str, start: str | None, end: str | None) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    if start:
        url += f"&cosd={start}"
    if end:
        url += f"&coed={end}"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https host
        text = resp.read().decode("utf-8")
    return parse_fred_csv(text, series_id)


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse FRED CSV text. FRED encodes missing observations as '.', kept as NaN."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = df.columns[0]
    s = pd.Series(pd.to_numeric(df[series_id], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]), name=series_id)
    return s


def download_fred(
    series_ids: Iterable[str], start: str | None, end: str | None = None,
    cache_dir: Path = CACHE_DIR, fetcher: Callable[[str, str | None, str | None], pd.Series] | None = None,
) -> PriceData:
    """Download FRED yield series (levels in percent). Yields are context, not returns."""
    fetch = fetcher or _fred_fetch
    out, per, unavailable = {}, {}, []
    for sid in series_ids:
        key = f"FRED_{sid}"
        try:
            s = fetch(sid, start, end)
            meta = {"source": "FRED public CSV (no API key)", "retrieved_at_utc": _utc_now(),
                    "requested_start": start, "requested_end": end, "rows": int(len(s)),
                    "missing_observations": int(s.isna().sum())}
            _write_cache(cache_dir, key, s, meta)
            out[sid], per[sid] = s, {**meta, "origin": "download"}
        except Exception as exc:  # noqa: BLE001
            log.warning("FRED download failed for %s: %s", sid, exc)
            c = _read_cache(cache_dir, key)
            if c:
                out[sid], per[sid] = c[0].rename(sid), {**c[1], "origin": "stale_cache", "warning": str(exc)}
            else:
                unavailable.append({"series": sid, "reason": str(exc)})
    df = pd.DataFrame(out).sort_index() if out else pd.DataFrame()
    return PriceData(df, {"source": "FRED", "per_series": per, "unavailable": unavailable})


def load_cached_fred(series_ids: Iterable[str], cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    out = {}
    for sid in series_ids:
        c = _read_cache(cache_dir, f"FRED_{sid}")
        if c:
            out[sid] = c[0].rename(sid)
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


def prepare_prices(raw: pd.DataFrame, duplicates: str = "error") -> tuple[pd.DataFrame, int]:
    """Sort by date and handle duplicate dates explicitly.

    duplicates: 'error' (default; raise), 'keep_last' or 'keep_first'.
    Returns (prices, n_duplicate_rows_found). Duplicates are never averaged silently.
    """
    if duplicates not in {"error", "keep_last", "keep_first"}:
        raise ValueError("duplicates must be 'error', 'keep_last' or 'keep_first'")
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    n_dup = int(df.index.duplicated().sum())
    if n_dup and duplicates == "error":
        dates = df.index[df.index.duplicated()].strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(f"{n_dup} duplicate date rows found (e.g. {dates}). "
                         "Fix the file or pass duplicates='keep_last'/'keep_first'.")
    if n_dup:
        log.warning("Resolving %d duplicate dates with policy %s", n_dup, duplicates)
        df = df[~df.index.duplicated(keep="last" if duplicates == "keep_last" else "first")]
    return df.sort_index(), n_dup


def read_price_csv(path: str | Path | io.IOBase, duplicates: str = "error") -> PriceData:
    """Vendor-neutral CSV import.

    Wide format: first column ``date``, then one price column per ticker.
    Long format: columns ``date, ticker, price`` (or ``adj_close``/``close``).
    Blank cells stay missing (NaN). Non-numeric prices raise.
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "date" not in cols:
        raise ValueError(f"CSV needs a 'date' column; found {list(df.columns)}")
    date_col = cols["date"]
    if "ticker" in cols:
        price_col = next((cols[k] for k in ("adj_close", "price", "close") if k in cols), None)
        if price_col is None:
            raise ValueError("Long-format CSV needs a price/adj_close/close column")
        df[date_col] = pd.to_datetime(df[date_col])
        dup = int(df.duplicated([date_col, cols["ticker"]]).sum())
        if dup and duplicates == "error":
            raise ValueError(f"{dup} duplicate (date, ticker) rows in long-format CSV")
        keep = "last" if duplicates != "keep_first" else "first"
        df = df.drop_duplicates([date_col, cols["ticker"]], keep=keep)
        wide = df.pivot(index=date_col, columns=cols["ticker"], values=price_col)
        n_dup = dup
    else:
        wide = df.set_index(date_col)
        wide.index = pd.to_datetime(wide.index)
        wide, n_dup = prepare_prices(wide, duplicates)
    wide = wide.apply(pd.to_numeric, errors="raise").astype(float).sort_index()
    wide.columns = [str(c).strip().upper() for c in wide.columns]
    meta = {"source": f"local CSV import ({getattr(path, 'name', path)})", "retrieved_at_utc": _utc_now(),
            "duplicates_found": n_dup, "unavailable": [], "synthetic": False}
    return PriceData(wide.sort_index(), meta)
