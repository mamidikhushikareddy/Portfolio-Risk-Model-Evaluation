"""Command line interface (PowerShell friendly).

  python -m riskplatform.cli download [--start 2005-01-01] [--end YYYY-MM-DD] [--refresh]
  python -m riskplatform.cli quality [--csv file.csv]
  python -m riskplatform.cli export-excel --out outputs/report.xlsx [--synthetic | --csv file.csv] [--weights VTI=30,...]
  python -m riskplatform.cli brief --out outputs/brief.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import config as C
from .analytics import validate_weights, weights_from_percent
from .ingestion import (PriceData, download_fred, download_prices, load_cached_fred, load_cached_prices, read_price_csv)
from .pipeline import AnalysisResult, run_analysis
from .reporting import export_excel, render_brief
from .store import save_prices
from .synthetic import make_synthetic_prices, make_synthetic_yields
from .validation import validate_prices, write_quality_reports

log = logging.getLogger("riskplatform")


def parse_weights(text: str | None) -> pd.Series:
    """Parse 'VTI=30,AGG=70' (percent) into validated fractional weights over the full universe."""
    pct = {t: 0.0 for t in C.TICKERS}
    if text:
        for part in text.split(","):
            k, v = part.split("=")
            if k.strip().upper() not in pct:
                raise ValueError(f"Unknown ticker '{k}'. Universe: {list(C.TICKERS)}")
            pct[k.strip().upper()] = float(v)
    else:
        pct = {k: v * 100 for k, v in C.DEFAULT_WEIGHTS.items()}
    return validate_weights(weights_from_percent(pct), tickers=list(C.TICKERS), tol=1e-6)


def _universe() -> list[str]:
    return list(C.TICKERS) + [C.BENCHMARK]


def build_result(args: argparse.Namespace) -> AnalysisResult:
    """Load data (synthetic, CSV or local cache), validate it, and run the full analysis."""
    yields = None
    if getattr(args, "synthetic", False):
        prices = make_synthetic_prices()
        data = PriceData(prices, {"source": "SYNTHETIC deterministic generator (not market data)", "synthetic": True,
                                  "retrieved_at_utc": "n/a", "unavailable": []})
        yields = make_synthetic_yields(prices.index)
    elif getattr(args, "csv", None):
        data = read_price_csv(args.csv, duplicates=args.duplicates)
    else:
        data = load_cached_prices(_universe())
        yields = load_cached_fred(C.FRED_SERIES)
    if data.prices.empty:
        raise SystemExit("No price data available. Run 'download', use --csv, or use --synthetic.")
    w = parse_weights(args.weights)
    missing = [t for t in [*w[w > 0].index, C.BENCHMARK] if t not in data.prices.columns]
    if missing:
        raise SystemExit(f"No data for: {missing}. Set their weights to 0 or import data for them.")
    w = w[w.index.isin(data.prices.columns)]  # only zero-weight tickers can be dropped here
    quality = validate_prices(data.prices, _universe(), args.start, None, data.metadata.get("unavailable"),
                              metadata=data.metadata, yields=yields)
    return run_analysis(data.prices, w, rf_annual=args.rf, yields=yields, quality=quality, metadata=data.metadata)


def cmd_download(a: argparse.Namespace) -> int:
    px = download_prices(_universe(), a.start, a.end, refresh=a.refresh)
    fred = download_fred(list(C.FRED_SERIES), a.start, a.end)
    print(f"Price tickers retrieved: {list(px.prices.columns)}")
    for u in px.metadata["unavailable"] + fred.metadata["unavailable"]:
        print(f"  UNAVAILABLE: {u}", file=sys.stderr)
    if px.prices.empty:
        print("No price data retrieved (network blocked?). Use 'export-excel --csv' with a local file.", file=sys.stderr)
        return 2
    q = validate_prices(px.prices, _universe(), a.start, a.end, px.metadata["unavailable"], metadata=px.metadata)
    jp, mp = write_quality_reports(q, C.OUTPUT_DIR)
    save_prices(px.prices, q)
    print(f"Quality status: {q['status']} -> {jp}, {mp}")
    return 0


def cmd_quality(a: argparse.Namespace) -> int:
    data = read_price_csv(a.csv, a.duplicates) if a.csv else load_cached_prices(_universe())
    if data.prices.empty:
        print("No data to validate.", file=sys.stderr)
        return 2
    q = validate_prices(data.prices, None, a.start, None, data.metadata.get("unavailable"), metadata=data.metadata)
    print(write_quality_reports(q, C.OUTPUT_DIR))
    print("Status:", q["status"])
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    res = build_result(a)
    out = export_excel(res, a.out)
    print(f"Workbook written: {out}" + ("  (SYNTHETIC DEMO DATA)" if res.metadata.get("synthetic") else ""))
    return 0


def cmd_brief(a: argparse.Namespace) -> int:
    res = build_result(a)
    text = render_brief(res, C.PROJECT_ROOT / "docs" / "RESEARCH_BRIEF_TEMPLATE.md")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"Brief written: {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="riskplatform", description="Educational historical research. Not financial advice.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download"); d.add_argument("--start", default=C.DEFAULT_START); d.add_argument("--end", default=C.DEFAULT_END)
    d.add_argument("--refresh", action="store_true"); d.set_defaults(fn=cmd_download)
    q = sub.add_parser("quality"); q.add_argument("--csv"); q.add_argument("--start", default=C.DEFAULT_START)
    q.add_argument("--duplicates", default="error"); q.set_defaults(fn=cmd_quality)
    for name, fn in (("export-excel", cmd_export), ("brief", cmd_brief)):
        e = sub.add_parser(name); e.add_argument("--out", required=True); e.add_argument("--synthetic", action="store_true")
        e.add_argument("--csv"); e.add_argument("--weights"); e.add_argument("--rf", type=float, default=0.0)
        e.add_argument("--duplicates", default="error"); e.add_argument("--start", default=C.DEFAULT_START)
        e.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
