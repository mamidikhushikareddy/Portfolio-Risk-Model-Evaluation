"""Data-quality diagnostics. Flags problems; never deletes or repairs observations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from .config import INCEPTION_DATES, LARGE_RETURN_THRESHOLD, MAX_GAP_CALENDAR_DAYS, MIN_HISTORY_YEARS


def _issue(severity: str, ticker: str | None, code: str, message: str) -> dict:
    return {"severity": severity, "ticker": ticker, "code": code, "message": message}


def validate_prices(
    prices: pd.DataFrame,
    requested_tickers: list[str] | None = None,
    requested_start: str | None = None,
    requested_end: str | None = None,
    unavailable: list[dict] | None = None,
    inception_dates: Mapping[str, str] = INCEPTION_DATES,
    large_return_threshold: float = LARGE_RETURN_THRESHOLD,
    min_years: float = MIN_HISTORY_YEARS,
    max_gap_days: int = MAX_GAP_CALENDAR_DAYS,
    metadata: dict | None = None,
    yields: pd.DataFrame | None = None,
) -> dict:
    """Return a machine-readable quality report (JSON-serialisable dict)."""
    issues: list[dict] = []
    requested = requested_tickers or list(prices.columns)
    unavailable = unavailable or []
    for u in unavailable:
        issues.append(_issue("error", u.get("ticker") or u.get("series"), "UNAVAILABLE",
                             f"No data: {u.get('reason', 'unknown reason')}"))
    for t in requested:
        if t not in prices.columns and not any(u.get("ticker") == t for u in unavailable):
            issues.append(_issue("error", t, "MISSING_COLUMN", "Requested ticker absent from price table"))

    n_dup = int(prices.index.duplicated().sum()) if len(prices) else 0
    if n_dup:
        dates = prices.index[prices.index.duplicated()].strftime("%Y-%m-%d").tolist()[:10]
        issues.append(_issue("error", None, "DUPLICATE_DATES", f"{n_dup} duplicate date rows, e.g. {dates}"))
    if len(prices) and not prices.index.is_monotonic_increasing:
        issues.append(_issue("warning", None, "UNSORTED_INDEX", "Dates are not sorted ascending"))

    px = prices[~prices.index.duplicated(keep="first")].sort_index() if len(prices) else prices
    per_ticker: dict[str, dict] = {}
    for t in px.columns:
        s = px[t]
        valid = s.dropna()
        info: dict = {"n_obs": int(len(valid))}
        if valid.empty:
            issues.append(_issue("error", t, "NO_VALID_PRICES", "Column has no valid prices"))
            per_ticker[t] = info
            continue
        first, last = valid.index[0], valid.index[-1]
        span = px.loc[first:last, t]
        n_missing = int(span.isna().sum())
        n_nonpos = int((valid <= 0).sum())
        years = (last - first).days / 365.25
        info.update(first_date=first.strftime("%Y-%m-%d"), last_date=last.strftime("%Y-%m-%d"),
                    n_missing_inside_span=n_missing,
                    missing_pct_inside_span=round(100 * n_missing / max(len(span), 1), 3),
                    n_nonpositive=n_nonpos, years_of_history=round(years, 2))
        if n_missing:
            issues.append(_issue("warning", t, "MISSING_VALUES",
                                 f"{n_missing} missing prices between first and last observation (not filled)"))
        if n_nonpos:
            issues.append(_issue("error", t, "NON_POSITIVE_PRICE", f"{n_nonpos} prices <= 0"))
        gaps = valid.index.to_series().diff().dt.days.dropna()
        big = gaps[gaps > max_gap_days]
        info["large_gaps"] = [{"before": d.strftime("%Y-%m-%d"), "calendar_days": int(g)} for d, g in big.items()][:20]
        if len(big):
            issues.append(_issue("warning", t, "LARGE_GAP", f"{len(big)} gaps longer than {max_gap_days} calendar days"))
        pos = valid[valid > 0]
        rets = pos / pos.shift(1) - 1
        flagged = rets[rets.abs() > large_return_threshold].dropna()
        info["large_return_flags"] = [{"date": d.strftime("%Y-%m-%d"), "return": round(float(r), 5)}
                                      for d, r in flagged.items()][:50]
        info["n_large_return_flags"] = int(len(flagged))
        if len(flagged):
            issues.append(_issue("info", t, "LARGE_RETURN",
                                 f"{len(flagged)} daily moves beyond +/-{large_return_threshold:.0%}; "
                                 "flagged only, kept in the data (may be genuine market moves)"))
        if years < min_years:
            issues.append(_issue("warning", t, "SHORT_HISTORY", f"Only {years:.1f} years of data (< {min_years})"))
        if requested_start and first > pd.Timestamp(requested_start) + pd.Timedelta(days=10):
            inc = inception_dates.get(t)
            why = f"ETF inception approx {inc}" if inc else "inception date unknown"
            issues.append(_issue("warning", t, "LATE_START",
                                 f"First price {info['first_date']} is after requested start {requested_start} ({why})"))
        if requested_start and inception_dates.get(t) and pd.Timestamp(requested_start) < pd.Timestamp(inception_dates[t]):
            issues.append(_issue("info", t, "BEFORE_INCEPTION",
                                 f"Requested start precedes approximate inception {inception_dates[t]}"))
        if requested_end and last < pd.Timestamp(requested_end) - pd.Timedelta(days=7):
            issues.append(_issue("warning", t, "INCOMPLETE_END",
                                 f"Last price {info['last_date']} is >7 days before requested end {requested_end}"))
        per_ticker[t] = info

    # Calendar diagnostics. Dates absent from EVERY series are treated as non-trading
    # days (holidays/closures) and are never filled. Weekday-only list is informational.
    cal: dict = {"n_dates": int(len(px))}
    if len(px):
        weekdays = pd.bdate_range(px.index[0], px.index[-1])
        absent = weekdays.difference(px.index)
        cal["weekdays_absent_from_all_series"] = int(len(absent))
        cal["absent_weekday_examples"] = [d.strftime("%Y-%m-%d") for d in absent[:15]]
        starts = [v["first_date"] for v in per_ticker.values() if "first_date" in v]
        ends = [v["last_date"] for v in per_ticker.values() if "last_date" in v]
        if starts:
            cs, ce = max(starts), min(ends)
            common = px.loc[cs:ce].dropna(how="any")
            cal.update(common_start=cs, common_end=ce, common_complete_rows=int(len(common)),
                       rows_in_common_span=int(len(px.loc[cs:ce])))
    yq = validate_yields(yields, requested_start, requested_end) if yields is not None else None
    if yq:
        issues += yq["issues"]
    if any(i["severity"] == "error" for i in issues):
        status = "FAIL"
    elif any(i["severity"] == "warning" for i in issues):
        status = "WARN"
    else:
        status = "PASS"
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": status, "requested": {"tickers": requested, "start": requested_start, "end": requested_end},
            "source_metadata": metadata or {}, "duplicate_date_rows": n_dup, "calendar": cal,
            "per_ticker": per_ticker, "yields": (yq or {}).get("series", {}), "issues": issues,
            "notes": ["Prices are never forward-filled.", "Large-return flags do not remove observations.",
                      "yfinance is an unofficial educational data source."]}


def validate_yields(yields: pd.DataFrame | None, requested_start: str | None = None,
                    requested_end: str | None = None) -> dict:
    """Diagnostics for FRED yield levels (percent). Missing observations (holidays) are
    counted, never filled. Flags values outside a plausible -1%..25% range."""
    out: dict = {"series": {}, "issues": []}
    if yields is None or yields.empty:
        out["issues"].append(_issue("warning", None, "NO_YIELDS", "No Treasury yield data loaded; yield context unavailable"))
        return out
    for c in yields.columns:
        v = yields[c].dropna()
        info = {"n_obs": int(len(v)), "n_missing": int(yields[c].isna().sum())}
        if len(v):
            info.update(first_date=v.index[0].strftime("%Y-%m-%d"), last_date=v.index[-1].strftime("%Y-%m-%d"),
                        min_pct=float(v.min()), max_pct=float(v.max()))
            if (v < -1).any() or (v > 25).any():
                out["issues"].append(_issue("warning", c, "IMPLAUSIBLE_YIELD", "Values outside -1%..25%"))
            if requested_end and v.index[-1] < pd.Timestamp(requested_end) - pd.Timedelta(days=10):
                out["issues"].append(_issue("warning", c, "INCOMPLETE_END", f"Last observation {info['last_date']}"))
        else:
            out["issues"].append(_issue("error", c, "NO_VALID_PRICES", "Series has no valid observations"))
        out["series"][c] = info
    return out


def quality_report_markdown(report: dict) -> str:
    """Render the JSON quality report as readable Markdown."""
    lines = ["# Data Quality Report", "", f"Generated (UTC): {report['generated_at_utc']}",
             f"Overall status: **{report['status']}**", ""]
    src = report.get("source_metadata", {})
    if src:
        lines += [f"Source: {src.get('source', 'n/a')}  ", f"Retrieved (UTC): {src.get('retrieved_at_utc', 'n/a')}", ""]
        if src.get("synthetic"):
            lines += ["> **SYNTHETIC DEMO DATA - not real market data.**", ""]
    lines += ["## Issues", ""]
    if not report["issues"]:
        lines.append("No issues detected.")
    for i in report["issues"]:
        lines.append(f"- [{i['severity'].upper()}] {i['ticker'] or 'ALL'} - {i['code']}: {i['message']}")
    lines += ["", "## Per-ticker summary", "",
              "| Ticker | First | Last | Obs | Missing | Non-positive | Years | Large-return flags |",
              "|---|---|---|---|---|---|---|---|"]
    for t, v in report["per_ticker"].items():
        lines.append(f"| {t} | {v.get('first_date', '-')} | {v.get('last_date', '-')} | {v['n_obs']} | "
                     f"{v.get('n_missing_inside_span', '-')} | {v.get('n_nonpositive', '-')} | "
                     f"{v.get('years_of_history', '-')} | {v.get('n_large_return_flags', '-')} |")
    if report.get("yields"):
        lines += ["", "## Treasury yields (FRED, context only)", ""] + [
            f"- {k}: {v['n_obs']} obs, {v['n_missing']} missing (holidays, not filled), {v.get('first_date', '-')} to {v.get('last_date', '-')}"
            for k, v in report["yields"].items()]
    cal = report.get("calendar", {})
    lines += ["", "## Calendar", ""] + [f"- {k}: {v}" for k, v in cal.items()]
    lines += ["", "## Notes", ""] + [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"


def write_quality_reports(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / "data_quality_report.json", out_dir / "data_quality_report.md"
    jp.write_text(json.dumps(report, indent=2, default=str))
    mp.write_text(quality_report_markdown(report), encoding="utf-8")
    return jp, mp
