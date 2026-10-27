"""Streamlit UI. Educational historical research - not financial advice."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from riskplatform import analytics as an
from riskplatform import config as C
from riskplatform import stress as stress_mod
from riskplatform.ingestion import (PriceData, download_fred, download_prices, load_cached_fred,
                                    load_cached_prices, read_price_csv)
from riskplatform.pipeline import run_analysis
from riskplatform.reporting import METHODOLOGY_LINES, export_excel, render_brief
from riskplatform.synthetic import make_synthetic_prices, make_synthetic_yields
from riskplatform.validation import quality_report_markdown, validate_prices

st.set_page_config(page_title="Portfolio Risk & Model Validation", layout="wide")
UNIVERSE = list(C.TICKERS) + [C.BENCHMARK]
DISCLAIMER = "Educational historical research. Not investment advice. Past results do not predict future results."
PAGES = ["1. Portfolio overview", "2. Allocation & risk contribution", "3. Historical risk & drawdown",
         "4. VaR, ES & backtesting", "5. Stress scenarios", "6. Data quality & methodology",
         "7. Research summary & exports"]


@st.cache_data(show_spinner=False)
def _load(source: str, csv_bytes: bytes | None, dup_policy: str):
    if source == "Synthetic demo":
        p = make_synthetic_prices()
        return PriceData(p, {"source": "SYNTHETIC deterministic generator (NOT market data)", "synthetic": True,
                             "retrieved_at_utc": "n/a", "unavailable": []}), make_synthetic_yields(p.index)
    if source == "Upload CSV":
        import io
        return read_price_csv(io.BytesIO(csv_bytes), duplicates=dup_policy), pd.DataFrame()
    return load_cached_prices(UNIVERSE), load_cached_fred(C.FRED_SERIES)


@st.cache_data(show_spinner="Computing risk analytics and rolling backtests...")
def _analyse(prices: pd.DataFrame, w_items: tuple, rf: float, window: int, yields: pd.DataFrame,
             quality: dict, metadata: dict):
    return run_analysis(prices, pd.Series(dict(w_items)), rf_annual=rf, window=window,
                        yields=yields, quality=quality, metadata=metadata)


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2%}"


# ---------------- sidebar: data + inputs ----------------
st.sidebar.title("Risk research platform")
st.sidebar.caption(DISCLAIMER)
source = st.sidebar.radio("Data source", ["Local cache (downloaded)", "Upload CSV", "Synthetic demo"])
upload, dup = None, "error"
if source == "Upload CSV":
    upload = st.sidebar.file_uploader("Wide or long price CSV", type="csv")
    dup = st.sidebar.selectbox("If duplicate dates exist", ["error", "keep_last", "keep_first"])
if source == "Local cache (downloaded)":
    d_start = st.sidebar.text_input("Download start", C.DEFAULT_START)
    if st.sidebar.button("Download / refresh (Yahoo + FRED)"):
        with st.spinner("Downloading..."):
            px_new = download_prices(UNIVERSE, d_start, None, refresh=True)
            download_fred(list(C.FRED_SERIES), d_start, None)
        if px_new.prices.empty:
            st.sidebar.error("Download failed for all tickers (offline or blocked). Use CSV import or the synthetic demo.")
        else:
            st.sidebar.success(f"Retrieved {list(px_new.prices.columns)}")
            if px_new.metadata["unavailable"]:
                st.sidebar.warning(f"Unavailable: {[u['ticker'] for u in px_new.metadata['unavailable']]}")
        st.cache_data.clear()

page = st.sidebar.radio("Page", PAGES)

try:
    data, yields = _load(source, upload.getvalue() if upload else None, dup)
except Exception as exc:  # noqa: BLE001 - show any parse problem to the user
    st.error(f"Could not load data: {exc}")
    st.stop()
if data.prices.empty:
    st.title("Multi-asset portfolio risk & model validation")
    st.warning("No price data available yet. Use the sidebar to download data, upload a CSV, or open the synthetic demo.")
    st.stop()

prices_all = data.prices
lo, hi = prices_all.index.min().date(), prices_all.index.max().date()
dr = st.sidebar.date_input("Analysis dates", (lo, hi), min_value=lo, max_value=hi)
if not (isinstance(dr, tuple) and len(dr) == 2):
    st.sidebar.info("Pick both a start and an end date.")
    st.stop()
prices = prices_all.loc[str(dr[0]):str(dr[1])]
rf = st.sidebar.number_input("Risk-free rate (annual, %)", 0.0, 20.0, C.DEFAULT_RF_ANNUAL * 100, 0.25) / 100
window = int(st.sidebar.number_input("Backtest window (trading days)", 100, 1500, C.DEFAULT_BACKTEST_WINDOW, 50))

st.sidebar.subheader("Allocation (% of portfolio)")
available = [t for t in C.TICKERS if t in prices.columns]
WEIGHT_TOL = 1e-4  # percentage points; equals the 1e-6 fractional tolerance in validate_weights


def _reset_weights() -> None:
    for t_ in C.TICKERS:
        st.session_state[f"w_{t_}"] = C.DEFAULT_WEIGHTS[t_] * 100


def _normalise_weights() -> None:
    tot = sum(st.session_state[f"w_{t_}"] for t_ in available)
    for t_ in available:
        st.session_state[f"w_{t_}"] = st.session_state[f"w_{t_}"] / tot * 100


for t in C.TICKERS:
    if f"w_{t}" not in st.session_state:
        st.session_state[f"w_{t}"] = C.DEFAULT_WEIGHTS[t] * 100
st.sidebar.button("Reset to illustrative default", on_click=_reset_weights)
for t in C.TICKERS:
    st.sidebar.number_input(f"{t} - {C.TICKERS[t]}", 0.0, 100.0, step=1.0, format="%.2f", key=f"w_{t}",
                            disabled=t not in available)
total = sum(st.session_state[f"w_{t}"] for t in available)
if abs(total - 100) < WEIGHT_TOL:
    st.sidebar.success(f"Total: {total:.2f}%")
else:
    st.sidebar.error(f"Total: {total:.2f}% (must be 100%)")
    if total > 0:
        st.sidebar.button("Normalise to 100%", on_click=_normalise_weights)

quality = validate_prices(prices, UNIVERSE, str(dr[0]), str(dr[1]), data.metadata.get("unavailable"),
                          metadata=data.metadata, yields=yields)

st.title("Multi-asset portfolio risk & model validation")
st.caption(f"{DISCLAIMER}  Data: {data.metadata.get('source')} | retrieved {data.metadata.get('retrieved_at_utc')}")
if data.metadata.get("synthetic"):
    st.error("SYNTHETIC DEMO DATA: simulated numbers, not market history. Do not read results as real.")
elif "yfinance" in str(data.metadata.get("source")):
    st.info("Yahoo Finance via yfinance is an unofficial educational source; verify before relying on it.")
if quality["status"] != "PASS":
    st.warning(f"Data quality status: {quality['status']} - see page 6 for details.")

if page == PAGES[5]:
    st.header("Data quality & methodology")
    st.json({k: quality[k] for k in ("status", "requested", "duplicate_date_rows", "calendar")}, expanded=False)
    st.dataframe(pd.DataFrame(quality["issues"]) if quality["issues"] else pd.DataFrame({"issues": ["none"]}),
                 width="stretch")
    st.dataframe(pd.DataFrame(quality["per_ticker"]).T.drop(columns=["large_gaps", "large_return_flags"], errors="ignore"),
                 width="stretch")
    st.subheader("Methodology and limitations")
    for line in METHODOLOGY_LINES:
        st.markdown(f"- {line}")
    st.markdown("See `docs/METHODOLOGY.md` for survivorship bias, lookahead bias, transaction costs, liquidity and short-history limits.")
    st.stop()

if not available:
    st.error("None of the universe tickers are present in the data.")
    st.stop()
if abs(total - 100) >= WEIGHT_TOL:
    st.error("Weights must sum to exactly 100% before metrics are computed. Adjust the sidebar or press Normalise.")
    st.stop()
w = pd.Series({t: st.session_state[f"w_{t}"] / 100 for t in available})
try:
    res = _analyse(prices, tuple(w.items()), rf, window, yields if yields is not None else pd.DataFrame(),
                   quality, data.metadata)
except (ValueError, KeyError) as exc:
    st.error(f"Cannot compute analytics: {exc}")
    st.stop()

i, ps = res.sample_info, res.portfolio_stats
st.caption(f"Analysis sample: {i['start']:%Y-%m-%d} to {i['end']:%Y-%m-%d}  |  {i['rows_after']:,} trading days "
           f"({i['rows_dropped_missing']:,} dates excluded for missing returns - never filled)")
growth = an.cumulative_growth(res.portfolio_returns, 10_000)
bench_g = an.cumulative_growth(res.returns[C.BENCHMARK].reindex(res.portfolio_returns.index), 10_000)

if page == PAGES[0]:
    st.header("Portfolio overview")
    c = st.columns(5)
    c[0].metric("Ann. return (arithmetic)", pct(ps["ann_return_arith"]))
    c[1].metric("Ann. volatility", pct(ps["ann_volatility"]))
    c[2].metric("Max drawdown", pct(ps["max_drawdown"]))
    c[3].metric("Sharpe", f"{ps['sharpe']:.2f}")
    c[4].metric(f"Beta vs {C.BENCHMARK}", f"{ps.get('beta_vs_benchmark', float('nan')):.2f}",
                help=f"95% CI (Newey-West): {ps['beta_ci95_low']:.2f} to {ps['beta_ci95_high']:.2f}")
    fig = go.Figure([go.Scatter(x=growth.index, y=growth, name="Portfolio"),
                     go.Scatter(x=bench_g.index, y=bench_g, name=C.BENCHMARK)])
    fig.update_layout(title="Growth of a hypothetical $10,000 (daily-rebalanced, no costs, log scale)",
                      yaxis_title="USD", yaxis_type="log", height=420, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Metric definitions and assumptions"):
        st.markdown(f"- Annualised return = mean daily simple return x {C.TRADING_DAYS} (arithmetic, not CAGR).\n"
                    f"- Volatility = daily sample std x sqrt({C.TRADING_DAYS}).\n"
                    f"- Sharpe/Sortino use risk-free {rf:.2%}/yr converted to a daily rate (/{C.TRADING_DAYS}).\n"
                    "- Max drawdown = largest peak-to-trough fall of cumulative value.\n"
                    f"- Beta = cov(portfolio, {C.BENCHMARK}) / var({C.BENCHMARK}) on daily returns.")

elif page == PAGES[1]:
    st.header("Asset allocation & risk contribution")
    rc = res.risk_contrib
    fig = go.Figure([go.Bar(x=rc.index, y=rc["weight"] * 100, name="Capital weight (%)"),
                     go.Bar(x=rc.index, y=rc["pct_contribution"] * 100, name="Share of portfolio volatility (%)")])
    fig.update_layout(barmode="group", height=400, yaxis_title="percent", legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(rc.style.format({"weight": "{:.1%}", "marginal_contribution": "{:.4f}",
                                  "component_contribution": "{:.2%}", "pct_contribution": "{:.1%}"}),
                 width="stretch")
    st.caption(f"Convention: Euler decomposition of annualised volatility ({ps['ann_volatility']:.2%}). Component contributions sum "
               "to portfolio volatility. Weight-based and risk-based shares differ because of volatility and correlation.")
    heat = go.Figure(go.Heatmap(z=res.corr.values, x=res.corr.columns, y=res.corr.index, zmin=-1, zmax=1,
                                colorscale="RdBu_r", text=np.round(res.corr.values, 2), texttemplate="%{text}"))
    heat.update_layout(title="Correlation of daily returns (common sample)", height=450)
    st.plotly_chart(heat, width="stretch")

elif page == PAGES[2]:
    st.header("Historical risk & drawdown")
    dd = an.drawdown_series(res.portfolio_returns)
    f1 = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", name="Drawdown"))
    f1.update_layout(title="Portfolio drawdown from running peak (%)", height=350, yaxis_title="percent")
    st.plotly_chart(f1, width="stretch")
    d = res.drawdown
    st.write(f"**Max drawdown {pct(d['max_drawdown'])}** - peak {d['peak_date'] or 'start of sample'}, trough {d['trough_date']:%Y-%m-%d}, "
             f"recovery {d['recovery_date'] if d['recovery_date'] is not None else 'not recovered in sample'}; "
             f"longest underwater spell {d['longest_underwater_days']} trading days.")
    rv = res.portfolio_returns.rolling(60).std() * np.sqrt(C.TRADING_DAYS) * 100
    f2 = go.Figure(go.Scatter(x=rv.index, y=rv, name="60-day rolling volatility"))
    f2.update_layout(title="60-day rolling annualised volatility (%)", height=320)
    st.plotly_chart(f2, width="stretch")
    show = res.asset_stats.copy()
    show.loc["PORTFOLIO"] = pd.Series(res.portfolio_stats).reindex(show.columns)
    st.subheader("Asset-level statistics (each asset on its own available history)")
    st.dataframe(show, width="stretch")

elif page == PAGES[3]:
    st.header("VaR, expected shortfall & backtesting")
    st.markdown("VaR and ES are **positive loss thresholds** (a 2% VaR means a 2% loss), 1-day horizon. Full-sample figures are in-sample descriptions; "
                "the backtest below is out-of-sample.")
    st.dataframe(res.var_table.style.format({"confidence": "{:.0%}", "var": "{:.2%}", "es": "{:.2%}"}), width="stretch")
    if res.backtest_summary.empty:
        st.warning(f"Not enough observations for a {window}-day rolling backtest. Reduce the window or widen the date range.")
    else:
        bts = res.backtest_summary
        st.dataframe(bts.style.format({"confidence": "{:.0%}", "breach_rate": "{:.2%}", "expected_breach_rate": "{:.2%}",
                                       "breach_rate_ci95_low": "{:.2%}", "breach_rate_ci95_high": "{:.2%}",
                                       "kupiec_p_value": "{:.3f}", "christoffersen_ind_p_value": "{:.3f}"}), width="stretch")
        m = st.selectbox("Model", ["historical", "parametric"])
        cl = st.selectbox("Confidence", list(C.CONFIDENCE_LEVELS), format_func=lambda x: f"{x:.0%}", index=1)
        f = res.forecasts[(m, cl)]
        fig = go.Figure([go.Scatter(x=f.index, y=f["realised_loss"] * 100, name="Realised loss (%)", mode="lines", line=dict(width=1)),
                         go.Scatter(x=f.index, y=f["var_forecast"] * 100, name="Forecast VaR (%)"),
                         go.Scatter(x=f.index[f["breach"]], y=f.loc[f["breach"], "realised_loss"] * 100, mode="markers",
                                    name="Breach", marker=dict(color="red", size=6))])
        fig.update_layout(height=420, title=f"{m} {cl:.0%} one-day-ahead VaR (forecast uses prior {window} days only)")
        st.plotly_chart(fig, width="stretch")
    st.info("A small breach count does not prove a model is reliable: at 99% the expected count is small, so the Kupiec test has low power; "
            "breaches can cluster (Christoffersen test); and VaR says nothing about loss size beyond the threshold. "
            "Passing a VaR backtest does NOT validate Expected Shortfall - the ES comparison shown is descriptive only.")

elif page == PAGES[4]:
    st.header("Stress scenarios")
    tab1, tab2, tab3 = st.tabs(["Historical event replay (observed)", "Hypothetical shocks", "Rate sensitivity (approximation)"])
    with tab1:
        summ = pd.DataFrame([{"event": r["event"], "start": r["start"].date(), "end": r["end"].date(),
                              "portfolio return (daily rebal.)": r.get("portfolio_return_daily_rebalanced"),
                              "max drawdown in window": r.get("max_drawdown_in_window"),
                              "recovery date": r.get("recovery_date"),
                              "status": "available" if r["portfolio_available"] else f"UNAVAILABLE - {r['reason']}"} for r in res.stress_replays])
        st.dataframe(summ.style.format({"portfolio return (daily rebal.)": "{:.2%}", "max drawdown in window": "{:.2%}"}), width="stretch")
        ev = st.selectbox("Inspect event", [r["event"] for r in res.stress_replays])
        r = next(x for x in res.stress_replays if x["event"] == ev)
        st.caption(f"{r['description']}  Return convention: last close before {r['start']:%Y-%m-%d} to close on {r['end']:%Y-%m-%d}. "
                   "Only observed prices are used; unavailable assets are never invented.")
        if len(r["asset_table"]):
            st.dataframe(r["asset_table"].style.format({"weight": "{:.1%}", "event_return": "{:.2%}", "contribution_buy_hold": "{:.2%}"}),
                         width="stretch")
        if r["portfolio_available"]:
            fig = go.Figure(go.Scatter(x=r["path"].index, y=r["path"], name="Portfolio value (start = 1.0)"))
            fig.update_layout(height=340, title="Portfolio path through the event window")
            st.plotly_chart(fig, width="stretch")
            st.write(f"Worst day {pct(r['worst_day'])} on {r['worst_day_date']:%Y-%m-%d}; trough "
                     f"{r['trough_date'] if r['trough_date'] is not None else 'n/a'}; "
                     f"recovery: {r.get('recovery_note', 'n/a')}"
                     + (f" ({r['days_trough_to_recovery']} trading days after trough)" if r.get("days_trough_to_recovery") else "") + ".")
        if ev in res.yield_context:
            st.markdown("**Treasury yield context (levels in %, FRED; context only, not bond returns)**")
            st.dataframe(res.yield_context[ev], width="stretch")
    with tab2:
        st.warning("HYPOTHETICAL shocks are not history and not forecasts.")
        for h in res.stress_hypothetical:
            st.markdown(f"**{h['scenario']}** - portfolio impact **{pct(h['portfolio_return'])}**")
            st.dataframe(h["table"].style.format({"weight": "{:.1%}", "shock": "{:.1%}", "contribution": "{:.2%}"}), width="stretch")
        if not res.stress_hypothetical:
            st.info("No predefined hypothetical scenario covers every asset you hold.")
    with tab3:
        bp = st.slider("Parallel yield change (basis points)", -300, 300, 100, 25)
        d = stress_mod.duration_approximation(res.weights, bp, C.APPROX_DURATION, C.APPROX_CONVEXITY)
        st.dataframe(d["table"].style.format({"weight": "{:.1%}", "approx_price_return": "{:.2%}", "contribution": "{:.2%}"}), width="stretch")
        st.write(f"Approximate bond-sleeve contribution to portfolio: **{pct(d['bond_contribution_to_portfolio'])}**")
        st.caption("Uses dP/P = -D*dy + 0.5*C*dy^2 with ASSUMED duration/convexity (edit in config.py). " + d["caveat"] +
                   " Treasury yield changes are contextual variables, not substitutes for bond total returns.")

elif page == PAGES[6]:
    st.header("Research summary & exports")
    top = res.risk_contrib.sort_values("pct_contribution", ascending=False).iloc[0]
    st.markdown(f"**Sample** {i['start']:%Y-%m-%d} to {i['end']:%Y-%m-%d} ({i['rows_after']:,} days). "
                f"Annualised volatility **{pct(ps['ann_volatility'])}**, max drawdown **{pct(ps['max_drawdown'])}**. "
                f"**{top.name}** holds {top['weight']:.1%} of capital but contributes {top['pct_contribution']:.1%} of volatility.")
    if not res.backtest_summary.empty:
        st.markdown("Backtest results (see page 4 for caveats):")
        for r in res.backtest_summary.itertuples():
            st.markdown(f"- {r.model} {r.confidence:.0%}: {r.breaches} breaches / {r.n_forecasts} forecasts "
                        f"({r.breach_rate:.2%} vs {r.expected_breach_rate:.2%} expected); Kupiec p = {r.kupiec_p_value:.3f}")
    with tempfile.TemporaryDirectory() as tmp:
        x = export_excel(res, Path(tmp) / "risk_report.xlsx")
        st.download_button("Download Excel workbook (.xlsx)", x.read_bytes(), "portfolio_risk_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download data-quality JSON", json.dumps(quality, indent=2, default=str), "data_quality_report.json")
    st.download_button("Download data-quality Markdown", quality_report_markdown(quality), "data_quality_report.md")
    try:
        brief = render_brief(res, C.PROJECT_ROOT / "docs" / "RESEARCH_BRIEF_TEMPLATE.md")
        st.download_button("Download populated research brief (.md)", brief, "research_brief.md")
    except ValueError as exc:
        st.info(f"Research brief not generated: {exc}")
