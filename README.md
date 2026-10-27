# Multi-Asset Portfolio Risk & Model Validation Platform

A free, local research platform that measures the risk of a multi-asset ETF portfolio **and tests whether those risk measures can be trusted**.
*Historical research for education. Not investment advice.*

## The challenge
- A single VaR or volatility figure is easy to compute and easy to over-trust. A model that looks fine can be wrong in the tail, and a handful of breaches proves little.
- Free market data is messy (gaps, duplicates, late-listed ETFs, holidays). Forward-filling prices or zero-filling returns quietly corrupts every downstream number.
- Backtests are easily contaminated by lookahead bias, and institutional tools (Bloomberg, Aladdin) are not available to an individual analyst.

## The solution
| Problem | What the platform does |
|---|---|
| Untrusted data | Ingests Yahoo Finance (unofficial), FRED and local CSV with provenance and caching; reports duplicates, gaps, non-positive prices, large-move flags and short histories as JSON and Markdown. **Never fills prices or treats missing returns as zero.** |
| What drives risk | Volatility, drawdown, beta (OLS with Newey-West 95% interval), Sharpe/Sortino, and Euler risk contributions that reconcile exactly to portfolio volatility. |
| Reliability of risk measures | Historical and parametric VaR and ES at 95%/99%; rolling out-of-sample one-day forecasts that use only prior data; breach rate with exact confidence interval; Kupiec and Christoffersen tests. ES is described, not claimed as validated. |
| Behaviour in crises | Replay of 2008, March 2020 and 2022 using observed prices only (withheld, with a reason, if data is missing); separate labelled hypothetical shocks; documented duration approximation for rate moves. |
| Delivery | 7-page Streamlit app, formatted Excel workbook (LibreOffice-compatible), vendor-neutral CSV interface, DuckDB storage, one-page research brief generated only from real, verified data. |

## Impact: what it reveals
`python examples/model_risk_demo.py` simulates 5,000 days of fat-tailed returns (Student-t, 3 d.o.f., 1% daily volatility; synthetic and seeded) so the true process is known. 99% one-day VaR, 500-day rolling window:

| Model | Breaches (expected 45) | Breach rate (95% CI) | Kupiec p | Realised loss on breach days / forecast ES |
|---|---|---|---|---|
| Historical | 55 | 1.22% (0.92%-1.59%) | 0.148 | 0.95 |
| Parametric (normal) | 88 | 1.96% (1.57%-2.40%) | <0.0001 | 1.27 |

The normal model breaches almost twice as often as promised, and on breach days realised losses exceeded its forecast ES by 27% on average. The historical model is not rejected, but its rate is still above 1%: with 45 expected breaches the test has limited power, which is why the platform reports intervals and tests rather than a pass/fail flag.

Running the pipeline on real data (`download`, then `brief`) produces the portfolio findings in `outputs/research_brief.md`; the generator refuses synthetic data and data with a FAIL quality status.

## Engineering quality
- **66 tests**, all offline and deterministic (synthetic fixtures): returns, annualisation, drawdown, beta, risk-contribution reconciliation, VaR/ES on known samples, breach counting, forecast timing, a lookahead test that mutates future returns, weights, duplicates, missing dates, stress maths, Excel contents and formats, CLI, and the Streamlit UI headless.
- An adversarial review found and fixed: a crash in the Normalise button, stale cached data reused indefinitely, Excel cells showing a 1.5% loss as "0", and drawdowns that bridged data gaps. Each has a regression test (`tests/test_regressions.py`).
- Tested on Python 3.12 with pandas 3.0 and pandas 2.2; Excel output round-tripped through LibreOffice. CI runs Linux and Windows (`.github/workflows/tests.yml`).

## Quick start (Windows PowerShell)
```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt; pip install -e .
python -m pytest -m "not app"                       # offline suite (about a minute); omit -m to include UI tests
python -m riskplatform.cli download --start 2005-01-01
python -m riskplatform.cli export-excel --out outputs\portfolio_risk_report.xlsx
python -m riskplatform.cli brief --out outputs\research_brief.md
streamlit run app\streamlit_app.py
```
No internet? Use `export-excel --synthetic` (clearly labelled demo data) or `--csv data\imports\prices.csv`. Custom weights: `--weights "VTI=40,AGG=40,GLD=20" --rf 0.02`.

## Limitations
yfinance is unofficial and can break; ETF histories are short; the chosen ETFs are survivors; no transaction costs, taxes or liquidity effects; results depend on the sample window; normal VaR understates fat tails; Excel formulas and Python outputs can differ in rounding. Details: `docs/METHODOLOGY.md`. Comparing against authorised institutional output: `docs/INSTITUTIONAL_WORKFLOW.md`.

## Layout
`src/riskplatform/` ingestion, validation, analytics, risk_measures, backtesting, stress, pipeline, reporting, store, cli | `app/` Streamlit | `examples/` model-risk demo | `tests/` | `docs/`
