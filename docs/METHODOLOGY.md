# Methodology, Assumptions and Limitations

Educational historical research. Not financial advice.

## 1. Data
* **Prices:** Yahoo Finance via `yfinance` (unofficial, undocumented scraping; may change, be rate-limited or return errors). Adjusted closes (`auto_adjust=True`) include dividends/splits, so returns approximate total returns. Treat as educational only.
* **Yields:** FRED public CSV (`DGS10`, `DGS2`, percent). Holidays appear as missing and stay missing. Yield levels are **context variables, not bond total returns**.
* **Local CSV import** (vendor-neutral) keeps the platform usable offline; see `INSTITUTIONAL_WORKFLOW.md`.
* Every download records source and retrieval timestamp (UTC) in `data/cache/*.meta.json` and the quality report. A cached request that reaches today is reused only if retrieved today; otherwise it is re-downloaded, and if that fails the stale cache is used with an explicit warning. The `--end` date is inclusive. Adjusted closes can be revised by the vendor after dividends, so a refresh can change history slightly.
* ETF inception dates in `config.py` are approximate and used only for warnings; verify with issuers.

## 2. Calendar and missing data
Rows exist only on dates where a series has a price. Weekdays absent from every series are treated as exchange holidays/closures and are reported, never created. Prices are **never forward-filled**. A return needs prices on two consecutive rows; a missing price makes both adjacent returns missing. Portfolio analysis uses complete-case dates for held assets (the count of excluded dates is shown), so a late-listed ETF shortens the common sample rather than being padded. Large moves (|daily return| > 10%) are flagged, never deleted.

## 3. Returns and annualisation
Simple daily returns `p_t/p_{t-1}-1`. Annualised mean = mean x 252 (arithmetic, not CAGR; arithmetic means overstate compound growth). Volatility = sample std (ddof=1) x sqrt(252). Risk-free rate: user-set annual rate / 252 per day (default 0%). Sharpe = annualised mean excess / annualised excess volatility. Sortino uses downside deviation below the risk-free daily rate over all observations. Beta is the covariance estimate; the reported 95% interval comes from OLS with Newey-West (5-lag) standard errors (statsmodels). Asset-level maximum drawdown is computed from observed prices, so data gaps are not bridged. Portfolio = constant weights rebalanced daily, with no costs.

## 4. Risk contribution convention
Euler decomposition of annualised volatility: marginal `(Σw)_i/σ_p`, component `w_i x marginal_i`, percentage = component/σ_p. Components sum exactly to σ_p. Covariance is estimated on the common complete-case sample. This is volatility attribution, not tail-risk (ES) attribution.

## 5. VaR and ES
Positive loss thresholds, 1-day horizon, 95% and 99%. **Historical:** empirical quantile of losses (linear interpolation); ES = mean of losses ≥ VaR (inclusive tail average). **Parametric:** normal with sample mean and std; ES = σφ(z)/(1-c) - μ. Normality understates tail risk for real returns. At 99% with a 500-day window only about 5 observations lie in the tail, so estimates are noisy.

## 6. Backtesting and lookahead protection
For each realised date `t`, the forecast uses only returns before `t` (the window ends at the previous trading day). Breach = realised loss > forecast VaR. Reported: breach count and rate, expected rate, Clopper-Pearson 95% interval, Kupiec POF (unconditional coverage) and Christoffersen independence tests, and a descriptive ES comparison. A unit test mutates future returns and confirms earlier forecasts are unchanged.
**Why few breaches are not proof:** breach counts are small at 99%, so tests have low power; overly conservative models also "pass" a one-sided intuition but fail two-sided coverage; breaches may cluster; windows overlap, so forecasts are serially dependent; and VaR ignores loss size beyond the threshold. Passing a VaR test says nothing about whether ES is correct; ES backtests need different tests (e.g. Acerbi-Szekely) not implemented here.

## 7. Stress testing
* **Historical replay** (observed only): from the last close before the start date to the end-date close, for each weighted asset and for the daily-rebalanced portfolio (a buy-and-hold figure is also shown). If any held asset lacks data across the window, the portfolio result is withheld and the reason shown. For pre-inception periods, supply a documented index proxy as CSV and label it; the platform never fabricates ETF history.
* **Hypothetical shocks:** explicit user-visible price shocks, labelled hypothetical; every held asset needs a shock (no silent zeros).
* **Rate sensitivity:** `dP/P ≈ -D·Δy + ½·C·Δy²`, parallel shifts, bond ETFs only. Durations/convexities in `config.py` are rough assumptions; real funds' duration drifts and curves twist. Yield changes are context, not returns.
* Recovery = trading days from the in-window trough until the constant-weight path regains its pre-event level (or "not recovered in available data").

## 8. Known limitations
Survivorship bias (these ETFs exist and were chosen with hindsight). Lookahead bias in the *choice* of sample and events. Short histories (EEM, GLD, AGG start in the 2000s). Single-regime dependence: results change with the window. No transaction costs, taxes, bid-ask spreads, fees, tracking error or liquidity stress (ETFs traded at discounts in March 2020). Adjusted-close quirks from the data vendor. Historical estimates are backward-looking, not forecasts. Rounding: Excel formulas and Python outputs can differ in the last displayed digits.
