# Institutional-style Risk Review Workflow (vendor-neutral)

This project does **not** connect to Bloomberg, Aladdin or any licensed system.

## CSV interface
* **Prices (wide):** `date,VTI,EFA,AGG,...` one adjusted-close column per ticker; blank cell = missing.
* **Prices (long):** `date,ticker,adj_close`.
* Duplicate dates raise an error unless you choose `--duplicates keep_last|keep_first`.
* Use: `python -m riskplatform.cli export-excel --csv data\imports\prices.csv --out outputs\report.xlsx`
* Drop index proxies (for pre-inception stress windows) into a separate, clearly labelled file; they are not merged automatically.

## Review workflow
1. **Ingest & validate** - run the quality report; resolve FAIL items, review WARN items.
2. **Define the portfolio** - weights in percent, validated to sum to 100%.
3. **Risk decomposition** - volatility, correlation, risk contributions vs capital weights.
4. **Tail risk** - VaR/ES, then out-of-sample backtests; note low-power caveats.
5. **Stress** - observed events first, hypothetical shocks second, rate sensitivity third.
6. **Challenge** - compare models, vary windows, check sensitivity to the sample.
7. **Document** - export the workbook; write the brief by hand.

## Comparing with authorised institutional output (if access ever exists)
Export that system's position-level returns/risk to CSV, then compare on a shared basis: same dates and price adjustment, same return convention, same annualisation, same confidence/horizon, same covariance estimator and decay, and same risk-contribution definition. Expect differences from data vendor, factor models, full-revaluation vs delta-normal VaR and lookback weighting. Investigate differences by metric: volatility first, then correlations, then VaR. Never assume either side is "right" before reconciling definitions.
