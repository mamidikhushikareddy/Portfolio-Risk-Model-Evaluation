# Investment Risk Research Brief (one page)

> Historical research for education. Not investment advice. Hypothetical allocation; past results do not predict future results.
> Populate ONLY from verified, real data: `python -m riskplatform.cli brief --out outputs/research_brief.md`
> (the command refuses synthetic data and data with a FAIL quality status).

**Question:** How risky is this multi-asset portfolio, what drives its risk, and how reliable are the risk measures?

**Portfolio (hypothetical):** {{WEIGHTS}}
**Sample:** {{SAMPLE_START}} to {{SAMPLE_END}} ({{N_OBS}} trading days) | **Source:** {{SOURCE}} | **Retrieved (UTC):** {{RETRIEVED}} | **Data quality:** {{DQ_STATUS}}

## Key figures (historical)
| Annualised return (arithmetic) | Annualised volatility | Max drawdown | Sharpe | Beta vs SPY | 1-day 99% hist. VaR | 1-day 99% hist. ES |
|---|---|---|---|---|---|---|
| {{ANN_RETURN}} | {{ANN_VOL}} | {{MAX_DD}} | {{SHARPE}} | {{BETA}} | {{HIST_VAR99}} | {{HIST_ES99}} |

## What drives risk
{{TOP_RISK_ASSET}} is {{TOP_RISK_WEIGHT}} of capital but {{TOP_RISK_PCT}} of portfolio volatility (Euler decomposition).
Analyst commentary: _[write after reviewing the risk-contribution and correlation pages]_

## How reliable are the risk measures? (rolling out-of-sample VaR backtest)
{{BACKTEST_TABLE}}
Caveats: few expected breaches at 99% means low test power; a pass does not validate Expected Shortfall; breaches can cluster.

## Stress behaviour (observed history only; hypothetical shocks are in the app/workbook)
{{STRESS_TABLE}}

## Limitations
Short ETF histories, unofficial data source, survivorship bias, no transaction costs/taxes/liquidity effects, estimates depend on the sample window, normal-distribution assumption in parametric VaR.

## Analyst conclusion
_[Written by the analyst. Describe what the evidence supports and what it does not. No recommendations.]_
