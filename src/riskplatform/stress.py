"""Stress testing: historical event replay, hypothetical shocks, duration approximation.

Historical replay uses ONLY observed prices of the weighted assets. If any held asset
lacks data across the event window, the portfolio result is reported as unavailable
(never invented). Hypothetical shocks and duration approximations are labelled as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def replay_event(prices: pd.DataFrame, weights: pd.Series, name: str, start: str, end: str,
                 max_edge_gap_days: int = 7) -> dict:
    """Replay an event window from the last close BEFORE ``start`` to the close on/before ``end``."""
    held = weights[weights > 0]
    px = prices[list(held.index)].dropna(how="all")
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    res: dict = {"event": name, "kind": "historical replay (observed prices)", "start": s, "end": e,
                 "portfolio_available": False, "reason": None, "asset_table": pd.DataFrame(), "path": None}
    pre = px.index[px.index < s]
    if len(pre) == 0:
        res["reason"] = "No price data before event start"
        return res
    base = pre[-1]
    window = px.loc[base:e]
    if (s - base).days > max_edge_gap_days or window.index[-1] < e - pd.Timedelta(days=max_edge_gap_days):
        res["reason"] = "Data does not cover the event window edges"
        return res
    rows, ok = [], []
    for t in held.index:
        col = window[t]
        if col.isna().any():
            rows.append({"asset": t, "weight": held[t], "event_return": np.nan, "status": "unavailable (missing data)"})
        else:
            rows.append({"asset": t, "weight": held[t], "event_return": float(col.iloc[-1] / col.iloc[0] - 1),
                         "status": "observed"})
            ok.append(t)
    table = pd.DataFrame(rows).set_index("asset")
    table["contribution_buy_hold"] = table["weight"] * table["event_return"]
    res["asset_table"] = table
    if len(ok) != len(held):
        res["reason"] = "Not all weighted assets have observed prices in the window; portfolio result withheld"
        return res
    rets = (window / window.shift(1) - 1).iloc[1:]
    rp = pd.Series(rets.values @ held.values, index=rets.index)
    wealth = (1 + rp).cumprod()
    ext = pd.concat([pd.Series([1.0], index=[base]), wealth])
    trough_date = ext.idxmin()
    res.update(portfolio_available=True, reason=None, path=wealth,
               portfolio_return_daily_rebalanced=float(wealth.iloc[-1] - 1),
               portfolio_return_buy_and_hold=float(table["contribution_buy_hold"].sum()),
               worst_day=float(rp.min()), worst_day_date=rp.idxmin(),
               max_drawdown_in_window=float((ext / np.maximum(ext.cummax(), 1.0) - 1).min()),
               trough_date=None if ext.min() >= 1.0 else trough_date,
               trading_days=int(len(rp)))
    # Recovery: continue the same constant-weight path after the window (observed data only).
    tail = px.loc[base:]
    tr = (tail / tail.shift(1) - 1).iloc[1:]
    bad = np.where(tr.isna().any(axis=1).values)[0]
    tr = tr.iloc[: bad[0]] if len(bad) else tr
    full = (1 + pd.Series(tr.values @ held.values, index=tr.index)).cumprod()
    res["recovery_date"], res["days_trough_to_recovery"] = None, None
    if res["trough_date"] is not None:
        after = full.loc[full.index > res["trough_date"]]
        rec = after[after >= 1.0]
        if len(rec):
            res["recovery_date"] = rec.index[0]
            res["days_trough_to_recovery"] = int(full.index.get_loc(rec.index[0]) - full.index.get_loc(res["trough_date"]))
        res["recovery_note"] = ("recovered to pre-event level" if len(rec)
                                else "not recovered within available observed data")
    return res


def hypothetical_shock(weights: pd.Series, shocks: dict[str, float], name: str = "Hypothetical") -> dict:
    """Instantaneous price shocks (fractions). Every held asset needs an explicit shock."""
    held = weights[weights > 0]
    missing = [t for t in held.index if t not in shocks]
    if missing:
        raise ValueError(f"Shock not specified for held assets {missing}; no silent zero is assumed")
    table = pd.DataFrame({"weight": held, "shock": pd.Series({t: shocks[t] for t in held.index})})
    table["contribution"] = table["weight"] * table["shock"]
    return {"scenario": name, "kind": "HYPOTHETICAL (not observed history)",
            "portfolio_return": float(table["contribution"].sum()), "table": table}


def duration_approximation(weights: pd.Series, yield_change_bp: float, durations: dict[str, float],
                           convexities: dict[str, float] | None = None) -> dict:
    """Parallel-shift price approximation dP/P ~ -D*dy + 0.5*C*dy^2 for bond ETFs only.

    Non-bond assets are NOT modelled (shown as unmodelled, not as zero risk). Real bond
    ETFs have changing duration, non-parallel curve moves and credit spread effects.
    """
    dy = yield_change_bp / 10_000.0
    rows = []
    for t, d in durations.items():
        w = float(weights.get(t, 0.0))
        c = (convexities or {}).get(t, 0.0)
        price_change = -d * dy + 0.5 * c * dy ** 2
        rows.append({"asset": t, "weight": w, "duration_assumed": d, "convexity_assumed": c,
                     "approx_price_return": price_change, "contribution": w * price_change})
    table = pd.DataFrame(rows).set_index("asset")
    return {"scenario": f"Parallel yield shift {yield_change_bp:+.0f} bp", "kind": "HYPOTHETICAL duration approximation",
            "bond_contribution_to_portfolio": float(table["contribution"].sum()), "table": table,
            "caveat": "Bond ETFs only; other assets unmodelled. Duration/convexity are user assumptions."}


def yield_context(yields: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Level changes (percentage points) in Treasury yields over a window, using the last
    valid observation on/before each date. CONTEXT ONLY - not bond total returns."""
    rows = []
    for col in yields.columns:
        s = yields[col].dropna()
        before, upto = s.loc[: pd.Timestamp(start) - pd.Timedelta(days=1)], s.loc[: pd.Timestamp(end)]
        if before.empty or upto.empty:
            rows.append({"series": col, "start_level_pct": np.nan, "end_level_pct": np.nan, "change_pp": np.nan})
            continue
        rows.append({"series": col, "start_level_pct": float(before.iloc[-1]), "end_level_pct": float(upto.iloc[-1]),
                     "change_pp": float(upto.iloc[-1] - before.iloc[-1])})
    return pd.DataFrame(rows).set_index("series")
