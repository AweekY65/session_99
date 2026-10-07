"""Deterministic stress scenarios: apply price shocks, compute P&L."""

from __future__ import annotations

import pandas as pd

from .errors import DataError
from .portfolio import Portfolio

ALL_ASSETS = "*"


def apply_stress(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    shocks: dict[str, float],
) -> dict[str, float | dict[str, float]]:
    """Apply proportional price shocks and compute portfolio P&L.

    ``shocks`` maps asset name -> relative price change, e.g.
    ``{"AAPL": -0.10, "MSFT": -0.05}``. The special key ``"*"`` applies
    a simultaneous shock to every asset; per-asset entries override it
    for their asset. Assets not covered by any entry are unshocked.

    P&L_i = q_i * P_i * shock_i (sign convention: profit positive).

    Returns a dict with total P&L, per-asset P&L and the shocked value.
    """
    if not shocks:
        raise DataError("stress scenario is empty")
    unknown = [a for a in shocks if a != ALL_ASSETS and a not in portfolio.holdings]
    if unknown:
        raise DataError(f"stress scenario references unknown assets: {unknown}")

    last = prices[portfolio.assets].iloc[-1]
    qty = portfolio.quantities(portfolio.assets)
    base_value = portfolio.current_value(prices)

    pnl_per_asset: dict[str, float] = {}
    for asset, q, price in zip(portfolio.assets, qty, last):
        shock = shocks.get(asset, shocks.get(ALL_ASSETS, 0.0))
        pnl_per_asset[asset] = float(q * price * shock)

    total_pnl = float(sum(pnl_per_asset.values()))
    return {
        "pnl": total_pnl,
        "pnl_per_asset": pnl_per_asset,
        "base_value": base_value,
        "stressed_value": base_value + total_pnl,
        "return": total_pnl / base_value,
    }
