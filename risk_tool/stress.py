"""Deterministic stress scenarios: apply relative price shocks."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .portfolio import Portfolio


@dataclass
class StressScenario:
    """A named set of relative price shocks.

    ``shocks`` maps asset name -> relative move (e.g. -0.15 for -15%).
    The special key ``ALL`` applies to every asset not explicitly listed.
    Assets without any applicable shock are left unchanged.
    """

    name: str
    shocks: dict[str, float] = field(default_factory=dict)

    def shock_for(self, asset: str) -> float:
        return self.shocks.get(asset, self.shocks.get("ALL", 0.0))


def apply_stress(
    portfolio: Portfolio, prices: pd.Series, scenario: StressScenario
) -> dict[str, object]:
    """Apply a stress scenario and compute portfolio P&L.

    Returns a dict with the scenario name, per-asset P&L and the total.
    """
    assets = portfolio.assets
    qty = portfolio.quantities(assets)
    px = prices[assets].to_numpy(dtype=float)
    shocks = pd.Series({a: scenario.shock_for(a) for a in assets})
    pnl_per_asset = qty * px * shocks.to_numpy(dtype=float)
    return {
        "scenario": scenario.name,
        "pnl_per_asset": dict(zip(assets, pnl_per_asset)),
        "total_pnl": float(pnl_per_asset.sum()),
    }
