"""Portfolio definition: positions, market values and weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import ZeroValuePortfolioError


@dataclass
class Portfolio:
    """A static portfolio of long/short cash positions.

    ``positions`` maps asset name -> quantity (signed; negative = short).
    """

    positions: dict[str, float]

    @property
    def assets(self) -> list[str]:
        return sorted(self.positions)

    def quantities(self, assets: list[str] | None = None) -> np.ndarray:
        assets = assets or self.assets
        return np.array([self.positions[a] for a in assets], dtype=float)

    def market_values(self, prices: pd.Series) -> np.ndarray:
        """Signed market value per asset at the given latest prices."""
        assets = self.assets
        qty = self.quantities(assets)
        px = prices[assets].to_numpy(dtype=float)
        return qty * px

    def total_value(self, prices: pd.Series) -> float:
        """Net portfolio value (sum of signed market values)."""
        return float(self.market_values(prices).sum())

    def weights(self, prices: pd.Series) -> np.ndarray:
        """Portfolio weights relative to total absolute market value.

        Raises ZeroValuePortfolioError if every position is worthless.
        """
        mv = self.market_values(prices)
        gross = np.abs(mv).sum()
        if gross <= 0:
            raise ZeroValuePortfolioError(
                "portfolio has zero total market value; risk metrics are undefined"
            )
        return mv / gross

    def pnl_series(self, asset_returns: pd.DataFrame, prices: pd.Series) -> pd.Series:
        """Historical portfolio P&L series: sum_i value_i * asset_return_i."""
        mv = self.market_values(prices)
        ret = asset_returns[self.assets].to_numpy(dtype=float)
        pnl = ret @ mv
        return pd.Series(pnl, index=asset_returns.index, name="pnl")
