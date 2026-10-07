"""Portfolio definition: static holdings over a local price panel."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .errors import DataError, ZeroValueError


@dataclass
class Portfolio:
    """A static set of holdings valued against a price panel.

    holdings maps asset name -> quantity. Negative quantities are short
    positions. The portfolio is assumed buy-and-hold: quantities do not
    change over the historical window.
    """

    holdings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.holdings:
            raise DataError("portfolio has no holdings")
        for asset, qty in self.holdings.items():
            if not np.isfinite(qty):
                raise DataError(f"holding for {asset!r} is not finite: {qty}")

    @property
    def assets(self) -> list[str]:
        return list(self.holdings.keys())

    def quantities(self, assets: list[str]) -> np.ndarray:
        try:
            return np.array([self.holdings[a] for a in assets], dtype=float)
        except KeyError as exc:
            raise DataError(f"no holding for asset {exc.args[0]!r}") from exc

    def value_series(self, prices: pd.DataFrame) -> pd.Series:
        """Portfolio market value at each date of the price panel."""
        missing = [a for a in self.assets if a not in prices.columns]
        if missing:
            raise DataError(f"holdings missing from price panel: {missing}")
        qty = self.quantities(self.assets)
        return (prices[self.assets] * qty).sum(axis=1)

    def current_value(self, prices: pd.DataFrame) -> float:
        """Market value at the last available date. Must be non-zero."""
        value = float(self.value_series(prices).iloc[-1])
        if value == 0.0:
            raise ZeroValueError(
                "portfolio market value is zero at the latest date; "
                "risk metrics relative to value are undefined"
            )
        return value

    def weights(self, prices: pd.DataFrame) -> np.ndarray:
        """Dollar weights w_i = q_i * P_i / V at the latest date.

        Weights sum to 1 (including cash-less shorts, which are negative).
        """
        value = self.current_value(prices)
        last = prices[self.assets].iloc[-1].to_numpy(dtype=float)
        return self.quantities(self.assets) * last / value

    def portfolio_returns(self, asset_returns: pd.DataFrame,
                          prices: pd.DataFrame) -> pd.Series:
        """Portfolio return series using fixed current weights.

        r_p,t = sum_i w_i * r_i,t  (linear approximation, standard for
        risk measurement over short horizons).
        """
        missing = [a for a in self.assets if a not in asset_returns.columns]
        if missing:
            raise DataError(f"holdings missing from return panel: {missing}")
        w = self.weights(prices)
        return asset_returns[self.assets] @ w
