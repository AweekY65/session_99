"""Local CSV loading and missing-data handling.

Missing-data rules (applied in this order):
1. Rows are sorted by date and duplicate dates are dropped (keep last).
2. Missing prices are forward-filled (a stale price is carried forward).
3. Any row that still contains a NaN after forward-filling (i.e. leading
   gaps before the first observed price of an asset) is dropped entirely,
   so every remaining row has a complete price for every asset.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import MissingDataError


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load a price CSV.

    Expected format: a ``date`` column plus one column per asset, e.g.::

        date,AAA,BBB
        2024-01-02,100.0,50.0

    Returns a DataFrame indexed by date with float columns per asset,
    cleaned according to the missing-data rules in the module docstring.
    """
    path = Path(path)
    if not path.is_file():
        raise MissingDataError(f"price file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        raise MissingDataError("price CSV must contain a 'date' column")
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape[1] == 0:
        raise MissingDataError("price CSV contains no asset columns")
    df = df.ffill().dropna()
    if df.empty:
        raise MissingDataError("no complete price rows remain after cleaning")
    return df


def load_positions_csv(path: str | Path) -> dict[str, float]:
    """Load positions CSV with columns ``asset,quantity``."""
    path = Path(path)
    if not path.is_file():
        raise MissingDataError(f"positions file not found: {path}")
    positions: dict[str, float] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not {"asset", "quantity"} <= set(reader.fieldnames):
            raise MissingDataError("positions CSV must have 'asset' and 'quantity' columns")
        for row in reader:
            positions[row["asset"]] = positions.get(row["asset"], 0.0) + float(row["quantity"])
    if not positions:
        raise MissingDataError("positions CSV is empty")
    return positions


def load_stress_csv(path: str | Path) -> dict[str, float]:
    """Load stress shocks CSV with columns ``asset,shock``.

    ``asset`` may be ``ALL`` to apply the shock to every asset.
    ``shock`` is a relative price move, e.g. ``-0.20`` for -20%.
    """
    path = Path(path)
    if not path.is_file():
        raise MissingDataError(f"stress file not found: {path}")
    shocks: dict[str, float] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not {"asset", "shock"} <= set(reader.fieldnames):
            raise MissingDataError("stress CSV must have 'asset' and 'shock' columns")
        for row in reader:
            shocks[row["asset"]] = float(row["shock"])
    if not shocks:
        raise MissingDataError("stress CSV is empty")
    return shocks


def align_prices_to_assets(prices: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Select and order price columns to match the portfolio assets."""
    missing = [a for a in assets if a not in prices.columns]
    if missing:
        raise MissingDataError(f"no price data for assets: {', '.join(missing)}")
    return prices[assets]


def window_slice(returns: pd.DataFrame, window: int | None) -> pd.DataFrame:
    """Keep the most recent ``window`` observations (None = all)."""
    if window is None:
        return returns
    if window <= 0:
        raise MissingDataError("observation window must be a positive integer")
    return returns.tail(window)


def as_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float)
