"""Local CSV loading. This is the ONLY data source the tool supports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import DataError
from .portfolio import Portfolio


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load a price CSV: first column is the date, one column per asset.

    Example::

        date,AAPL,MSFT
        2024-01-02,185.64,374.51
        ...
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"price file not found: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty or df.shape[1] == 0:
        raise DataError(f"price file {path} has no data")
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_holdings_csv(path: str | Path) -> Portfolio:
    """Load holdings CSV with columns ``asset,quantity``.

    Example::

        asset,quantity
        AAPL,100
        MSFT,-20
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"holdings file not found: {path}")
    df = pd.read_csv(path)
    cols = {c.strip().lower() for c in df.columns}
    if not {"asset", "quantity"} <= cols:
        raise DataError(
            f"holdings file {path} must have 'asset' and 'quantity' columns"
        )
    df.columns = [c.strip().lower() for c in df.columns]
    holdings = {str(r.asset): float(r.quantity) for r in df.itertuples()}
    return Portfolio(holdings)
