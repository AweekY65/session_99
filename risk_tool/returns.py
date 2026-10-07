"""Return series computation with explicit missing-data rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import DataError

VALID_METHODS = ("simple", "log")
VALID_MISSING_RULES = ("drop", "ffill")


def compute_returns(
    prices: pd.DataFrame,
    method: str = "simple",
    missing: str = "drop",
) -> pd.DataFrame:
    """Compute asset returns from a price panel.

    Parameters
    ----------
    prices:
        DataFrame indexed by date, one column per asset, values are prices.
    method:
        ``"simple"`` for pct-change returns ``P_t / P_{t-1} - 1`` or
        ``"log"`` for log returns ``ln(P_t / P_{t-1})``.
    missing:
        Missing-data rule applied to *prices* before differencing:

        - ``"drop"``: any row (date) containing a NaN in any asset is
          removed entirely. This keeps the cross-section aligned.
        - ``"ffill"``: missing prices are forward-filled; rows that are
          still NaN at the start (no prior observation) are dropped.

    Returns
    -------
    DataFrame of returns with one fewer (or more, after drops) row than
    the cleaned price panel.
    """
    if method not in VALID_METHODS:
        raise DataError(f"unknown return method {method!r}; use {VALID_METHODS}")
    if missing not in VALID_MISSING_RULES:
        raise DataError(f"unknown missing rule {missing!r}; use {VALID_MISSING_RULES}")
    if prices.empty:
        raise DataError("price panel is empty")
    if (prices <= 0).any().any():
        bad = prices.columns[(prices <= 0).any()].tolist()
        raise DataError(f"non-positive prices found in columns: {bad}")

    if missing == "drop":
        clean = prices.dropna(axis=0, how="any")
    else:  # ffill
        clean = prices.ffill().dropna(axis=0, how="any")

    if len(clean) < 2:
        raise DataError(
            "fewer than 2 usable price rows after applying missing-data rule "
            f"{missing!r}; cannot compute returns"
        )

    if method == "simple":
        rets = clean.pct_change().iloc[1:]
    else:
        rets = np.log(clean / clean.shift(1)).iloc[1:]

    # Differencing must not introduce NaNs given the cleaning above; guard anyway.
    if rets.isna().any().any():
        raise DataError("NaN produced in return series; check input prices")
    return rets
