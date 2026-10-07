"""Return series computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import InsufficientDataError

SIMPLE = "simple"
LOG = "log"


def compute_returns(prices: pd.DataFrame, method: str = SIMPLE) -> pd.DataFrame:
    """Compute per-asset returns from a cleaned price DataFrame.

    ``method`` is ``"simple"`` (P_t / P_{t-1} - 1) or ``"log"``
    (ln(P_t / P_{t-1})). The first price row is consumed as the base
    observation, so the result has one row fewer than ``prices``.
    """
    if method not in (SIMPLE, LOG):
        raise ValueError(f"unknown return method: {method!r} (use 'simple' or 'log')")
    if len(prices) < 2:
        raise InsufficientDataError(
            f"need at least 2 price rows to compute returns, got {len(prices)}"
        )
    if (prices <= 0).any().any():
        raise InsufficientDataError("prices must be strictly positive to compute returns")
    if method == SIMPLE:
        return prices.pct_change().iloc[1:]
    return np.log(prices / prices.shift(1)).iloc[1:]


def require_min_observations(returns: pd.DataFrame, min_obs: int, context: str) -> None:
    if len(returns) < min_obs:
        raise InsufficientDataError(
            f"{context}: need at least {min_obs} return observations, got {len(returns)}"
        )
