"""Historical-simulation VaR and Expected Shortfall."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import InvalidConfidenceLevelError
from .returns import require_min_observations

MIN_OBSERVATIONS = 2


def validate_confidence(confidence: float) -> float:
    conf = float(confidence)
    if not 0.0 < conf < 1.0:
        raise InvalidConfidenceLevelError(
            f"confidence level must be in the open interval (0, 1), got {confidence}"
        )
    return conf


def historical_var(pnl: pd.Series, confidence: float) -> float:
    """VaR as a positive loss amount from a P&L series.

    VaR = -quantile(P&L, 1 - confidence). A positive result means a loss.
    """
    conf = validate_confidence(confidence)
    require_min_observations(pnl.to_frame("pnl"), MIN_OBSERVATIONS, "historical VaR")
    quantile = float(np.quantile(pnl.to_numpy(dtype=float), 1.0 - conf))
    return -quantile


def historical_es(pnl: pd.Series, confidence: float) -> float:
    """Expected Shortfall: mean loss conditional on breaching VaR.

    Reported as a positive loss amount. ES >= VaR by construction.
    """
    conf = validate_confidence(confidence)
    require_min_observations(pnl.to_frame("pnl"), MIN_OBSERVATIONS, "historical ES")
    var = historical_var(pnl, conf)
    tail = pnl[pnl <= -var]
    if tail.empty:  # degenerate constant series
        return var
    return -float(tail.mean())
