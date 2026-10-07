"""Monte Carlo risk estimation with multivariate normal scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .historical import validate_confidence
from .parametric import validate_covariance
from .returns import require_min_observations


def monte_carlo_risk(
    mean: np.ndarray,
    cov: np.ndarray,
    market_values: np.ndarray,
    confidence: float,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> dict[str, float]:
    """Simulate one-period P&L and return VaR / Expected Shortfall.

    Asset returns are drawn from a multivariate normal N(mean, cov) via a
    Cholesky factor. Passing a fixed ``seed`` makes the result fully
    reproducible. Returns a dict with keys ``var`` and ``es`` expressed as
    positive loss amounts in currency units.
    """
    conf = validate_confidence(confidence)
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
    mean = np.asarray(mean, dtype=float)
    cov = validate_covariance(cov)
    market_values = np.asarray(market_values, dtype=float)
    if mean.shape[0] != cov.shape[0] or market_values.shape[0] != cov.shape[0]:
        raise ValueError("mean, covariance and market values dimensions must match")

    rng = np.random.default_rng(seed)
    # PSD matrices may be rank-deficient; use eigen-decomposition for safety.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    factor = eigvecs @ np.diag(np.sqrt(eigvals))
    z = rng.standard_normal((n_simulations, cov.shape[0]))
    simulated_returns = mean + z @ factor.T
    pnl = simulated_returns @ market_values

    var = -float(np.quantile(pnl, 1.0 - conf))
    tail = pnl[pnl <= -var]
    es = -float(tail.mean()) if tail.size else var
    return {"var": var, "es": es}


def estimate_from_returns(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean vector and covariance matrix of asset returns."""
    require_min_observations(returns, 2, "Monte Carlo estimation")
    data = returns.to_numpy(dtype=float)
    return data.mean(axis=0), np.atleast_2d(np.cov(data, rowvar=False, ddof=1))
