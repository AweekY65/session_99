"""Parametric (variance-covariance) VaR under a normality assumption."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .errors import CovarianceError
from .historical import validate_confidence
from .returns import require_min_observations

_EIGEN_TOL = -1e-10


def validate_covariance(cov: np.ndarray) -> np.ndarray:
    """Check that a covariance matrix is square, symmetric and PSD."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise CovarianceError(f"covariance matrix must be square, got shape {cov.shape}")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise CovarianceError("covariance matrix is not symmetric")
    min_eigen = float(np.linalg.eigvalsh(cov).min())
    if min_eigen < _EIGEN_TOL:
        raise CovarianceError(
            f"covariance matrix is not positive semi-definite (min eigenvalue {min_eigen:.3e})"
        )
    return cov


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio return volatility: sqrt(w' Sigma w)."""
    weights = np.asarray(weights, dtype=float)
    cov = validate_covariance(cov)
    if cov.shape[0] != weights.shape[0]:
        raise CovarianceError(
            f"covariance dimension {cov.shape[0]} does not match {weights.shape[0]} weights"
        )
    variance = float(weights @ cov @ weights)
    if variance < 0.0:  # numerical noise on a PSD matrix
        variance = 0.0
    return float(np.sqrt(variance))


def parametric_var(
    weights: np.ndarray,
    cov: np.ndarray,
    confidence: float,
    portfolio_value: float,
    mean_return: float = 0.0,
) -> float:
    """Parametric VaR as a positive loss amount in currency units.

    Assumes normally distributed portfolio returns:
    VaR = (z_conf * sigma_p - mu_p) * |portfolio_value|.
    """
    conf = validate_confidence(confidence)
    sigma_p = portfolio_volatility(weights, cov)
    z = NormalDist().inv_cdf(conf)
    var = (z * sigma_p - mean_return) * abs(portfolio_value)
    return float(var)


def estimate_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Sample covariance of asset returns (requires >= 2 observations)."""
    require_min_observations(returns, 2, "covariance estimation")
    cov = np.atleast_2d(np.cov(returns.to_numpy(dtype=float), rowvar=False, ddof=1))
    return validate_covariance(cov)
