"""Risk estimators: historical simulation, parametric, Monte Carlo.

Sign convention: losses are positive numbers. VaR and ES are reported in
currency units (same unit as prices * quantities) as a positive loss
amount. A VaR of 1000 at 99% means: with 99% confidence the one-period
loss will not exceed 1000.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .errors import (
    CovarianceError,
    InsufficientDataError,
    InvalidConfidenceError,
)

MIN_OBSERVATIONS = 2
_COV_EIGEN_TOL = 1e-10


def validate_confidence(confidence: float) -> float:
    """Confidence level must lie in the open interval (0, 1)."""
    c = float(confidence)
    if not 0.0 < c < 1.0:
        raise InvalidConfidenceError(
            f"confidence level must be in (0, 1), got {confidence!r}"
        )
    return c


def select_window(returns: pd.DataFrame | pd.Series, window: int | None):
    """Take the most recent ``window`` observations; validate sample size."""
    if window is not None:
        if window < MIN_OBSERVATIONS:
            raise InsufficientDataError(
                f"observation window must be >= {MIN_OBSERVATIONS}, got {window}"
            )
        returns = returns.iloc[-window:]
    if len(returns) < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_OBSERVATIONS} return observations, "
            f"got {len(returns)}"
        )
    return returns


def historical_var(
    portfolio_returns: pd.Series,
    confidence: float = 0.99,
    window: int | None = None,
    portfolio_value: float = 1.0,
) -> float:
    """Historical-simulation VaR as a positive loss amount.

    VaR = -quantile(r_p, 1 - confidence) * V, using the linear
    (type-7 style) quantile interpolation of numpy.
    """
    c = validate_confidence(confidence)
    rets = select_window(portfolio_returns, window)
    quantile = float(np.quantile(rets.to_numpy(dtype=float), 1.0 - c))
    return -quantile * portfolio_value


def historical_es(
    portfolio_returns: pd.Series,
    confidence: float = 0.99,
    window: int | None = None,
    portfolio_value: float = 1.0,
) -> float:
    """Historical-simulation Expected Shortfall (CVaR), positive loss.

    ES = -E[r_p | r_p <= quantile(r_p, 1 - confidence)] * V.
    """
    c = validate_confidence(confidence)
    rets = select_window(portfolio_returns, window)
    arr = rets.to_numpy(dtype=float)
    quantile = float(np.quantile(arr, 1.0 - c))
    tail = arr[arr <= quantile]
    if tail.size == 0:  # pragma: no cover - defensive
        raise InsufficientDataError("no observations in the loss tail")
    return -float(tail.mean()) * portfolio_value


def validate_covariance(cov: np.ndarray) -> np.ndarray:
    """Covariance matrix must be square, symmetric and positive semidefinite."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise CovarianceError(f"covariance matrix must be square, got {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise CovarianceError("covariance matrix contains NaN or inf")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise CovarianceError("covariance matrix is not symmetric")
    min_eig = float(np.linalg.eigvalsh(cov).min())
    if min_eig < -_COV_EIGEN_TOL:
        raise CovarianceError(
            f"covariance matrix is not positive semidefinite "
            f"(min eigenvalue {min_eig:.3e})"
        )
    return cov


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """sigma_p = sqrt(w' Sigma w). Validates the covariance matrix."""
    w = np.asarray(weights, dtype=float)
    cov = validate_covariance(cov)
    if cov.shape[0] != w.shape[0]:
        raise CovarianceError(
            f"covariance shape {cov.shape} does not match {w.shape[0]} weights"
        )
    var = float(w @ cov @ w)
    if var < 0.0:  # tiny negative from floating point noise
        if var > -_COV_EIGEN_TOL:
            var = 0.0
        else:
            raise CovarianceError(f"negative portfolio variance: {var}")
    return float(np.sqrt(var))


def parametric_var(
    weights: np.ndarray,
    cov: np.ndarray,
    confidence: float = 0.99,
    portfolio_value: float = 1.0,
    mean: np.ndarray | None = None,
) -> float:
    """Variance-covariance (delta-normal) VaR as a positive loss amount.

    Assumes portfolio returns are normally distributed:
    VaR = (z_c * sigma_p - mu_p) * V, with mu_p = w'mu (0 if not given).
    """
    c = validate_confidence(confidence)
    w = np.asarray(weights, dtype=float)
    sigma_p = portfolio_volatility(w, cov)
    mu_p = 0.0
    if mean is not None:
        mean = np.asarray(mean, dtype=float)
        if mean.shape != w.shape:
            raise CovarianceError(
                f"mean vector shape {mean.shape} does not match weights {w.shape}"
            )
        mu_p = float(w @ mean)
    z = NormalDist().inv_cdf(c)
    return (z * sigma_p - mu_p) * portfolio_value


def monte_carlo_var(
    weights: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    confidence: float = 0.99,
    portfolio_value: float = 1.0,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> dict[str, float]:
    """Monte Carlo VaR and ES from multivariate normal return scenarios.

    Draws ``n_simulations`` asset-return scenarios from N(mean, cov),
    computes portfolio P&L for each, and returns the empirical VaR/ES.
    With a fixed ``seed`` the result is exactly reproducible.
    """
    c = validate_confidence(confidence)
    if n_simulations < 100:
        raise InsufficientDataError(
            f"Monte Carlo needs at least 100 simulations, got {n_simulations}"
        )
    w = np.asarray(weights, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = validate_covariance(cov)
    if mean.shape != w.shape or cov.shape[0] != w.shape[0]:
        raise CovarianceError("mean/covariance shapes do not match weights")

    rng = np.random.default_rng(seed)
    scenarios = rng.multivariate_normal(mean, cov, size=n_simulations)
    pnl = scenarios @ w * portfolio_value  # profit positive, loss negative
    losses = -pnl
    var = float(np.quantile(losses, c))
    tail = losses[losses >= var]
    es = float(tail.mean())
    return {"var": var, "es": es, "simulations": float(n_simulations)}
