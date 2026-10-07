"""Automated tests for the local risk tool. All output is terminal-only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_tool import (
    CovarianceError,
    DataError,
    InsufficientDataError,
    InvalidConfidenceError,
    Portfolio,
    ZeroValueError,
    apply_stress,
    compute_returns,
    historical_es,
    historical_var,
    monte_carlo_var,
    parametric_var,
)
from risk_tool.risk import portfolio_volatility

IDX = pd.date_range("2024-01-01", periods=12, freq="D")


def make_prices(data: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(data, index=IDX[: len(next(iter(data.values())))])


# ---------------------------------------------------------------- returns

def test_simple_returns_known_series():
    prices = make_prices({"A": [100.0, 110.0, 99.0, 132.0]})
    rets = compute_returns(prices, method="simple")
    expected = [0.10, -0.10, 1.0 / 3.0]
    assert np.allclose(rets["A"].to_numpy(), expected)
    print("\nsimple returns:", np.round(rets["A"].to_numpy(), 6).tolist())


def test_log_returns_known_series():
    prices = make_prices({"A": [100.0, 110.0, 99.0]})
    rets = compute_returns(prices, method="log")
    expected = [np.log(1.1), np.log(0.9)]
    assert np.allclose(rets["A"].to_numpy(), expected)
    print("\nlog returns:", np.round(rets["A"].to_numpy(), 6).tolist())


def test_missing_data_drop_rule():
    prices = make_prices({"A": [100.0, np.nan, 110.0, 121.0],
                          "B": [50.0, 51.0, 52.0, 53.0]})
    rets = compute_returns(prices, missing="drop")
    # date 2 dropped for the whole cross-section; returns span the gap
    assert len(rets) == 2
    assert np.allclose(rets["B"].to_numpy(), [52.0 / 50.0 - 1, 53.0 / 52.0 - 1])


def test_missing_data_ffill_rule():
    prices = make_prices({"A": [100.0, np.nan, 110.0, 121.0]})
    rets = compute_returns(prices, missing="ffill")
    assert len(rets) == 3
    assert rets["A"].iloc[0] == pytest.approx(0.0)  # filled 100 -> 100


def test_unknown_method_and_rule_rejected():
    prices = make_prices({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(DataError):
        compute_returns(prices, method="geometric")
    with pytest.raises(DataError):
        compute_returns(prices, missing="interpolate")


# ------------------------------------------------------- historical VaR/ES

def test_historical_var_known_series_single_asset():
    # 100 returns: losses of -1%..-10% (one each), then 90 x +1%
    values = [-0.01 * k for k in range(1, 11)] + [0.01] * 90
    rets = pd.Series(values)
    var = historical_var(rets, confidence=0.95, portfolio_value=1_000_000)
    # sorted: -0.10..-0.01 at idx 0..9; 5% quantile at position 4.95
    # interpolates between -0.06 (idx 4) and -0.05 (idx 5) -> -0.0505
    assert var == pytest.approx(50500.0)
    print(f"\nhistorical VaR 95% on known series: {var:,.2f}")


def test_historical_es_known_series():
    values = [-0.05, -0.04, -0.03] + [0.01] * 97
    rets = pd.Series(values)
    es = historical_es(rets, confidence=0.98, portfolio_value=1_000_000)
    # 2% quantile interpolates between -0.05 and -0.04 -> tail = both points
    assert es == pytest.approx(45000.0)
    print(f"\nhistorical ES 98% on known series: {es:,.2f}")


def test_historical_var_window_selection():
    rets = pd.Series([0.5] * 10 + [-0.02, 0.01, 0.01, 0.01, 0.01])
    var = historical_var(rets, confidence=0.95, window=5, portfolio_value=100)
    # last 5 obs: [-0.02, 0.01 x4]; 5% quantile at position 0.2 -> -0.014
    assert var == pytest.approx(1.4)


def test_es_at_least_as_large_as_var():
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0, 0.02, 500))
    var = historical_var(rets, 0.95, portfolio_value=100)
    es = historical_es(rets, 0.95, portfolio_value=100)
    assert es >= var


# ------------------------------------------------------- parametric VaR

def test_parametric_var_two_correlated_assets():
    # Two assets, sigma 2% and 1%, correlation 0.5, equal weights.
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0001]])
    w = np.array([0.5, 0.5])
    sigma_p = portfolio_volatility(w, cov)
    # w'Σw = 0.25*4e-4 + 0.25*1e-4 + 2*0.25*1e-4 = 1.75e-4
    assert sigma_p == pytest.approx(np.sqrt(1.75e-4))
    var = parametric_var(w, cov, confidence=0.99, portfolio_value=1_000_000)
    from statistics import NormalDist
    expected = NormalDist().inv_cdf(0.99) * np.sqrt(1.75e-4) * 1_000_000
    assert var == pytest.approx(expected)
    print(f"\nparametric VaR 99% (2 correlated assets): {var:,.2f}")


def test_parametric_var_with_mean_reduces_var():
    cov = np.array([[0.0004]])
    w = np.array([1.0])
    var0 = parametric_var(w, cov, 0.99, 100.0)
    var_pos = parametric_var(w, cov, 0.99, 100.0, mean=np.array([0.001]))
    assert var_pos < var0


def test_diversification_lowers_volatility():
    cov_indep = np.array([[0.0004, 0.0], [0.0, 0.0004]])
    cov_corr = np.array([[0.0004, 0.0004], [0.0004, 0.0004]])
    w = np.array([0.5, 0.5])
    assert portfolio_volatility(w, cov_indep) < portfolio_volatility(w, cov_corr)


# --------------------------------------------------------- Monte Carlo

def test_monte_carlo_reproducible_with_seed():
    mean = np.array([0.0005, -0.0002])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0002]])
    w = np.array([0.6, 0.4])
    r1 = monte_carlo_var(w, mean, cov, 0.99, 100_000, 5000, seed=42)
    r2 = monte_carlo_var(w, mean, cov, 0.99, 100_000, 5000, seed=42)
    r3 = monte_carlo_var(w, mean, cov, 0.99, 100_000, 5000, seed=7)
    assert r1 == r2, "same seed must give identical results"
    assert r1 != r3, "different seed should give different results"
    print(f"\nMC VaR (seed=42): {r1['var']:,.2f}  ES: {r1['es']:,.2f}")


def test_monte_carlo_converges_to_parametric():
    rng = np.random.default_rng(0)
    mean = np.array([0.0, 0.0])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0001]])
    w = np.array([0.5, 0.5])
    mc = monte_carlo_var(w, mean, cov, 0.975, 1.0, 200_000, seed=123)
    pv = parametric_var(w, cov, 0.975, 1.0)
    assert mc["var"] == pytest.approx(pv, rel=0.02)


# ------------------------------------------------------------- stress

def test_stress_simultaneous_and_individual():
    prices = make_prices({"A": [100.0] * 3, "B": [200.0] * 3})
    pf = Portfolio({"A": 10, "B": 5})  # value = 1000 + 1000 = 2000
    both = apply_stress(pf, prices, {"*": -0.10})
    assert both["pnl"] == pytest.approx(-200.0)
    one = apply_stress(pf, prices, {"A": -0.10})
    assert one["pnl"] == pytest.approx(-100.0)
    assert one["pnl_per_asset"]["B"] == 0.0
    mixed = apply_stress(pf, prices, {"*": -0.05, "B": 0.20})
    assert mixed["pnl"] == pytest.approx(-50.0 + 200.0)
    print(f"\nstress -10% all: {both['pnl']:,.2f}; A only: {one['pnl']:,.2f}")


def test_stress_unknown_asset_rejected():
    prices = make_prices({"A": [100.0] * 3})
    pf = Portfolio({"A": 1})
    with pytest.raises(DataError):
        apply_stress(pf, prices, {"NOPE": -0.1})


# ------------------------------------------------------- invalid inputs

def test_invalid_confidence_levels():
    rets = pd.Series([0.01, -0.01, 0.02])
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(InvalidConfidenceError):
            historical_var(rets, confidence=bad)
        with pytest.raises(InvalidConfidenceError):
            parametric_var(np.array([1.0]), np.array([[0.01]]), confidence=bad)


def test_insufficient_samples():
    with pytest.raises(InsufficientDataError):
        historical_var(pd.Series([0.01]), confidence=0.95)
    with pytest.raises(InsufficientDataError):
        historical_var(pd.Series(range(100), dtype=float), confidence=0.95,
                       window=1)
    with pytest.raises(InsufficientDataError):
        monte_carlo_var(np.array([1.0]), np.array([0.0]), np.array([[0.01]]),
                        0.99, n_simulations=10)


def test_covariance_anomalies():
    w = np.array([0.5, 0.5])
    with pytest.raises(CovarianceError):  # not symmetric
        portfolio_volatility(w, np.array([[0.01, 0.05], [0.0, 0.01]]))
    with pytest.raises(CovarianceError):  # not PSD
        portfolio_volatility(w, np.array([[0.01, 0.2], [0.2, 0.01]]))
    with pytest.raises(CovarianceError):  # NaN
        portfolio_volatility(w, np.array([[0.01, np.nan], [np.nan, 0.01]]))
    with pytest.raises(CovarianceError):  # shape mismatch
        portfolio_volatility(np.array([1.0]), np.eye(2))


def test_zero_value_portfolio():
    prices = make_prices({"A": [100.0] * 3, "B": [100.0] * 3})
    pf = Portfolio({"A": 1, "B": -1})  # perfectly offsetting -> value 0
    with pytest.raises(ZeroValueError):
        pf.current_value(prices)


def test_portfolio_validation():
    with pytest.raises(DataError):
        Portfolio({})
    with pytest.raises(DataError):
        Portfolio({"A": float("nan")})
    prices = make_prices({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(DataError):
        Portfolio({"B": 1}).current_value(prices)


# ------------------------------------------------ multi-asset integration

def test_multi_asset_portfolio_end_to_end():
    rng = np.random.default_rng(11)
    n = 300
    cov = np.array([[0.0004, 0.0002], [0.0002, 0.0009]])
    rets = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    pdf = pd.DataFrame(prices, columns=["X", "Y"],
                       index=pd.date_range("2024-01-01", periods=n))
    pf = Portfolio({"X": 50, "Y": 20})
    rdf = compute_returns(pdf)
    prets = pf.portfolio_returns(rdf, pdf)
    value = pf.current_value(pdf)
    w = pf.weights(pdf)
    assert w.sum() == pytest.approx(1.0)

    hv = historical_var(prets, 0.99, window=250, portfolio_value=value)
    he = historical_es(prets, 0.99, window=250, portfolio_value=value)
    pv = parametric_var(w, rdf[["X", "Y"]].cov().to_numpy(), 0.99, value)
    assert hv > 0 and he >= hv and pv > 0
    print(f"\nend-to-end 99%: histVaR={hv:,.2f} histES={he:,.2f} "
          f"paramVaR={pv:,.2f} value={value:,.2f}")
