"""Automated tests for the local portfolio risk tool.

Run with ``python -m pytest -s`` to see the computed risk metrics printed
to the terminal (no chart windows are used anywhere).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_tool import (
    CovarianceError,
    InsufficientDataError,
    InvalidConfidenceLevelError,
    Portfolio,
    StressScenario,
    ZeroValuePortfolioError,
    apply_stress,
    compute_returns,
    historical_es,
    historical_var,
    monte_carlo_risk,
    parametric_var,
    portfolio_volatility,
)
from risk_tool.data import align_prices_to_assets, load_prices_csv, window_slice
from risk_tool.errors import MissingDataError
from risk_tool.montecarlo import estimate_from_returns
from risk_tool.parametric import estimate_covariance


def make_prices(rows: list[list[float]], assets=("A",)) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=list(assets))


# ---------------------------------------------------------------- returns

def test_simple_returns_known_series():
    prices = make_prices([[100.0], [110.0], [99.0]])
    ret = compute_returns(prices, "simple")
    assert np.allclose(ret["A"].to_numpy(), [0.10, -0.10])
    print(f"\nsimple returns: {ret['A'].round(4).tolist()}")


def test_log_returns_known_series():
    prices = make_prices([[100.0], [110.0], [99.0]])
    ret = compute_returns(prices, "log")
    expected = [np.log(1.1), np.log(0.9)]
    assert np.allclose(ret["A"].to_numpy(), expected)
    print(f"\nlog returns: {ret['A'].round(6).tolist()}")


def test_missing_data_forward_fill_then_drop(tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "date,A,B\n"
        "2024-01-02,,10\n"      # leading gap for A -> row dropped
        "2024-01-03,100,\n"     # B forward-filled from 10
        "2024-01-04,101,11\n"
        "2024-01-05,102,12\n"
    )
    prices = load_prices_csv(csv_path)
    assert len(prices) == 3
    assert prices.loc[pd.Timestamp("2024-01-03"), "B"] == 10.0
    assert prices.iloc[0]["A"] == 100.0
    print(f"\ncleaned prices after ffill+drop:\n{prices}")


# ------------------------------------------------------- historical VaR/ES

def test_historical_var_known_pnl():
    pnl = pd.Series([-50.0, -30.0, -20.0, -10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    var = historical_var(pnl, 0.90)
    # 10% quantile with linear interpolation: -50 + 0.9 * 20 = -32.
    assert var == pytest.approx(32.0)
    print(f"\nhistorical VaR(90%) on known P&L: {var:.2f}")


def test_historical_es_known_pnl():
    pnl = pd.Series([-50.0, -30.0, -20.0, -10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    es = historical_es(pnl, 0.90)
    var = historical_var(pnl, 0.90)
    # Tail beyond VaR: only -50 is <= -32, so ES = 50.
    assert es == pytest.approx(50.0)
    assert es >= var
    print(f"\nhistorical ES(90%) on known P&L: {es:.2f} (VaR {var:.2f})")


def test_single_asset_portfolio_historical():
    prices = make_prices([[100.0], [101.0], [99.0], [102.0], [98.0]])
    portfolio = Portfolio({"A": 10})
    returns = compute_returns(prices, "simple")
    pnl = portfolio.pnl_series(returns, prices.iloc[-1])
    var = historical_var(pnl, 0.80)
    es = historical_es(pnl, 0.80)
    assert var > 0 and es >= var
    print(f"\nsingle asset: VaR(80%)={var:.2f}, ES(80%)={es:.2f}")


def test_correlated_assets_parametric():
    rng = np.random.default_rng(1)
    cov_true = np.array([[0.0004, 0.0003], [0.0003, 0.0009]])
    sims = rng.multivariate_normal(np.zeros(2), cov_true, size=2000)
    idx = pd.bdate_range("2024-01-02", periods=2001)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(sims, axis=0)), index=idx[1:], columns=["X", "Y"])
    prices = pd.concat([pd.DataFrame([[100.0, 100.0]], index=idx[:1], columns=["X", "Y"]), prices])

    portfolio = Portfolio({"X": 10, "Y": 5})
    returns = compute_returns(prices, "simple")
    cov = estimate_covariance(returns)
    weights = portfolio.weights(prices.iloc[-1])
    vol = portfolio_volatility(weights, cov)
    var = parametric_var(weights, cov, 0.99, abs(portfolio.market_values(prices.iloc[-1])).sum())
    assert vol > 0 and var > 0
    print(f"\ncorrelated assets: vol={vol:.6f}, parametric VaR(99%)={var:.2f}")
    # Correlation should be recovered approximately.
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert corr == pytest.approx(0.5, abs=0.1)


# ------------------------------------------------------------ Monte Carlo

def test_monte_carlo_deterministic_with_seed():
    mean = np.array([0.001, -0.0005])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    mv = np.array([1000.0, -500.0])
    first = monte_carlo_risk(mean, cov, mv, 0.99, n_simulations=5000, seed=123)
    second = monte_carlo_risk(mean, cov, mv, 0.99, n_simulations=5000, seed=123)
    assert first == second
    assert first["es"] >= first["var"]
    print(f"\nMC(seed=123): VaR={first['var']:.2f}, ES={first['es']:.2f} (reproducible)")


def test_monte_carlo_close_to_parametric_single_asset():
    rng = np.random.default_rng(5)
    rets = pd.DataFrame({"A": rng.normal(0.0, 0.02, size=500)})
    mean, cov = estimate_from_returns(rets)
    mv = np.array([10_000.0])
    mc = monte_carlo_risk(mean, cov, mv, 0.99, n_simulations=100_000, seed=9)
    para = parametric_var(np.array([1.0]), cov, 0.99, 10_000.0, mean_return=float(mean[0]))
    assert mc["var"] == pytest.approx(para, rel=0.05)
    print(f"\nMC VaR={mc['var']:.2f} vs parametric VaR={para:.2f}")


# ----------------------------------------------------------------- stress

def test_stress_scenario_per_asset_and_all():
    prices = make_prices([[100.0, 200.0]], assets=("A", "B")).iloc[-1]
    portfolio = Portfolio({"A": 10, "B": -5})  # long A, short B
    scenario = StressScenario("crash", {"ALL": -0.10, "B": -0.20})
    result = apply_stress(portfolio, prices, scenario)
    # A: 10 * 100 * -0.10 = -100 ; B: -5 * 200 * -0.20 = +200
    assert result["pnl_per_asset"]["A"] == pytest.approx(-100.0)
    assert result["pnl_per_asset"]["B"] == pytest.approx(200.0)
    assert result["total_pnl"] == pytest.approx(100.0)
    print(f"\nstress P&L per asset: {result['pnl_per_asset']}, total={result['total_pnl']:.2f}")


# ----------------------------------------------------------------- errors

def test_invalid_confidence_levels():
    pnl = pd.Series([1.0, -1.0, 0.0])
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(InvalidConfidenceLevelError):
            historical_var(pnl, bad)
    print("\ninvalid confidence levels correctly rejected: 0.0, 1.0, -0.5, 1.5")


def test_insufficient_data():
    prices = make_prices([[100.0]])
    with pytest.raises(InsufficientDataError):
        compute_returns(prices, "simple")
    with pytest.raises(InsufficientDataError):
        historical_var(pd.Series([1.0]), 0.95)
    print("\ninsufficient data correctly rejected")


def test_covariance_anomalies():
    with pytest.raises(CovarianceError):  # not positive semi-definite
        portfolio_volatility(np.array([0.5, 0.5]), np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(CovarianceError):  # asymmetric
        portfolio_volatility(np.array([0.5, 0.5]), np.array([[1.0, 0.9], [0.1, 1.0]]))
    with pytest.raises(CovarianceError):  # not square
        portfolio_volatility(np.array([1.0]), np.array([[1.0, 0.0]]))
    print("\nanomalous covariance matrices correctly rejected")


def test_zero_value_portfolio():
    portfolio = Portfolio({"A": 0.0, "B": 0.0})
    prices = make_prices([[100.0, 50.0]], assets=("A", "B")).iloc[-1]
    with pytest.raises(ZeroValuePortfolioError):
        portfolio.weights(prices)
    print("\nzero-value portfolio correctly rejected")


def test_missing_asset_prices():
    prices = make_prices([[100.0], [101.0]], assets=("A",))
    with pytest.raises(MissingDataError):
        align_prices_to_assets(prices, ["A", "ZZZ"])
    print("\nmissing asset price column correctly rejected")


def test_window_slice():
    prices = make_prices([[float(p)] for p in range(100, 110)])
    returns = compute_returns(prices, "simple")
    sliced = window_slice(returns, 3)
    assert len(sliced) == 3
    assert (sliced.index == returns.index[-3:]).all()
    print(f"\nwindow slice keeps last 3 of {len(returns)} observations")
