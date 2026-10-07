"""Terminal CLI: compute portfolio market risk from local CSV files only."""

from __future__ import annotations

import argparse
import sys

from .data import (
    align_prices_to_assets,
    load_positions_csv,
    load_prices_csv,
    load_stress_csv,
    window_slice,
)
from .errors import RiskToolError
from .historical import historical_es, historical_var, validate_confidence
from .montecarlo import estimate_from_returns, monte_carlo_risk
from .parametric import estimate_covariance, parametric_var, portfolio_volatility
from .portfolio import Portfolio
from .returns import compute_returns
from .stress import StressScenario, apply_stress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk_tool",
        description="Local portfolio market risk tool (CSV inputs only, no external services).",
    )
    parser.add_argument("--prices", required=True, help="CSV: date column + one column per asset")
    parser.add_argument("--positions", required=True, help="CSV: asset,quantity")
    parser.add_argument("--confidence", type=float, default=0.99, help="confidence level in (0,1)")
    parser.add_argument("--window", type=int, default=None, help="observation window (default: all)")
    parser.add_argument(
        "--return-type", choices=["simple", "log"], default="simple", help="return convention"
    )
    parser.add_argument("--mc-sims", type=int, default=10_000, help="Monte Carlo simulations")
    parser.add_argument("--seed", type=int, default=42, help="Monte Carlo random seed")
    parser.add_argument("--stress", default=None, help="optional CSV: asset,shock (asset may be ALL)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_confidence(args.confidence)
        prices = load_prices_csv(args.prices)
        portfolio = Portfolio(load_positions_csv(args.positions))
        prices = align_prices_to_assets(prices, portfolio.assets)
        returns = window_slice(compute_returns(prices, args.return_type), args.window)

        latest = prices.iloc[-1]
        weights = portfolio.weights(latest)
        gross_value = float(abs(portfolio.market_values(latest)).sum())
        pnl = portfolio.pnl_series(returns, latest)

        cov = estimate_covariance(returns)
        mean, _ = estimate_from_returns(returns)

        print(f"assets              : {', '.join(portfolio.assets)}")
        print(f"observations        : {len(returns)} (return type: {args.return_type})")
        print(f"gross market value  : {gross_value:,.2f}")
        print(f"confidence level    : {args.confidence:.2%}")
        print(f"portfolio volatility: {portfolio_volatility(weights, cov):.6f} (per period)")
        print(f"historical VaR      : {historical_var(pnl, args.confidence):,.2f}")
        print(f"historical ES       : {historical_es(pnl, args.confidence):,.2f}")
        print(
            f"parametric VaR      : "
            f"{parametric_var(weights, cov, args.confidence, gross_value):,.2f}"
        )
        mc = monte_carlo_risk(
            mean, cov, portfolio.market_values(latest), args.confidence,
            n_simulations=args.mc_sims, seed=args.seed,
        )
        print(f"monte carlo VaR     : {mc['var']:,.2f} (sims={args.mc_sims}, seed={args.seed})")
        print(f"monte carlo ES      : {mc['es']:,.2f}")

        if args.stress:
            scenario = StressScenario(name="csv-scenario", shocks=load_stress_csv(args.stress))
            result = apply_stress(portfolio, latest, scenario)
            print(f"stress scenario     : {result['scenario']}")
            for asset, value in result["pnl_per_asset"].items():
                print(f"  P&L {asset:<14}: {value:,.2f}")
            print(f"stress total P&L    : {result['total_pnl']:,.2f}")
        return 0
    except RiskToolError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
