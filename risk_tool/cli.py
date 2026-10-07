"""Terminal CLI: computes and prints all risk metrics. No charts, no network.

Example:
    python -m risk_tool.cli --prices data/prices.csv --holdings data/holdings.csv \
        --confidence 0.99 --window 250 --seed 42
"""

from __future__ import annotations

import argparse
import sys

from .errors import RiskToolError
from .io import load_holdings_csv, load_prices_csv
from .returns import compute_returns
from .risk import (
    historical_es,
    historical_var,
    monte_carlo_var,
    parametric_var,
)
from .stress import apply_stress


def parse_shock(items: list[str]) -> dict[str, float]:
    """Parse ``ASSET=SHOCK`` pairs, e.g. ``AAPL=-0.1`` or ``*=-0.2``."""
    shocks: dict[str, float] = {}
    for item in items:
        try:
            asset, value = item.split("=", 1)
            shocks[asset.strip()] = float(value)
        except ValueError:
            raise RiskToolError(f"invalid --shock entry {item!r}; use ASSET=VALUE")
    return shocks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="risk_tool",
        description="Local-only portfolio market risk calculator "
        "(historical / parametric / Monte Carlo VaR & ES, stress tests).",
    )
    p.add_argument("--prices", required=True, help="CSV of historical prices")
    p.add_argument("--holdings", required=True, help="CSV of asset,quantity")
    p.add_argument("--confidence", type=float, default=0.99,
                   help="confidence level in (0,1), default 0.99")
    p.add_argument("--window", type=int, default=None,
                   help="observation window (most recent N returns)")
    p.add_argument("--returns", choices=["simple", "log"], default="simple",
                   help="return type, default simple")
    p.add_argument("--missing", choices=["drop", "ffill"], default="drop",
                   help="missing price rule, default drop")
    p.add_argument("--simulations", type=int, default=10_000,
                   help="Monte Carlo scenarios, default 10000")
    p.add_argument("--seed", type=int, default=None,
                   help="Monte Carlo random seed for reproducibility")
    p.add_argument("--shock", action="append", default=[], metavar="ASSET=PCT",
                   help="stress shock, e.g. --shock AAPL=-0.1 --shock '*= -0.05'")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prices = load_prices_csv(args.prices)
        portfolio = load_holdings_csv(args.holdings)
        rets = compute_returns(prices, method=args.returns, missing=args.missing)
        value = portfolio.current_value(prices)
        weights = portfolio.weights(prices)
        port_rets = portfolio.portfolio_returns(rets, prices)

        windowed = rets.iloc[-args.window:] if args.window else rets
        mean = windowed[portfolio.assets].mean().to_numpy()
        cov = windowed[portfolio.assets].cov().to_numpy()

        print(f"Portfolio value      : {value:,.2f}")
        print(f"Assets               : {', '.join(portfolio.assets)}")
        print(f"Observations (window): {len(windowed)}"
              f" (of {len(rets)} available)")
        print(f"Confidence level     : {args.confidence:.2%}")
        print("-" * 52)
        h_var = historical_var(port_rets, args.confidence, args.window, value)
        h_es = historical_es(port_rets, args.confidence, args.window, value)
        print(f"Historical VaR       : {h_var:,.2f}")
        print(f"Historical ES        : {h_es:,.2f}")
        p_var = parametric_var(weights, cov, args.confidence, value, mean)
        print(f"Parametric VaR       : {p_var:,.2f}")
        mc = monte_carlo_var(weights, mean, cov, args.confidence, value,
                             n_simulations=args.simulations, seed=args.seed)
        print(f"Monte Carlo VaR      : {mc['var']:,.2f}"
              f"  (n={int(mc['simulations'])}, seed={args.seed})")
        print(f"Monte Carlo ES       : {mc['es']:,.2f}")

        if args.shock:
            shocks = parse_shock(args.shock)
            result = apply_stress(portfolio, prices, shocks)
            print("-" * 52)
            print(f"Stress scenario      : {shocks}")
            print(f"Stress P&L           : {result['pnl']:,.2f}"
                  f"  ({result['return']:+.2%})")
            for asset, pnl in result["pnl_per_asset"].items():
                print(f"  {asset:<20}: {pnl:,.2f}")
        return 0
    except RiskToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
