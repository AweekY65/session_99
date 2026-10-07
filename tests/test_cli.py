"""CLI smoke test: run the terminal entry point on generated CSV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_tool.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(3)
    n = 260
    rets = rng.multivariate_normal(
        [0.0, 0.0], [[0.0004, 0.0001], [0.0001, 0.0002]], size=n)
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    pdf = pd.DataFrame(prices, columns=["AAA", "BBB"],
                       index=pd.date_range("2024-01-01", periods=n))
    prices_csv = tmp_path / "prices.csv"
    holdings_csv = tmp_path / "holdings.csv"
    pdf.to_csv(prices_csv, index_label="date")
    holdings_csv.write_text("asset,quantity\nAAA,10\nBBB,5\n")

    rc = main(["--prices", str(prices_csv), "--holdings", str(holdings_csv),
               "--confidence", "0.99", "--window", "250", "--seed", "42",
               "--shock", "*=-0.1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Historical VaR" in out
    assert "Monte Carlo ES" in out
    assert "Stress P&L" in out
    print("\n" + out)


def test_cli_reports_error_on_bad_confidence(tmp_path, capsys):
    prices_csv = tmp_path / "prices.csv"
    holdings_csv = tmp_path / "holdings.csv"
    pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}).to_csv(prices_csv,
                                                     index_label="date")
    holdings_csv.write_text("asset,quantity\nA,1\n")
    rc = main(["--prices", str(prices_csv), "--holdings", str(holdings_csv),
               "--confidence", "1.5"])
    assert rc == 2
    assert "confidence" in capsys.readouterr().err
