"""Generate sample local CSV data (prices + holdings) with a fixed seed.

Usage: python scripts/generate_sample_data.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(out_dir: str = "data") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20240101)
    n = 500
    mean = [0.0003, 0.0001, -0.0001]
    cov = [
        [0.0004, 0.00015, 0.00005],
        [0.00015, 0.0009, 0.00010],
        [0.00005, 0.00010, 0.0002],
    ]
    rets = rng.multivariate_normal(mean, cov, size=n)
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    dates = pd.bdate_range("2023-01-02", periods=n)
    df = pd.DataFrame(prices, columns=["ALPHA", "BETA", "GAMMA"], index=dates)
    df.to_csv(out / "prices.csv", index_label="date")
    (out / "holdings.csv").write_text(
        "asset,quantity\nALPHA,120\nBETA,40\nGAMMA,-30\n"
    )
    print(f"wrote {out/'prices.csv'} and {out/'holdings.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
