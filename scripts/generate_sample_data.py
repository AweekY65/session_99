"""Generate deterministic sample CSV data (prices, positions, stress).

Purely code-generated synthetic data; no external data source is used.
Run: python scripts/generate_sample_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    n_days, n_assets = 504, 3
    dates = pd.bdate_range("2024-01-02", periods=n_days)

    # Correlated daily returns via a fixed covariance matrix.
    cov = np.array([
        [0.0004, 0.0002, 0.0001],
        [0.0002, 0.0009, 0.0003],
        [0.0001, 0.0003, 0.0016],
    ])
    mean = np.array([0.0005, 0.0003, 0.0001])
    daily = rng.multivariate_normal(mean, cov, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(daily, axis=0))

    df = pd.DataFrame(prices, columns=["AAA", "BBB", "CCC"])
    df.insert(0, "date", dates.strftime("%Y-%m-%d"))
    # Inject a few missing values to demonstrate the cleaning rules.
    df.loc[10, "AAA"] = np.nan
    df.loc[25, "CCC"] = np.nan
    df.to_csv(DATA_DIR / "prices.csv", index=False)

    pd.DataFrame({
        "asset": ["AAA", "BBB", "CCC"],
        "quantity": [100, 50, -30],
    }).to_csv(DATA_DIR / "positions.csv", index=False)

    pd.DataFrame({
        "asset": ["ALL", "CCC"],
        "shock": [-0.10, -0.25],
    }).to_csv(DATA_DIR / "stress.csv", index=False)

    print(f"sample data written to {DATA_DIR}")


if __name__ == "__main__":
    main()
