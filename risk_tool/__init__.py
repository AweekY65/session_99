"""Local portfolio market risk computation tool.

All inputs come from local CSV files or code-generated data. No network,
exchange, market-data API, database, or external service is ever accessed.
"""

from .errors import (
    CovarianceError,
    InsufficientDataError,
    InvalidConfidenceLevelError,
    MissingDataError,
    RiskToolError,
    ZeroValuePortfolioError,
)
from .portfolio import Portfolio
from .returns import compute_returns
from .historical import historical_var, historical_es
from .parametric import parametric_var, portfolio_volatility
from .montecarlo import monte_carlo_risk
from .stress import apply_stress, StressScenario

__all__ = [
    "CovarianceError",
    "InsufficientDataError",
    "InvalidConfidenceLevelError",
    "MissingDataError",
    "RiskToolError",
    "ZeroValuePortfolioError",
    "Portfolio",
    "compute_returns",
    "historical_var",
    "historical_es",
    "parametric_var",
    "portfolio_volatility",
    "monte_carlo_risk",
    "apply_stress",
    "StressScenario",
]
