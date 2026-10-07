"""Domain-specific exceptions for the risk tool."""


class RiskToolError(Exception):
    """Base class for all risk tool errors."""


class InsufficientDataError(RiskToolError):
    """Raised when a price/return series is too short for the requested window."""


class CovarianceError(RiskToolError):
    """Raised when a covariance matrix is invalid (asymmetric or not PSD)."""


class ZeroValuePortfolioError(RiskToolError):
    """Raised when the portfolio has zero total absolute market value."""


class InvalidConfidenceLevelError(RiskToolError):
    """Raised when a confidence level is outside the open interval (0, 1)."""


class MissingDataError(RiskToolError):
    """Raised when required data columns or assets are missing."""
