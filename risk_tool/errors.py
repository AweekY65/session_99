"""Domain errors for the risk tool."""


class RiskToolError(Exception):
    """Base class for all risk tool errors."""


class DataError(RiskToolError):
    """Raised when input data (prices / holdings) is invalid."""


class InsufficientDataError(RiskToolError):
    """Raised when there are not enough observations for an estimate."""


class CovarianceError(RiskToolError):
    """Raised when a covariance matrix is invalid (not symmetric / not PSD)."""


class ZeroValueError(RiskToolError):
    """Raised when the portfolio market value is zero."""


class InvalidConfidenceError(RiskToolError):
    """Raised when a confidence level is outside the open interval (0, 1)."""
