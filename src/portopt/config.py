"""
config.py
----------
Configuration dataclass for portfolio optimization project.
Stores user-adjustable parameters like tickers, date range, and risk-free rate.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """
    Configuration container for portfolio optimization.

    Attributes
    ----------
    tickers : list[str]
        Asset tickers to include in the portfolio (e.g., ["AAPL", "MSFT", "TLT", "GLD"])
    start : str
        Start date for historical data download (format: 'YYYY-MM-DD')
    end : str or None
        End date for historical data; None = today
    risk_free : float
        Annualized risk-free rate (e.g., 0.02 for 2%)
    price_col : str
        Column to use from downloaded data (usually 'Adj Close')
    return_type : str
        'log' for log returns or 'simple' for percentage returns
    trading_days : int
        Number of trading days per year (default: 252)
    long_only : bool
        If True, disallow short positions (w_i ≥ 0)
    """

    tickers: List[str]
    start: Optional[str] = "2015-01-01"
    end: Optional[str] = None
    risk_free: float = 0.02
    price_col: str = "Adj Close"
    return_type: str = "log"
    trading_days: int = 252
    long_only: bool = True

    def __post_init__(self):
        # Default tickers if none are provided
        if not self.tickers:
            self.tickers = ["AAPL", "MSFT", "TLT", "GLD"]
