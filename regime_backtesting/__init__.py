"""Strategy backtesting and performance analytics for regime detection."""

from regime_backtesting.backtest import (
    backtest,
    performance_summary,
    regime_allocation,
    compute_cumulative_returns,
    compute_drawdowns,
    compute_turnover,
)

__all__ = [
    "backtest",
    "performance_summary",
    "regime_allocation",
    "compute_cumulative_returns",
    "compute_drawdowns",
    "compute_turnover",
]
