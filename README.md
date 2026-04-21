# regime-backtesting

Regime-aware backtesting framework with transaction costs and performance analytics.

## Bounded Context

Converts regime probabilities into allocation signals, simulates strategy execution with transaction costs, and calculates performance metrics against a buy-and-hold benchmark.

## Installation

```bash
pip install git+https://github.com/govid13427742/regime-backtesting.git@main
```

## API

| Function | Description |
|----------|-------------|
| `backtest(regime_df, log_returns, ...)` | Run regime-aware backtest with costs |
| `performance_summary(bt)` | Annualized metrics: Sharpe, Sortino, Calmar, CVaR, etc. |

## Input Contract

`regime_df` must have columns: `prob_crash`, `prob_high_vol`, `prob_low_vol`

## Allocation Rules

| Condition | Exposure |
|-----------|----------|
| P(Crash) > threshold | 10% |
| P(High Vol) > threshold | 50% |
| Otherwise (Low Vol) | 100% |

Signals are lagged by 1 day to avoid look-ahead bias.

## Dependencies

- numpy, pandas
