"""Tests for regime_backtesting.backtest."""

import numpy as np
import pandas as pd
import pytest
from regime_backtesting import (
    regime_allocation, compute_cumulative_returns,
    compute_drawdowns, backtest, performance_summary,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    regime_df = pd.DataFrame({
        "prob_crash": np.random.uniform(0, 0.3, n),
        "prob_high_vol": np.random.uniform(0, 0.3, n),
        "prob_low_vol": np.random.uniform(0.4, 1.0, n),
        "regime": np.random.choice(
            ["Low Volatility", "High Volatility", "Crash"],
            n, p=[0.6, 0.25, 0.15]),
    }, index=dates)
    log_returns = pd.Series(np.random.normal(0.0003, 0.01, n), index=dates)
    return regime_df, log_returns


def test_regime_allocation_shape(sample_data):
    regime_df, _ = sample_data
    exp = regime_allocation(regime_df)
    assert len(exp) == len(regime_df)


def test_regime_allocation_bounded(sample_data):
    regime_df, _ = sample_data
    exp = regime_allocation(regime_df)
    assert (exp >= 0).all() and (exp <= 1).all()


def test_regime_allocation_first_is_default(sample_data):
    regime_df, _ = sample_data
    exp = regime_allocation(regime_df)
    assert exp.iloc[0] == 1.0  # shifted, so first day is default


def test_cumulative_returns_starts_near_one():
    rets = pd.Series([0.01, -0.005, 0.02])
    cum = compute_cumulative_returns(rets)
    assert abs(cum.iloc[0] - 1.01) < 0.001


def test_drawdowns_non_positive():
    cum = pd.Series([1.0, 1.1, 1.05, 1.2, 1.15])
    dd = compute_drawdowns(cum)
    assert (dd <= 0).all()


def test_backtest_returns_expected_columns(sample_data):
    regime_df, log_returns = sample_data
    bt = backtest(regime_df, log_returns)
    expected = {"exposure", "turnover", "strategy_ret", "benchmark_ret",
                "strategy_cum", "benchmark_cum", "strategy_dd", "benchmark_dd"}
    assert expected.issubset(set(bt.columns))


def test_performance_summary_structure(sample_data):
    regime_df, log_returns = sample_data
    bt = backtest(regime_df, log_returns)
    perf = performance_summary(bt)
    assert "Strategy" in perf.columns
    assert "Benchmark" in perf.columns
    assert "Sharpe Ratio" in perf.index
