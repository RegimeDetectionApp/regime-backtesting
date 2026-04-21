"""
Regime-aware backtesting framework.

Implements a simple but realistic allocation strategy driven by the HMM's
posterior regime probabilities and compares it against a passive
buy-and-hold benchmark.
"""

import numpy as np
import pandas as pd


# ── Strategy logic ────────────────────────────────────────────────────────

def regime_allocation(
    regime_df: pd.DataFrame,
    crash_threshold: float = 0.50,
    high_vol_threshold: float = 0.50,
) -> pd.Series:
    """
    Determine daily equity exposure based on regime probabilities.

    Rules
    -----
    1. If prob(Crash) > crash_threshold   → 10 % exposure  (defensive)
    2. If prob(High Vol) > high_vol_threshold → 50 % exposure (reduced)
    3. Otherwise (Low Volatility dominant) → 100 % exposure (full)

    Exposure is applied to the *next* day's return to avoid look-ahead bias.

    Returns
    -------
    pd.Series
        Daily exposure weights in [0, 1], shifted forward by one day.
    """
    exposure = pd.Series(1.0, index=regime_df.index)

    crash_mask = regime_df["prob_crash"] > crash_threshold
    high_vol_mask = regime_df["prob_high_vol"] > high_vol_threshold

    # Crash takes priority over high-vol
    exposure[high_vol_mask] = 0.50
    exposure[crash_mask] = 0.10

    # Shift forward: today's signal drives tomorrow's position
    return exposure.shift(1).fillna(1.0)


# ── Performance analytics ─────────────────────────────────────────────────

def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative compounded returns (starts at 1)."""
    return (1 + returns).cumprod()


def compute_drawdowns(cumulative: pd.Series) -> pd.Series:
    """Running drawdown series from peak equity."""
    peak = cumulative.cummax()
    return (cumulative - peak) / peak


def compute_turnover(exposure: pd.Series) -> pd.Series:
    """Daily turnover — absolute change in exposure."""
    return exposure.diff().abs().fillna(0.0)


def backtest(
    regime_df: pd.DataFrame,
    log_returns: pd.Series,
    crash_threshold: float = 0.50,
    high_vol_threshold: float = 0.50,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Run the regime-aware backtest.

    Parameters
    ----------
    regime_df : pd.DataFrame
        Output of model.build_regime_df (must share index with log_returns).
    log_returns : pd.Series
        Daily log returns of the asset.
    crash_threshold / high_vol_threshold : float
        Probability thresholds for regime-based allocation.
    cost_bps : float
        Round-trip transaction cost in basis points (default: 10 bps).
        Applied proportionally to daily turnover.

    Returns
    -------
    pd.DataFrame with columns:
        exposure       — daily weight applied
        turnover       — daily absolute change in exposure
        strategy_ret   — weighted daily return (net of costs)
        benchmark_ret  — unweighted (buy-and-hold) daily return
        strategy_cum   — cumulative strategy equity curve
        benchmark_cum  — cumulative benchmark equity curve
        strategy_dd    — strategy drawdown
        benchmark_dd   — benchmark drawdown
    """
    # Align on common dates
    common = regime_df.index.intersection(log_returns.index)
    regime_df = regime_df.loc[common]
    rets = log_returns.loc[common]

    # Convert log returns to simple returns for compounding
    simple_rets = np.exp(rets) - 1

    exposure = regime_allocation(
        regime_df,
        crash_threshold=crash_threshold,
        high_vol_threshold=high_vol_threshold,
    )

    turnover = compute_turnover(exposure)
    cost_per_day = turnover * (cost_bps / 10_000)

    strategy_rets = simple_rets * exposure - cost_per_day

    result = pd.DataFrame(
        {
            "exposure": exposure,
            "turnover": turnover,
            "strategy_ret": strategy_rets,
            "benchmark_ret": simple_rets,
        },
        index=common,
    )

    result["strategy_cum"] = compute_cumulative_returns(result["strategy_ret"])
    result["benchmark_cum"] = compute_cumulative_returns(result["benchmark_ret"])
    result["strategy_dd"] = compute_drawdowns(result["strategy_cum"])
    result["benchmark_dd"] = compute_drawdowns(result["benchmark_cum"])

    return result


# ── Summary statistics ────────────────────────────────────────────────────

def performance_summary(bt: pd.DataFrame) -> pd.DataFrame:
    """
    Annualised performance metrics for strategy vs. benchmark.

    Metrics: total return, CAGR, annualised volatility, Sharpe ratio,
    Sortino ratio, Calmar ratio, CVaR (5%), maximum drawdown,
    annualised turnover.
    """
    n_days = len(bt)
    years = n_days / 252

    summaries = {}
    for label, ret_col, dd_col in [
        ("Strategy", "strategy_ret", "strategy_dd"),
        ("Benchmark", "benchmark_ret", "benchmark_dd"),
    ]:
        rets = bt[ret_col]
        total_ret = bt[ret_col.replace("ret", "cum")].iloc[-1] - 1
        cagr = (1 + total_ret) ** (1 / years) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = cagr / ann_vol if ann_vol > 0 else np.nan
        max_dd = bt[dd_col].min()

        # Sortino: penalise only downside deviation
        downside_returns = np.minimum(rets, 0)
        downside_std = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
        sortino = cagr / downside_std if downside_std > 0 else np.nan

        # Calmar: CAGR relative to worst drawdown
        calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

        # CVaR 5%: expected shortfall (mean of worst 5% of daily returns)
        cutoff = int(max(1, len(rets) * 0.05))
        worst = rets.nsmallest(cutoff)
        cvar_5 = worst.mean()

        entry = {
            "Total Return": f"{total_ret:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Calmar Ratio": f"{calmar:.2f}",
            "CVaR (5%)": f"{cvar_5:.2%}",
            "Max Drawdown": f"{max_dd:.2%}",
        }

        # Turnover only applies to strategy
        if label == "Strategy" and "turnover" in bt.columns:
            ann_turnover = bt["turnover"].sum() / years
            entry["Ann. Turnover"] = f"{ann_turnover:.1f}x"

        summaries[label] = entry

    return pd.DataFrame(summaries)
