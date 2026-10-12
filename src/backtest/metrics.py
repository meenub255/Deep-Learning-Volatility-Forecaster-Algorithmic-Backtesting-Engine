"""
Quantitative Risk & Performance Attribution Metrics for Capital Markets Backtesting.

Computes institutional tear-sheet metrics:
- Annualized CAGR & Annualized Volatility
- Sharpe Ratio, Sortino Ratio (Downside Semi-Deviation), Calmar Ratio
- Maximum Drawdown (MDD) & Underwater Drawdown Curve
- 95% Value-at-Risk (VaR) & 95% Conditional Value-at-Risk (CVaR / Expected Shortfall)
- CAPM Alpha (Annualized %), Market Beta, & Information Ratio vs. Nasdaq Benchmark
- Win Rate, Profit Factor, & Execution Cost Drag
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd


def compute_underwater_drawdown(equity_series: pd.Series) -> pd.Series:
    """Computes the continuous percentage drawdown from the running high-water mark."""
    running_max = equity_series.cummax()
    return (equity_series - running_max) / running_max.replace(0, np.nan)


def compute_performance_tearsheet(
    run_id: str,
    strategy_name: str,
    df_equity: pd.DataFrame,
    df_trades: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    risk_free_rate_annual: float = 0.04,
    benchmark_returns: Optional[ pd.Series ] = None,
) -> Dict[str, float | int | str]:
    """
    Generates a complete institutional performance tear sheet from daily equity curve and trade ledger.
    """
    eq = df_equity["equity"].astype(float)
    rets = eq.pct_change().dropna()
    n_days = max(len(rets), 1)
    years = n_days / 252.0

    final_equity = float(eq.iloc[-1])
    total_return = (final_equity / initial_capital) - 1.0
    cagr = (final_equity / initial_capital) ** (1.0 / max(years, 0.01)) - 1.0

    rf_daily = (1.0 + risk_free_rate_annual) ** (1.0 / 252.0) - 1.0
    excess_rets = rets - rf_daily

    ann_vol = float(rets.std() * np.sqrt(252.0))
    sharpe = float((excess_rets.mean() * 252.0) / (ann_vol + 1e-9))

    downside_rets = excess_rets[excess_rets < 0.0]
    downside_vol = float(np.sqrt(np.mean(downside_rets ** 2)) * np.sqrt(252.0)) if len(downside_rets) > 0 else 1e-6
    sortino = float((excess_rets.mean() * 252.0) / (downside_vol + 1e-9))

    dd_series = compute_underwater_drawdown(eq)
    max_dd = float(dd_series.min())
    calmar = float(cagr / (abs(max_dd) + 1e-9))

    # Tail Risk: 95% Daily Historical VaR & Conditional VaR (Expected Shortfall)
    var_95 = float(np.percentile(rets, 5))
    tail_losses = rets[rets <= var_95]
    cvar_95 = float(tail_losses.mean()) if len(tail_losses) > 0 else var_95

    # Win Rate & Profit Factor
    winning_days = rets[rets > 0.0]
    losing_days = rets[rets < 0.0]
    win_rate = float(len(winning_days) / n_days)
    gross_profits = float(winning_days.sum())
    gross_losses = float(abs(losing_days.sum()))
    profit_factor = float(gross_profits / (gross_losses + 1e-9))

    # CAPM Alpha, Beta & Information Ratio vs Benchmark
    alpha_ann_pct = 0.0
    beta_val = 1.0
    info_ratio = 0.0
    if benchmark_returns is not None and len(benchmark_returns) == len(rets):
        b_rets = benchmark_returns.to_numpy()
        s_rets = rets.to_numpy()
        cov_mat = np.cov(s_rets, b_rets)
        if cov_mat[1, 1] > 1e-12:
            beta_val = float(cov_mat[0, 1] / cov_mat[1, 1])
        alpha_daily = (np.mean(s_rets) - rf_daily) - beta_val * (np.mean(b_rets) - rf_daily)
        alpha_ann_pct = float(alpha_daily * 252.0 * 100.0)
        active_ret = s_rets - b_rets
        tracking_error = float(np.std(active_ret) * np.sqrt(252.0))
        if tracking_error > 1e-9:
            info_ratio = float((np.mean(active_ret) * 252.0) / tracking_error)

    total_trades = int(len(df_trades)) if not df_trades.empty else 0
    total_costs = (
        float(df_trades["commission_usd"].sum() + df_trades["slippage_usd"].sum())
        if not df_trades.empty
        else 0.0
    )

    return {
        "run_id": run_id,
        "strategy_name": strategy_name,
        "start_date": pd.to_datetime(df_equity["trade_date"].iloc[0]).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(df_equity["trade_date"].iloc[-1]).strftime("%Y-%m-%d"),
        "initial_capital": round(initial_capital, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "annualized_vol_pct": round(ann_vol * 100.0, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "var_95_pct": round(var_95 * 100.0, 2),
        "cvar_95_pct": round(cvar_95 * 100.0, 2),
        "win_rate_pct": round(win_rate * 100.0, 2),
        "profit_factor": round(profit_factor, 2),
        "alpha_annual_pct": round(alpha_ann_pct, 2),
        "beta": round(beta_val, 2),
        "information_ratio": round(info_ratio, 2),
        "total_trades": total_trades,
        "total_costs_usd": round(total_costs, 2),
    }
