"""
Institutional Quantitative Tear-Sheet Visualizer.

Generates a high-resolution 4-panel executive summary figure (`reports/tearsheet_dashboard.png`):
1. Out-of-Sample Cumulative Equity Curves ($1M Initial Capital)
2. Underwater Drawdown Profile (%)
3. PyTorch Probabilistic Volatility Forecast Ribbon (10th-90th Quantile CI vs. Realized Vol)
4. Multi-Head Temporal Self-Attention Lag Attribution Heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def generate_tearsheet_figure(
    equity_curves: Dict[str, pd.DataFrame],
    df_oos: pd.DataFrame,
    attention_map: np.ndarray | None,
    output_path: str | Path = "reports/tearsheet_dashboard.png",
) -> Path:
    """Renders and saves a 4-panel institutional quantitative tear-sheet chart."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), dpi=160)
    fig.patch.set_facecolor("#0f172a")

    text_color = "#f8fafc"
    muted_color = "#94a3b8"
    grid_color = "#1e293b"
    panel_bg = "#111827"

    for ax in axes.flat:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=muted_color, labelsize=9)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.grid(True, color=grid_color, linestyle="--", linewidth=0.7, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    palette = {
        "PyTorch Temporal-Attention Vol-Target Alpha": ("#38bdf8", 2.4, "-"),
        "GARCH(1,1) Risk-Parity Baseline": ("#a855f7", 1.8, "--"),
        "Nasdaq-100 Equal-Weight Benchmark": ("#f59e0b", 1.8, ":"),
    }

    # Panel 1: Cumulative Portfolio Equity Curve
    ax1 = axes[0, 0]
    for name, df_eq in equity_curves.items():
        color, lw, ls = palette.get(name, ("#22c55e", 1.8, "-"))
        ax1.plot(
            pd.to_datetime(df_eq["trade_date"]),
            df_eq["equity"] / 1e6,
            label=name,
            color=color,
            linewidth=lw,
            linestyle=ls,
        )
    ax1.set_title("1. Out-of-Sample Cumulative Portfolio Equity ($M, Net of Slippage)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Portfolio NAV ($ Millions)", fontsize=10)
    ax1.legend(loc="upper left", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor=text_color, fontsize=8.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel 2: Underwater Drawdown Curve
    ax2 = axes[0, 1]
    for name, df_eq in equity_curves.items():
        color, lw, ls = palette.get(name, ("#22c55e", 1.6, "-"))
        dates = pd.to_datetime(df_eq["trade_date"])
        dd = df_eq["drawdown_pct"]
        ax2.plot(dates, dd, label=name, color=color, linewidth=lw, linestyle=ls)
        if "PyTorch" in name:
            ax2.fill_between(dates, dd, 0, color=color, alpha=0.18)
    ax2.set_title("2. Underwater Drawdown Profile (% From High-Water Mark)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.legend(loc="lower left", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor=text_color, fontsize=8.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel 3: PyTorch Volatility Forecast & 10th-90th Quantile Band (QQQ Index)
    ax3 = axes[1, 0]
    qqq_slice = df_oos[df_oos["symbol"] == "QQQ"].sort_values("trade_date").tail(220)
    q_dates = pd.to_datetime(qqq_slice["trade_date"])
    ax3.fill_between(
        q_dates,
        qqq_slice["pred_vol_q10"] * 100.0,
        qqq_slice["pred_vol_q90"] * 100.0,
        color="#38bdf8",
        alpha=0.22,
        label="PyTorch 10th–90th Quantile CI Band",
    )
    ax3.plot(q_dates, qqq_slice["actual_fwd_vol"] * 100.0, color="#f43f5e", linewidth=1.7, label="Actual Forward 5d Realized Vol (%)")
    ax3.plot(q_dates, qqq_slice["pred_vol_mean"] * 100.0, color="#38bdf8", linewidth=2.0, label="PyTorch Predicted Vol (Median %)")
    ax3.plot(q_dates, qqq_slice["garch_baseline_vol"] * 100.0, color="#a855f7", linewidth=1.3, linestyle="--", label="GARCH(1,1) Baseline (%)")
    ax3.set_title("3. QQQ Out-of-Sample Volatility Forecast vs. Realized Volatility", fontsize=11, fontweight="bold", pad=10)
    ax3.set_ylabel("Annualized Volatility (%)", fontsize=10)
    ax3.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor=text_color, fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel 4: PyTorch Multi-Head Self-Attention Temporal Lag Weights / Gross Exposure
    ax4 = axes[1, 1]
    if attention_map is not None and attention_map.ndim == 2:
        im = ax4.imshow(attention_map, cmap="YlGnBu", aspect="auto")
        ax4.set_title("4. PyTorch Multi-Head Temporal Self-Attention Map (20-Day Lookback)", fontsize=11, fontweight="bold", pad=10)
        ax4.set_xlabel("Key Historical Lag Step (t-19 to t)", fontsize=10)
        ax4.set_ylabel("Query Temporal Step (t-19 to t)", fontsize=10)
        cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=muted_color)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=muted_color, fontsize=8)
    else:
        pt_eq = equity_curves["PyTorch Temporal-Attention Vol-Target Alpha"]
        ax4.plot(pd.to_datetime(pt_eq["trade_date"]), pt_eq["gross_exposure"] * 100.0, color="#22c55e", linewidth=1.8)
        ax4.set_title("4. Dynamic Portfolio Gross Exposure (% of NAV)", fontsize=11, fontweight="bold", pad=10)
        ax4.set_ylabel("Gross Exposure (%)", fontsize=10)

    fig.suptitle(
        "NASDAQ CAPITAL MARKETS — PYTORCH VOLATILITY FORECASTING & ALGORITHMIC BACKTESTING ENGINE",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_file, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return out_file
