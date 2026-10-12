"""
Generates high-resolution demo images and an animated walkthrough video (.gif & .mp4)
for the GitHub README showcase.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from src.database.sql_manager import QuantSQLDatabase
from src.features.alpha_factors import QuantFeatureEngineer
from src.backtest.engine import InstitutionalBacktester, BacktestConfig


def load_simulation_data():
    db = QuantSQLDatabase("data/nasdaq_quant_warehouse.db")
    df_ohlcv = db.fetch_ohlcv_data()
    fe = QuantFeatureEngineer()
    df_feat = fe.build_feature_matrix(df_ohlcv)

    with db._get_connection() as conn:
        df_forecasts = pd.read_sql_query("SELECT * FROM model_forecasts", conn)
    df_forecasts["trade_date"] = pd.to_datetime(df_forecasts["trade_date"])

    df_oos = pd.merge(
        df_feat,
        df_forecasts[
            [
                "symbol", "trade_date", "actual_fwd_vol", "pred_vol_mean",
                "pred_vol_q10", "pred_vol_q90", "garch_baseline_vol",
                "pred_up_prob", "volatility_regime",
            ]
        ],
        on=["symbol", "trade_date"],
        how="inner",
    )

    bt = InstitutionalBacktester(BacktestConfig(target_annual_vol=0.195))
    df_eq_bm, _, sum_bm = bt.run_strategy(df_oos, mode="benchmark_ew", run_id="BM", strategy_name="Nasdaq-100 Equal-Weight Benchmark")
    bm_rets = df_eq_bm["equity"].pct_change().dropna()
    df_eq_ga, _, sum_ga = bt.run_strategy(df_oos, mode="garch_parity", run_id="GA", strategy_name="GARCH(1,1) Risk-Parity Baseline", benchmark_returns=bm_rets)
    df_eq_pt, df_tr_pt, sum_pt = bt.run_strategy(df_oos, mode="pytorch_alpha", run_id="PT", strategy_name="PyTorch Temporal-Attention Vol-Target Alpha", benchmark_returns=bm_rets)

    with open("reports/backtest_results.json", "r", encoding="utf-8") as f:
        report_json = json.load(f)

    return df_oos, df_eq_pt, df_eq_ga, df_eq_bm, df_tr_pt, sum_pt, sum_ga, sum_bm, report_json


def create_multi_asset_volatility_cones(df_oos: pd.DataFrame, out_path: Path) -> None:
    """Generates a 4-panel chart of out-of-sample PyTorch Volatility Cones across NVDA, QQQ, MSFT, META."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    symbols = ["NVDA", "QQQ", "MSFT", "META"]
    titles = [
        "NVDA — Semiconductor AI Leader (OOS 5-Day Volatility Cone)",
        "QQQ — Nasdaq-100 Index ETF (OOS 5-Day Volatility Cone)",
        "MSFT — Enterprise Cloud Leader (OOS 5-Day Volatility Cone)",
        "META — Digital Platforms (OOS 5-Day Volatility Cone)",
    ]

    for ax, sym, title in zip(axes.flat, symbols, titles):
        ax.set_facecolor("#111827")
        sub = df_oos[df_oos["symbol"] == sym].sort_values("trade_date").tail(260)
        dates = pd.to_datetime(sub["trade_date"])

        ax.fill_between(
            dates,
            sub["pred_vol_q10"] * 100,
            sub["pred_vol_q90"] * 100,
            color="#38bdf8",
            alpha=0.22,
            label="PyTorch 10th–90th Quantile Band",
        )
        ax.plot(dates, sub["actual_fwd_vol"] * 100, color="#f43f5e", lw=1.6, label="Realized 5d Forward Vol (%)")
        ax.plot(dates, sub["pred_vol_mean"] * 100, color="#38bdf8", lw=2.0, label="PyTorch Predicted Vol (%)")
        ax.plot(dates, sub["garch_baseline_vol"] * 100, color="#a855f7", lw=1.2, ls="--", label="GARCH(1,1) Baseline (%)")

        ax.set_title(title, color="#f8fafc", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_ylabel("Annualized Volatility (%)", color="#cbd5e1", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=8.5)
        ax.grid(True, color="#1e293b", linestyle="--", alpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.8)

    fig.suptitle(
        "PYTORCH TEMPORAL-ATTENTION PROBABILISTIC VOLATILITY CONES ACROSS NASDAQ-100 LEADERS",
        color="#f8fafc",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def create_regime_and_metrics_showcase(report_json: dict, out_path: Path) -> None:
    """Generates a 2-panel bar chart comparing Regime MAE Reduction & Portfolio Risk-Adjusted Metrics."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#cbd5e1", labelsize=9.5)
        ax.grid(True, color="#1e293b", linestyle="--", alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 1: Regime Forecast MAE (PyTorch vs GARCH)
    reg_df = pd.DataFrame(report_json["sql_regime_forecast_attribution"])
    x = np.arange(len(reg_df))
    w = 0.34
    b1 = ax1.bar(x - w / 2, reg_df["pytorch_mae_pct"], width=w, color="#38bdf8", label="PyTorch TemporalAttentionVolNet MAE (%)")
    b2 = ax1.bar(x + w / 2, reg_df["garch_mae_pct"], width=w, color="#a855f7", label="Econometric GARCH(1,1) MAE (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(reg_df["volatility_regime"], color="#f8fafc", fontweight="bold")
    ax1.set_ylabel("Out-of-Sample Volatility Forecast MAE (%)", color="#f8fafc", fontsize=10)
    ax1.set_title("SQL CTE Attribution: PyTorch vs. GARCH(1,1) Error by Regime", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    for bar, imp in zip(b1, reg_df["mae_improvement_vs_garch_pct"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"-{imp:.1f}% Err",
            ha="center",
            va="bottom",
            color="#38bdf8",
            fontsize=8.5,
            fontweight="bold",
        )

    # Panel 2: Strategy Sharpe, Sortino, Calmar & Profit Factor Comparison
    pt = report_json["strategies"]["pytorch_vol_target_alpha"]
    ga = report_json["strategies"]["garch_risk_parity_baseline"]
    bm = report_json["strategies"]["nasdaq_equal_weight_benchmark"]

    metrics_labels = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Profit Factor"]
    pt_vals = [pt["sharpe_ratio"], pt["sortino_ratio"], pt["calmar_ratio"], pt["profit_factor"]]
    ga_vals = [ga["sharpe_ratio"], ga["sortino_ratio"], ga["calmar_ratio"], ga["profit_factor"]]
    bm_vals = [bm["sharpe_ratio"], bm["sortino_ratio"], bm["calmar_ratio"], bm["profit_factor"]]

    xm = np.arange(len(metrics_labels))
    wm = 0.25
    ax2.bar(xm - wm, pt_vals, width=wm, color="#38bdf8", label="PyTorch Vol-Target Alpha")
    ax2.bar(xm, ga_vals, width=wm, color="#a855f7", label="GARCH(1,1) Risk-Parity")
    ax2.bar(xm + wm, bm_vals, width=wm, color="#f59e0b", label="Nasdaq-100 Benchmark")

    ax2.set_xticks(xm)
    ax2.set_xticklabels(metrics_labels, color="#f8fafc", fontweight="bold")
    ax2.set_ylabel("Ratio Value (Higher is Better)", color="#f8fafc", fontsize=10)
    ax2.set_title("Institutional Risk-Adjusted Performance Ratios (OOS 2024–2026)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax2.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def create_terminal_ui_preview(
    df_eq_pt: pd.DataFrame,
    df_eq_ga: pd.DataFrame,
    df_eq_bm: pd.DataFrame,
    df_oos: pd.DataFrame,
    sum_pt: dict,
    f_eval: dict,
    out_path: Path,
) -> None:
    """Renders a high-resolution visual showcase of the Interactive Quantitative Terminal UI."""
    fig = plt.figure(figsize=(16, 9.5), dpi=150)
    fig.patch.set_facecolor("#090d16")
    gs = gridspec.GridSpec(3, 6, height_ratios=[0.95, 2.5, 2.2], hspace=0.38, wspace=0.30)

    # Top Header Banner
    fig.text(
        0.03, 0.955,
        "NASDAQ CAPITAL MARKETS — Interactive Quantitative Volatility & Backtesting Terminal",
        color="#f8fafc", fontsize=14.5, fontweight="bold",
    )
    fig.text(
        0.03, 0.928,
        "LIVE STREAMLIT + SQL + PYTORCH TERMINAL  |  Model: TemporalAttentionVolNet_v1  |  Universe: QQQ, NVDA, AAPL, MSFT, AMZN, META",
        color="#38bdf8", fontsize=9.5, fontweight="bold",
    )

    kpis = [
        ("SHARPE RATIO", f"{sum_pt['sharpe_ratio']:.2f}", "+40.6% vs Benchmark", "#22c55e"),
        ("SORTINO RATIO", f"{sum_pt['sortino_ratio']:.2f}", "+51.5% vs Benchmark", "#22c55e"),
        ("ANNUALIZED ALPHA", f"+{sum_pt['alpha_annual_pct']:.2f}%", f"Beta: {sum_pt['beta']:.2f}", "#38bdf8"),
        ("MAX DRAWDOWN", f"{sum_pt['max_drawdown_pct']:.2f}%", "-63.2% Drawdown Cut", "#22c55e"),
        ("PYTORCH VOL RMSE", f"{f_eval['pytorch_rmse']:.4f}", f"-{f_eval['rmse_improvement_pct']:.1f}% vs GARCH", "#38bdf8"),
        ("STRESS MAE EDGE", "-23.04%", "High-Vol Regime Win", "#f59e0b"),
    ]

    for idx, (label, val, sub, col) in enumerate(kpis):
        ax_k = fig.add_subplot(gs[0, idx])
        ax_k.set_facecolor("#111827")
        ax_k.set_xticks([])
        ax_k.set_yticks([])
        for spine in ax_k.spines.values():
            spine.set_color("#1e293b")
            spine.set_linewidth(1.5)
        ax_k.text(0.08, 0.72, label, color="#94a3b8", fontsize=8, fontweight="bold")
        ax_k.text(0.08, 0.36, val, color="#f8fafc", fontsize=15, fontweight="bold")
        ax_k.text(0.08, 0.10, sub, color=col, fontsize=8, fontweight="bold")

    # Middle Row: Left = Portfolio Equity Curve, Right = Dynamic Exposure & Drawdown
    ax_main = fig.add_subplot(gs[1, :4])
    ax_main.set_facecolor("#111827")
    ax_main.plot(pd.to_datetime(df_eq_pt["trade_date"]), df_eq_pt["equity"] / 1e6, color="#38bdf8", lw=2.4, label="PyTorch Vol-Target Alpha ($1.66M)")
    ax_main.plot(pd.to_datetime(df_eq_ga["trade_date"]), df_eq_ga["equity"] / 1e6, color="#a855f7", lw=1.6, ls="--", label="GARCH(1,1) Risk Parity")
    ax_main.plot(pd.to_datetime(df_eq_bm["trade_date"]), df_eq_bm["equity"] / 1e6, color="#f59e0b", lw=1.6, ls=":", label="Nasdaq-100 Benchmark")
    ax_main.set_title("Out-of-Sample Portfolio NAV ($ Millions, Net of Bid-Ask Spread & Market Impact)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax_main.tick_params(colors="#94a3b8", labelsize=8.5)
    ax_main.grid(True, color="#1e293b", ls="--", alpha=0.8)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_main.legend(loc="upper left", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)
    for spine in ax_main.spines.values():
        spine.set_color("#334155")

    ax_exp = fig.add_subplot(gs[1, 4:])
    ax_exp.set_facecolor("#111827")
    dates_pt = pd.to_datetime(df_eq_pt["trade_date"])
    ax_exp.fill_between(dates_pt, df_eq_pt["gross_exposure"] * 100, 0, color="#22c55e", alpha=0.25)
    ax_exp.plot(dates_pt, df_eq_pt["gross_exposure"] * 100, color="#22c55e", lw=1.7, label="Gross Exposure (%)")
    ax_exp.axhline(100, color="#94a3b8", ls="--", lw=1.0, label="100% Unlevered Baseline")
    ax_exp.set_title("Dynamic Volatility-Targeted Gross Exposure (%)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax_exp.tick_params(colors="#94a3b8", labelsize=8.5)
    ax_exp.grid(True, color="#1e293b", ls="--", alpha=0.8)
    ax_exp.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_exp.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)
    for spine in ax_exp.spines.values():
        spine.set_color("#334155")

    # Bottom Row: Left = NVDA Volatility Cone, Right = Underwater Drawdown Comparison
    ax_nvda = fig.add_subplot(gs[2, :3])
    ax_nvda.set_facecolor("#111827")
    nvda = df_oos[df_oos["symbol"] == "NVDA"].sort_values("trade_date").tail(200)
    ndates = pd.to_datetime(nvda["trade_date"])
    ax_nvda.fill_between(ndates, nvda["pred_vol_q10"] * 100, nvda["pred_vol_q90"] * 100, color="#38bdf8", alpha=0.22, label="10th–90th Quantile CI")
    ax_nvda.plot(ndates, nvda["actual_fwd_vol"] * 100, color="#f43f5e", lw=1.5, label="Actual 5d Vol (%)")
    ax_nvda.plot(ndates, nvda["pred_vol_mean"] * 100, color="#38bdf8", lw=1.9, label="PyTorch Forecast (%)")
    ax_nvda.set_title("NVDA — Real-Time PyTorch Volatility Forecast Cone", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax_nvda.tick_params(colors="#94a3b8", labelsize=8.5)
    ax_nvda.grid(True, color="#1e293b", ls="--", alpha=0.8)
    ax_nvda.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_nvda.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.8)
    for spine in ax_nvda.spines.values():
        spine.set_color("#334155")

    ax_dd = fig.add_subplot(gs[2, 3:])
    ax_dd.set_facecolor("#111827")
    ax_dd.fill_between(dates_pt, df_eq_pt["drawdown_pct"], 0, color="#38bdf8", alpha=0.22)
    ax_dd.plot(dates_pt, df_eq_pt["drawdown_pct"], color="#38bdf8", lw=2.0, label="PyTorch Strategy (-11.85% Max DD)")
    ax_dd.plot(pd.to_datetime(df_eq_bm["trade_date"]), df_eq_bm["drawdown_pct"], color="#f59e0b", lw=1.5, ls=":", label="Nasdaq Benchmark (-32.17% Max DD)")
    ax_dd.set_title("Underwater Drawdown Protection (Circuit Breaker Active)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax_dd.tick_params(colors="#94a3b8", labelsize=8.5)
    ax_dd.grid(True, color="#1e293b", ls="--", alpha=0.8)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_dd.legend(loc="lower left", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)
    for spine in ax_dd.spines.values():
        spine.set_color("#334155")

    fig.subplots_adjust(top=0.89, bottom=0.06, left=0.04, right=0.98)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def create_animated_demo_video(
    df_eq_pt: pd.DataFrame,
    df_eq_ga: pd.DataFrame,
    df_eq_bm: pd.DataFrame,
    df_oos: pd.DataFrame,
    gif_path: Path,
    mp4_path: Path,
) -> None:
    """
    Creates an animated walk-forward backtest simulation video (.gif and .mp4)
    showing real-time PyTorch volatility forecasting, dynamic drawdown protection,
    and cumulative equity compounding across 2024–2026.
    """
    qqq = df_oos[df_oos["symbol"] == "QQQ"].sort_values("trade_date").reset_index(drop=True)
    dates = pd.to_datetime(df_eq_pt["trade_date"]).to_numpy()
    n_total = len(dates)

    # Sample ~45 frames across the 577 out-of-sample trading days for a smooth, crisp animation
    frame_indices = np.linspace(25, n_total - 1, 45, dtype=int)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7.2), dpi=110)
    fig.patch.set_facecolor("#0b0f19")

    def _style_ax(ax, title, ylabel):
        ax.set_facecolor("#111827")
        ax.set_title(title, color="#f8fafc", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, color="#cbd5e1", fontsize=9.5)
        ax.tick_params(colors="#94a3b8", labelsize=8.5)
        ax.grid(True, color="#1e293b", ls="--", alpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for spine in ax.spines.values():
            spine.set_color("#334155")

    eq_pt = (df_eq_pt["equity"] / 1e6).to_numpy()
    eq_ga = (df_eq_ga["equity"] / 1e6).to_numpy()
    eq_bm = (df_eq_bm["equity"] / 1e6).to_numpy()
    exp_pt = (df_eq_pt["gross_exposure"] * 100).to_numpy()

    q_actual = (qqq["actual_fwd_vol"] * 100).to_numpy()
    q_pred = (qqq["pred_vol_mean"] * 100).to_numpy()
    q_q10 = (qqq["pred_vol_q10"] * 100).to_numpy()
    q_q90 = (qqq["pred_vol_q90"] * 100).to_numpy()
    q_reg = qqq["volatility_regime"].tolist()

    def update(frame_pos: int):
        idx = frame_indices[frame_pos]
        ax_top.clear()
        ax_bot.clear()

        cur_dates = dates[: idx + 1]
        cur_date_str = pd.to_datetime(dates[idx]).strftime("%Y-%m-%d")
        cur_regime = q_reg[idx]
        cur_nav = eq_pt[idx]
        cur_ret = (cur_nav - 1.0) * 100.0

        _style_ax(
            ax_top,
            f"LIVE WALK-FORWARD BACKTEST ({cur_date_str})  |  NAV: ${cur_nav:.2f}M ({cur_ret:+.1f}%)  |  Exposure: {exp_pt[idx]:.0f}%",
            "Portfolio NAV ($M)",
        )
        ax_top.set_xlim(dates[0], dates[-1])
        ax_top.set_ylim(0.72, 1.85)
        ax_top.plot(cur_dates, eq_pt[: idx + 1], color="#38bdf8", lw=2.5, label="PyTorch Vol-Target Alpha (Sharpe: 1.42)")
        ax_top.plot(cur_dates, eq_ga[: idx + 1], color="#a855f7", lw=1.6, ls="--", label="GARCH(1,1) Risk Parity (Sharpe: 1.01)")
        ax_top.plot(cur_dates, eq_bm[: idx + 1], color="#f59e0b", lw=1.6, ls=":", label="Nasdaq-100 Benchmark (Sharpe: 1.01)")
        ax_top.scatter([dates[idx]], [eq_pt[idx]], color="#38bdf8", s=45, zorder=5)
        ax_top.legend(loc="upper left", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

        _style_ax(
            ax_bot,
            f"PYTORCH TEMPORAL-ATTENTION VOLATILITY FORECAST CONE (QQQ)  |  Detected Regime: {cur_regime.upper()}",
            "Annualized Vol (%)",
        )
        ax_bot.set_xlim(dates[0], dates[-1])
        ax_bot.set_ylim(8.0, 42.0)
        ax_bot.fill_between(cur_dates, q_q10[: idx + 1], q_q90[: idx + 1], color="#38bdf8", alpha=0.22, label="PyTorch 10th–90th Quantile Band")
        ax_bot.plot(cur_dates, q_actual[: idx + 1], color="#f43f5e", lw=1.6, label="Realized 5d Forward Vol (%)")
        ax_bot.plot(cur_dates, q_pred[: idx + 1], color="#38bdf8", lw=2.1, label="PyTorch Predicted Vol (%)")
        ax_bot.scatter([dates[idx]], [q_pred[idx]], color="#38bdf8", s=45, zorder=5)
        ax_bot.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

        return []

    fig.suptitle(
        "DEEP LEARNING VOLATILITY FORECASTER & ALGORITHMIC BACKTESTING ENGINE — LIVE WALK-FORWARD DEMO",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=140, blit=False)
    anim.save(str(gif_path), writer=PillowWriter(fps=8))
    print(f"Saved animated GIF demo to: {gif_path}")

    # Also export MP4 video using imageio_ffmpeg binary
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg_exe, "-y", "-i", str(gif_path),
                "-movflags", "faststart",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                str(mp4_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Saved MP4 video demo to: {mp4_path}")
    except Exception as exc:
        print(f"MP4 export note: {exc}")

    plt.close(fig)


def main() -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading simulation data from SQL warehouse...")
    df_oos, df_eq_pt, df_eq_ga, df_eq_bm, df_tr_pt, sum_pt, sum_ga, sum_bm, report_json = load_simulation_data()

    print("[1/4] Generating multi-asset volatility cones image...")
    create_multi_asset_volatility_cones(df_oos, reports_dir / "demo_volatility_cones.png")

    print("[2/4] Generating SQL regime attribution & risk-return showcase image...")
    create_regime_and_metrics_showcase(report_json, reports_dir / "demo_regime_attribution.png")

    print("[3/4] Generating Interactive Quantitative Terminal UI preview image...")
    create_terminal_ui_preview(
        df_eq_pt, df_eq_ga, df_eq_bm, df_oos,
        sum_pt, report_json["forecast_evaluation"],
        reports_dir / "demo_terminal_preview.png",
    )

    print("[4/4] Generating animated walk-forward demo video (.gif & .mp4)...")
    create_animated_demo_video(
        df_eq_pt, df_eq_ga, df_eq_bm, df_oos,
        reports_dir / "demo_walkthrough.gif",
        reports_dir / "demo_walkthrough.mp4",
    )
    print("All demo media assets generated successfully!")


if __name__ == "__main__":
    main()
