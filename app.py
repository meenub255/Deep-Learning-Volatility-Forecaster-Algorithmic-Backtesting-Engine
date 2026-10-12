"""
Interactive Quantitative Research & Backtesting Terminal (Streamlit + Plotly + SQL).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.database.sql_manager import QuantSQLDatabase
from src.backtest.engine import InstitutionalBacktester, BacktestConfig
from src.features.alpha_factors import QuantFeatureEngineer


st.set_page_config(
    page_title="Nasdaq AI/ML Volatility & Backtesting Terminal",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Nasdaq Capital Markets — PyTorch Volatility Forecaster & Backtesting Engine")
st.caption(
    "Institutional Time-Series Volatility Forecasting (`TemporalAttentionVolNet` in PyTorch) "
    "coupled with a Relational SQL Feature Store & Microstructure-Aware Backtesting Framework."
)

db = QuantSQLDatabase("data/nasdaq_quant_warehouse.db")
report_path = Path("reports/backtest_results.json")

if not report_path.exists():
    st.warning("Pipeline artifacts not found. Please run `python run_pipeline.py` first.")
    st.stop()

with open(report_path, "r", encoding="utf-8") as f:
    report_data = json.load(f)

f_eval = report_data["forecast_evaluation"]
pt_strat = report_data["strategies"]["pytorch_vol_target_alpha"]
ga_strat = report_data["strategies"]["garch_risk_parity_baseline"]
bm_strat = report_data["strategies"]["nasdaq_equal_weight_benchmark"]

# Top KPI Ribbon
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric(
    "Strategy Sharpe Ratio",
    f"{pt_strat['sharpe_ratio']:.2f}",
    f"{pt_strat['sharpe_ratio'] - bm_strat['sharpe_ratio']:+.2f} vs Benchmark",
)
col2.metric(
    "Sortino Ratio",
    f"{pt_strat['sortino_ratio']:.2f}",
    f"{pt_strat['sortino_ratio'] - bm_strat['sortino_ratio']:+.2f} vs Benchmark",
)
col3.metric(
    "Annualized Return (CAGR)",
    f"{pt_strat['cagr_pct']:.2f}%",
    f"Alpha: {pt_strat['alpha_annual_pct']:+.2f}%",
)
col4.metric(
    "Max Drawdown",
    f"{pt_strat['max_drawdown_pct']:.2f}%",
    f"{pt_strat['max_drawdown_pct'] - bm_strat['max_drawdown_pct']:+.2f}% vs Benchmark",
)
col5.metric(
    "PyTorch Vol RMSE Edge",
    f"{f_eval['pytorch_rmse']:.4f}",
    f"-{f_eval['rmse_improvement_pct']:.2f}% vs GARCH(1,1)",
)
col6.metric(
    "80% Quantile CI Coverage",
    f"{f_eval['quantile_80_coverage_pct']:.1f}%",
    f"Dir Acc: {f_eval['directional_auc_acc_pct']:.1f}%",
)

tab1, tab2, tab3 = st.tabs([
    "📊 Strategy Backtest & Stress Simulator",
    "🧠 PyTorch Volatility & Quantile Forecasts",
    "🗄️ Relational SQL Warehouse & CTE Analytics",
])

with tab1:
    st.subheader("Interactive Execution Cost & Volatility Target Stress Test")
    c_s1, c_s2, c_s3 = st.columns(3)
    target_vol = c_s1.slider("Target Portfolio Annualized Volatility (%)", 10.0, 28.0, 16.5, 0.5) / 100.0
    comm_bps = c_s2.slider("Exchange Commission (bps)", 0.0, 10.0, 1.0, 0.5)
    impact_coeff = c_s3.slider("Almgren-Chriss Market Impact Coefficient", 0.01, 0.30, 0.08, 0.01)

    @st.cache_data
    def load_cached_simulation_data() -> pd.DataFrame:
        with db._get_connection() as conn:
            df_fc = pd.read_sql_query("SELECT * FROM model_forecasts", conn)
        df_fc["trade_date"] = pd.to_datetime(df_fc["trade_date"])

        df_bars = db.fetch_ohlcv_data()
        feature_eng = QuantFeatureEngineer()
        df_ft = feature_eng.build_feature_matrix(df_bars)
        return pd.merge(
            df_ft,
            df_fc[
                [
                    "symbol", "trade_date", "actual_fwd_vol", "pred_vol_mean",
                    "pred_vol_q10", "pred_vol_q90", "garch_baseline_vol",
                    "pred_up_prob", "volatility_regime",
                ]
            ],
            on=["symbol", "trade_date"],
            how="inner",
        )

    df_merged = load_cached_simulation_data()

    sim_bt = InstitutionalBacktester(
        BacktestConfig(
            target_annual_vol=target_vol,
            commission_bps=comm_bps,
            impact_coeff=impact_coeff,
        )
    )
    df_eq_bm, _, sum_bm_sim = sim_bt.run_strategy(df_merged, mode="benchmark_ew", run_id="SIM_BM", strategy_name="Nasdaq Benchmark")
    bm_rets = df_eq_bm["equity"].pct_change().dropna()
    df_eq_ga, _, sum_ga_sim = sim_bt.run_strategy(df_merged, mode="garch_parity", run_id="SIM_GA", strategy_name="GARCH(1,1) Risk Parity", benchmark_returns=bm_rets)
    df_eq_pt, df_tr_pt, sum_pt_sim = sim_bt.run_strategy(df_merged, mode="pytorch_alpha", run_id="SIM_PT", strategy_name="PyTorch Vol-Target Alpha", benchmark_returns=bm_rets)

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=df_eq_pt["trade_date"], y=df_eq_pt["equity"], name="PyTorch Vol-Target Alpha", line=dict(color="#38bdf8", width=2.5)))
    fig_eq.add_trace(go.Scatter(x=df_eq_ga["trade_date"], y=df_eq_ga["equity"], name="GARCH(1,1) Risk Parity", line=dict(color="#a855f7", width=1.8, dash="dash")))
    fig_eq.add_trace(go.Scatter(x=df_eq_bm["trade_date"], y=df_eq_bm["equity"], name="Nasdaq Equal-Weight Benchmark", line=dict(color="#f59e0b", width=1.8, dash="dot")))
    fig_eq.update_layout(
        title="Out-of-Sample Cumulative Portfolio Equity ($1,000,000 Initial Capital)",
        xaxis_title="Trade Date",
        yaxis_title="Portfolio Equity (USD)",
        template="plotly_dark",
        height=440,
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown("#### Multi-Strategy Quantitative Tear-Sheet Comparison")
    st.dataframe(pd.DataFrame([sum_pt_sim, sum_ga_sim, sum_bm_sim]), use_container_width=True)

with tab2:
    st.subheader("Out-of-Sample PyTorch Probabilistic Volatility Forecast Explorer")
    selected_sym = st.selectbox("Select Nasdaq Instrument", ["QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "META"])
    sym_df = df_merged[df_merged["symbol"] == selected_sym].sort_values("trade_date")

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=sym_df["trade_date"], y=sym_df["pred_vol_q90"] * 100,
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig_vol.add_trace(go.Scatter(
        x=sym_df["trade_date"], y=sym_df["pred_vol_q10"] * 100,
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(56, 189, 248, 0.22)", name="10th–90th Quantile CI Band",
    ))
    fig_vol.add_trace(go.Scatter(
        x=sym_df["trade_date"], y=sym_df["actual_fwd_vol"] * 100,
        name="Actual 5-Day Forward Realized Vol (%)", line=dict(color="#f43f5e", width=1.8),
    ))
    fig_vol.add_trace(go.Scatter(
        x=sym_df["trade_date"], y=sym_df["pred_vol_mean"] * 100,
        name="PyTorch Predicted Vol (%)", line=dict(color="#38bdf8", width=2.2),
    ))
    fig_vol.add_trace(go.Scatter(
        x=sym_df["trade_date"], y=sym_df["garch_baseline_vol"] * 100,
        name="GARCH(1,1) Baseline (%)", line=dict(color="#a855f7", width=1.4, dash="dash"),
    ))
    fig_vol.update_layout(
        title=f"{selected_sym} — Out-of-Sample Annualized Volatility Forecast vs. Realized Volatility",
        yaxis_title="Annualized Volatility (%)",
        template="plotly_dark",
        height=440,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with tab3:
    st.subheader("SQL Window-Function Market Analytics & Regime Attribution")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("**1. SQL CTE Regime-Conditional Forecast Error Attribution**")
        st.dataframe(pd.DataFrame(report_data["sql_regime_forecast_attribution"]), use_container_width=True)
    with col_q2:
        st.markdown("**2. SQL Window-Function Cross-Sectional Nasdaq Analytics**")
        st.dataframe(pd.DataFrame(report_data["sql_window_market_analytics"]), use_container_width=True)

    st.markdown("**3. Granular SQL Execution Trade Ledger (Latest 50 Rebalance Fills)**")
    st.dataframe(df_tr_pt.tail(50), use_container_width=True)
