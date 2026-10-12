"""
End-to-End Execution Pipeline:
1. SQL Data Warehouse Initialization & Multi-Regime Nasdaq-100 OHLCV Ingestion
2. Quantitative Feature Engineering (Parkinson, Garman-Klass, Yang-Zhang Volatility + Microstructure)
3. PyTorch TemporalAttentionVolNet Training & Out-of-Sample Walk-Forward Forecasting
4. Multi-Strategy Institutional Backtesting with Slippage & Market Impact Attribution
5. SQL Window-Function & CTE Analytics + Tear-Sheet Visualization Export
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from src.database.sql_manager import QuantSQLDatabase
from src.features.alpha_factors import QuantFeatureEngineer, FEATURE_COLUMNS
from src.models.volatility_forecaster import VolatilityForecastTrainer
from src.backtest.engine import InstitutionalBacktester, BacktestConfig
from src.backtest.visualizer import generate_tearsheet_figure


def main() -> None:
    print("=" * 88)
    print("NASDAQ CAPITAL MARKETS AI/ML VOLATILITY FORECASTER & BACKTESTING ENGINE")
    print("=" * 88)

    # 1. Seed / Load Relational SQL Database
    db = QuantSQLDatabase("data/nasdaq_quant_warehouse.db")
    print("[1/5] Seeding & querying relational SQL OHLCV warehouse (2019-01-02 to 2026-03-31)...")
    df_ohlcv = db.seed_nasdaq_market_data(force_refresh=False)
    print(f"      Loaded {len(df_ohlcv):,} daily OHLCV bars across {df_ohlcv['symbol'].nunique()} Nasdaq instruments.")

    # 2. Engineer Quantitative Features & Persist to SQL Feature Store
    print("[2/5] Computing range-based volatility estimators (Parkinson, Garman-Klass, Yang-Zhang) & alpha factors...")
    fe = QuantFeatureEngineer(annualization_factor=252, forecast_horizon=5)
    df_features = fe.build_feature_matrix(df_ohlcv)
    db.save_engineered_features(df_features)
    print(f"      Persisted {len(df_features):,} feature vectors (16 causal factors) to SQL `feature_store`.")

    # 3. Train PyTorch TemporalAttentionVolNet via Purged Walk-Forward Validation
    print("[3/5] Training PyTorch TemporalAttentionVolNet (LSTM + Multi-Head Self-Attention + Quantile/QLIKE Loss)...")
    trainer = VolatilityForecastTrainer(
        feature_cols=FEATURE_COLUMNS,
        seq_len=20,
        embargo_days=5,
        hidden_dim=64,
        epochs=12,
        batch_size=256,
        lr=2.2e-3,
        seed=42,
    )
    df_oos, forecast_metrics = trainer.fit_and_forecast_walk_forward(
        df_features=df_features,
        train_cutoff_date="2023-12-31",
    )
    db.save_model_forecasts(df_oos, model_version="TemporalAttentionVolNet_v1")
    print(
        f"      Out-of-Sample Forecast Results ({len(df_oos):,} OOS asset-days):\n"
        f"      - PyTorch Volatility RMSE : {forecast_metrics.pytorch_rmse:.4f} vs. GARCH(1,1) RMSE: {forecast_metrics.garch_rmse:.4f} "
        f"(+{forecast_metrics.rmse_improvement_pct:.2f}% Improvement)\n"
        f"      - PyTorch Volatility MAE  : {forecast_metrics.pytorch_mae:.4f} vs. GARCH(1,1) MAE : {forecast_metrics.garch_mae:.4f} "
        f"(+{forecast_metrics.mae_improvement_pct:.2f}% Improvement)\n"
        f"      - 10th-90th Quantile CI Coverage: {forecast_metrics.quantile_80_coverage_pct:.2f}% | "
        f"Directional Regime Accuracy: {forecast_metrics.directional_auc_acc_pct:.2f}%"
    )

    # 4. Execute Multi-Strategy Institutional Backtest
    print("[4/5] Executing multi-asset backtesting engine with bid-ask spread & square-root market impact...")
    bt_config = BacktestConfig(
        initial_capital=1_000_000.0,
        target_annual_vol=0.195,
        max_leverage=1.35,
        commission_bps=1.0,
        impact_coeff=0.08,
        rebalance_every_n_days=5,
        min_trade_weight_delta=0.025,
        risk_free_rate_annual=0.04,
    )
    backtester = InstitutionalBacktester(config=bt_config)

    # Run Benchmark first to compute active CAPM Alpha, Beta, and Information Ratio
    df_eq_bench, df_tr_bench, sum_bench = backtester.run_strategy(
        df_oos=df_oos,
        mode="benchmark_ew",
        run_id="NASDAQ_EW_BENCHMARK",
        strategy_name="Nasdaq-100 Equal-Weight Benchmark",
    )
    bench_rets = df_eq_bench["equity"].pct_change().dropna()
    db.save_backtest_run(sum_bench, df_tr_bench)

    # Run GARCH(1,1) Risk-Parity Baseline
    df_eq_garch, df_tr_garch, sum_garch = backtester.run_strategy(
        df_oos=df_oos,
        mode="garch_parity",
        run_id="GARCH_RISK_PARITY",
        strategy_name="GARCH(1,1) Risk-Parity Baseline",
        benchmark_returns=bench_rets,
    )
    db.save_backtest_run(sum_garch, df_tr_garch)

    # Run PyTorch Temporal-Attention Vol-Target Alpha Strategy
    df_eq_pt, df_tr_pt, sum_pt = backtester.run_strategy(
        df_oos=df_oos,
        mode="pytorch_alpha",
        run_id="PYTORCH_VOL_ALPHA",
        strategy_name="PyTorch Temporal-Attention Vol-Target Alpha",
        benchmark_returns=bench_rets,
    )
    db.save_backtest_run(sum_pt, df_tr_pt)

    # 5. Execute SQL Window-Function Analytics & Export Reports
    print("[5/5] Running SQL CTE & Window-Function analytics and generating tear-sheet plots...")
    df_sql_market = db.query_sql_window_market_analytics()
    df_sql_regime = db.query_regime_forecast_attribution()

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_sql_market.to_csv(reports_dir / "sql_window_market_analytics.csv", index=False)
    df_sql_regime.to_csv(reports_dir / "sql_regime_forecast_attribution.csv", index=False)

    equity_curves = {
        "PyTorch Temporal-Attention Vol-Target Alpha": df_eq_pt,
        "GARCH(1,1) Risk-Parity Baseline": df_eq_garch,
        "Nasdaq-100 Equal-Weight Benchmark": df_eq_bench,
    }
    chart_path = generate_tearsheet_figure(
        equity_curves=equity_curves,
        df_oos=df_oos,
        attention_map=trainer.last_attention_map,
        output_path=reports_dir / "tearsheet_dashboard.png",
    )

    full_report = {
        "forecast_evaluation": asdict(forecast_metrics),
        "strategies": {
            "pytorch_vol_target_alpha": sum_pt,
            "garch_risk_parity_baseline": sum_garch,
            "nasdaq_equal_weight_benchmark": sum_bench,
        },
        "sql_window_market_analytics": df_sql_market.to_dict(orient="records"),
        "sql_regime_forecast_attribution": df_sql_regime.to_dict(orient="records"),
    }
    with open(reports_dir / "backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n--- STRATEGY PERFORMANCE TEAR SHEET (OUT-OF-SAMPLE 2024–2026) ---")
    df_summary_table = pd.DataFrame([sum_pt, sum_garch, sum_bench])[
        [
            "strategy_name",
            "total_return_pct",
            "cagr_pct",
            "annualized_vol_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "calmar_ratio",
            "alpha_annual_pct",
            "beta",
            "win_rate_pct",
            "profit_factor",
            "total_trades",
        ]
    ]
    print(df_summary_table.to_string(index=False))

    print("\n--- SQL CTE REGIME FORECAST ATTRIBUTION ---")
    print(df_sql_regime.to_string(index=False))
    print(f"\nTear-sheet visual saved to: {chart_path.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
