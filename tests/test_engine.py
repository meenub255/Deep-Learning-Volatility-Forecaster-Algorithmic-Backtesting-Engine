"""
Automated Pytest Suite for the Nasdaq PyTorch Volatility Forecaster & Backtesting Engine.

Verifies:
1. Relational SQL schema creation, OHLCV seeding, and CTE/Window-Function queries.
2. Range-based volatility estimators (Parkinson, Garman-Klass, Yang-Zhang) mathematical bounds.
3. PyTorch TemporalAttentionVolNet tensor shapes, monotonic quantile ordering (q10 <= q50 <= q90),
   and QuantileQLIKELoss gradient backpropagation.
4. Institutional backtester execution, slippage deduction, and tear-sheet risk metrics.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.database.sql_manager import QuantSQLDatabase
from src.features.alpha_factors import QuantFeatureEngineer, FEATURE_COLUMNS
from src.models.volatility_forecaster import TemporalAttentionVolNet, QuantileQLIKELoss
from src.backtest.metrics import compute_performance_tearsheet, compute_underwater_drawdown


def test_sql_warehouse_and_window_queries(tmp_path: Path) -> None:
    db_file = tmp_path / "test_quant.db"
    db = QuantSQLDatabase(db_file)
    df_ohlcv = db.seed_nasdaq_market_data(start_date="2023-01-02", end_date="2023-06-30", force_refresh=True)
    assert not df_ohlcv.empty
    assert set(df_ohlcv["symbol"].unique()) == {"QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "META"}

    df_win = db.query_sql_window_market_analytics()
    assert len(df_win) == 6
    assert "max_historical_drawdown_pct" in df_win.columns


def test_range_volatility_estimators_non_negative(tmp_path: Path) -> None:
    db = QuantSQLDatabase(tmp_path / "test_feat.db")
    df_ohlcv = db.seed_nasdaq_market_data(start_date="2023-01-02", end_date="2023-08-30", force_refresh=True)
    fe = QuantFeatureEngineer()
    df_feat = fe.build_feature_matrix(df_ohlcv)

    assert not df_feat.empty
    for col in ["realized_vol_21d", "parkinson_vol_21d", "garman_klass_vol_21d", "yang_zhang_vol_21d"]:
        assert (df_feat[col] > 0.0).all(), f"{col} must be strictly positive"


def test_pytorch_temporal_attention_volnet_forward_and_backward() -> None:
    batch_size, seq_len, n_features = 16, 20, len(FEATURE_COLUMNS)
    model = TemporalAttentionVolNet(num_features=n_features, hidden_dim=32, num_layers=2, num_heads=4)
    criterion = QuantileQLIKELoss()

    x = torch.randn(batch_size, seq_len, n_features)
    y_vol = torch.rand(batch_size) * 0.25 + 0.10
    y_dir = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

    vol_q, dir_logits, attn_weights = model(x)
    assert vol_q.shape == (batch_size, 3)
    assert dir_logits.shape == (batch_size, 1)
    assert attn_weights.shape == (batch_size, seq_len, seq_len)

    # Verify monotonic quantile ordering: q10 <= q50 <= q90
    q10, q50, q90 = vol_q[:, 0], vol_q[:, 1], vol_q[:, 2]
    assert torch.all(q10 <= q50 + 1e-6)
    assert torch.all(q50 <= q90 + 1e-6)

    loss = criterion(vol_q, y_vol, dir_logits, y_dir)
    loss.backward()
    assert loss.item() > -10.0


def test_performance_tearsheet_metrics_accuracy() -> None:
    dates = pd.bdate_range("2025-01-01", periods=100)
    # Create deterministic upward equity curve with a small dip
    nav = np.linspace(1_000_000, 1_180_000, 100)
    nav[40:45] *= 0.96
    df_eq = pd.DataFrame({"trade_date": dates, "equity": nav})
    df_trades = pd.DataFrame({
        "commission_usd": [12.5, 15.0],
        "slippage_usd": [45.0, 52.0],
    })

    dd = compute_underwater_drawdown(df_eq["equity"])
    assert dd.min() < -0.03

    summary = compute_performance_tearsheet(
        run_id="UNIT_TEST_RUN",
        strategy_name="Unit Test Strategy",
        df_equity=df_eq,
        df_trades=df_trades,
    )
    assert summary["total_return_pct"] == 18.0
    assert summary["sharpe_ratio"] > 1.0
    assert summary["max_drawdown_pct"] < -3.0
    assert summary["total_costs_usd"] == 124.5


def test_garch_baseline_properties() -> None:
    from src.models.volatility_forecaster import compute_garch_baseline
    # Generate mock log returns
    returns = np.random.normal(0.0005, 0.015, 300)
    df_asset = pd.DataFrame({"log_return": returns})
    garch_vol = compute_garch_baseline(df_asset)
    assert len(garch_vol) == 300
    assert (garch_vol > 0.0).all()
    # Annualized volatility should be in a plausible equity range (~10% to 50%)
    assert 0.10 <= np.mean(garch_vol) <= 0.60


def test_microstructure_execution_costs_attribution(tmp_path: Path) -> None:
    from src.backtest.engine import InstitutionalBacktester, BacktestConfig
    db = QuantSQLDatabase(tmp_path / "test_exec.db")
    df_ohlcv = db.seed_nasdaq_market_data(start_date="2024-01-02", end_date="2024-03-31", force_refresh=True)
    fe = QuantFeatureEngineer()
    df_feat = fe.build_feature_matrix(df_ohlcv)
    # Add mock forecast fields for backtest
    df_feat["pred_vol_mean"] = df_feat["realized_vol_21d"]
    df_feat["pred_vol_q90"] = df_feat["realized_vol_21d"] * 1.3
    df_feat["pred_up_prob"] = 0.55
    df_feat["garch_baseline_vol"] = df_feat["realized_vol_21d"]
    df_feat["actual_fwd_vol"] = df_feat["realized_vol_21d"]
    df_feat["volatility_regime"] = "Normal Rotational"

    config = BacktestConfig(
        initial_capital=100_000.0,
        commission_bps=2.0,
        impact_coeff=0.10,
    )
    bt = InstitutionalBacktester(config=config)
    df_eq, df_trades, summary = bt.run_strategy(df_feat, mode="pytorch_alpha", run_id="TEST_RUN")
    assert not df_eq.empty
    assert not df_trades.empty
    assert summary["total_trades"] > 0
    assert summary["total_costs_usd"] > 0.0
    # Both commission and slippage must be strictly positive
    assert (df_trades["commission_usd"] > 0.0).all()
    assert (df_trades["slippage_usd"] > 0.0).all()

