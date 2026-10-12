-- ============================================================================
-- Quantitative Backtesting & Volatility Forecasting Engine - Relational Schema
-- Designed for Capital Markets Analytics (SQLite / PostgreSQL Compatible)
-- ============================================================================

PRAGMA foreign_keys = ON;

-- 1. Exchange Traded Instruments Metadata
CREATE TABLE IF NOT EXISTS instruments (
    symbol           VARCHAR(16) PRIMARY KEY,
    company_name     VARCHAR(128) NOT NULL,
    exchange         VARCHAR(32) DEFAULT 'NASDAQ',
    sector           VARCHAR(64) NOT NULL,
    market_cap_tier  VARCHAR(32) DEFAULT 'Mega-Cap',
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Historical Daily OHLCV Market Data
CREATE TABLE IF NOT EXISTS ohlcv_bars (
    symbol           VARCHAR(16) NOT NULL,
    trade_date       DATE NOT NULL,
    open_price        DOUBLE PRECISION NOT NULL,
    high_price       DOUBLE PRECISION NOT NULL,
    low_price        DOUBLE PRECISION NOT NULL,
    close_price      DOUBLE PRECISION NOT NULL,
    adj_close        DOUBLE PRECISION NOT NULL,
    volume           BIGINT NOT NULL,
    vwap             DOUBLE PRECISION NOT NULL,
    spread_bps       DOUBLE PRECISION DEFAULT 1.5,
    PRIMARY KEY (symbol, trade_date),
    FOREIGN KEY (symbol) REFERENCES instruments(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv_bars(trade_date);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date ON ohlcv_bars(symbol, trade_date);

-- 3. Quantitative Feature Store (Engineered Alpha & Volatility Estimators)
CREATE TABLE IF NOT EXISTS feature_store (
    symbol                  VARCHAR(16) NOT NULL,
    trade_date              DATE NOT NULL,
    log_return              DOUBLE PRECISION,
    realized_vol_5d         DOUBLE PRECISION,
    realized_vol_21d        DOUBLE PRECISION,
    parkinson_vol_21d       DOUBLE PRECISION,
    garman_klass_vol_21d    DOUBLE PRECISION,
    yang_zhang_vol_21d      DOUBLE PRECISION,
    rsi_14                  DOUBLE PRECISION,
    macd_hist               DOUBLE PRECISION,
    bb_width                DOUBLE PRECISION,
    atr_pct                 DOUBLE PRECISION,
    amihud_illiquidity      DOUBLE PRECISION,
    volume_zscore           DOUBLE PRECISION,
    return_skew_21d         DOUBLE PRECISION,
    target_fwd_vol_5d       DOUBLE PRECISION,
    target_fwd_ret_5d       DOUBLE PRECISION,
    PRIMARY KEY (symbol, trade_date),
    FOREIGN KEY (symbol) REFERENCES instruments(symbol) ON DELETE CASCADE
);

-- 4. PyTorch AI/ML Volatility & Regime Forecasts
CREATE TABLE IF NOT EXISTS model_forecasts (
    forecast_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version           VARCHAR(64) NOT NULL,
    symbol                  VARCHAR(16) NOT NULL,
    trade_date              DATE NOT NULL,
    actual_fwd_vol          DOUBLE PRECISION,
    pred_vol_mean           DOUBLE PRECISION NOT NULL,
    pred_vol_q10            DOUBLE PRECISION NOT NULL,
    pred_vol_q90            DOUBLE PRECISION NOT NULL,
    garch_baseline_vol      DOUBLE PRECISION,
    pred_up_prob            DOUBLE PRECISION NOT NULL,
    volatility_regime       VARCHAR(32) NOT NULL,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_version, symbol, trade_date),
    FOREIGN KEY (symbol) REFERENCES instruments(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_forecasts_lookup ON model_forecasts(symbol, trade_date);

-- 5. Backtest Strategy Runs Summary
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id                  VARCHAR(64) PRIMARY KEY,
    strategy_name           VARCHAR(128) NOT NULL,
    start_date              DATE NOT NULL,
    end_date                DATE NOT NULL,
    initial_capital         DOUBLE PRECISION NOT NULL,
    final_equity            DOUBLE PRECISION NOT NULL,
    total_return_pct        DOUBLE PRECISION NOT NULL,
    cagr_pct                DOUBLE PRECISION NOT NULL,
    annualized_vol_pct      DOUBLE PRECISION NOT NULL,
    sharpe_ratio            DOUBLE PRECISION NOT NULL,
    sortino_ratio           DOUBLE PRECISION NOT NULL,
    calmar_ratio            DOUBLE PRECISION NOT NULL,
    max_drawdown_pct        DOUBLE PRECISION NOT NULL,
    var_95_pct              DOUBLE PRECISION NOT NULL,
    cvar_95_pct             DOUBLE PRECISION NOT NULL,
    win_rate_pct            DOUBLE PRECISION NOT NULL,
    profit_factor           DOUBLE PRECISION NOT NULL,
    total_trades            INTEGER NOT NULL,
    total_costs_usd         DOUBLE PRECISION NOT NULL,
    executed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Execution Trade Ledger & Position Attribution
CREATE TABLE IF NOT EXISTS trade_ledger (
    trade_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  VARCHAR(64) NOT NULL,
    trade_date              DATE NOT NULL,
    symbol                  VARCHAR(16) NOT NULL,
    side                    VARCHAR(8) NOT NULL,
    target_weight           DOUBLE PRECISION NOT NULL,
    shares_traded           DOUBLE PRECISION NOT NULL,
    reference_price         DOUBLE PRECISION NOT NULL,
    execution_price         DOUBLE PRECISION NOT NULL,
    notional_usd            DOUBLE PRECISION NOT NULL,
    commission_usd          DOUBLE PRECISION NOT NULL,
    slippage_usd            DOUBLE PRECISION NOT NULL,
    pred_vol_signal         DOUBLE PRECISION,
    regime_label            VARCHAR(32),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (symbol) REFERENCES instruments(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ledger_run_date ON trade_ledger(run_id, trade_date);
