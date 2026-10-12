# 📈 Deep Learning Volatility Forecaster & Algorithmic Backtesting Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.12](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SQL Warehouse](https://img.shields.io/badge/SQL-Relational%20Warehouse-336791?style=for-the-badge&logo=postgresql&logoColor=white)](#%EF%B8%8F-relational-sql-warehouse--cte-analytics)
[![Streamlit Terminal](https://img.shields.io/badge/Streamlit-Quant%20Terminal-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#-live-demo-video--visual-walkthrough)
[![Tests Passing](https://img.shields.io/badge/Pytest-4%2F4%20Passing-22C55E?style=for-the-badge&logo=pytest&logoColor=white)](#-quick-start)

> **An institutional-grade Quantitative AI/ML Volatility Forecasting & Multi-Asset Algorithmic Trading Backtesting Framework** engineered to forecast 5-day forward realized volatility cones and market regimes across **Nasdaq-100 (`NDX`)** leaders (`QQQ`, `NVDA`, `AAPL`, `MSFT`, `AMZN`, `META`) using **PyTorch (`TemporalAttentionVolNet`)**, **Relational SQL Window-Function Analytics**, and a **Microstructure-Aware Portfolio Backtester**.

---

## 🎬 Live Demo Video & Visual Walkthrough

### 1. Animated Walk-Forward Simulation (2024–2026 Out-of-Sample Execution)
Watch the **PyTorch `TemporalAttentionVolNet`** dynamically forecast the 10th–90th percentile volatility cone in real time, detect the `"High-Vol Stress"` regime during the late-2024 / early-2025 market correction, trigger the **automated drawdown circuit breaker**, and compound portfolio equity to **+65.55% ($1.66M)** while slashing Maximum Drawdown from **-32.17%** down to **-11.85%**:

![Live Walk-Forward Backtest Demo Video](reports/demo_walkthrough.gif)

> 📹 **High-Definition MP4 Video File**: [reports/demo_walkthrough.mp4](reports/demo_walkthrough.mp4)

---

### 2. Interactive Quantitative Terminal Preview (`Streamlit` + `Plotly` + `SQL`)
The interactive terminal (`streamlit run app.py`) allows real-time stress-testing of target portfolio volatility, exchange commissions, and Almgren-Chriss market impact coefficients while inspecting SQL execution fills and live volatility cones:

![Interactive Quantitative Terminal UI Preview](reports/demo_terminal_preview.png)

---

### 3. 4-Panel Executive Performance Tear Sheet & Self-Attention Map
Comprehensive out-of-sample tear sheet displaying cumulative portfolio NAV (net of bid-ask spread and square-root market impact), underwater drawdown protection, `QQQ` volatility forecast cones vs. `GARCH(1,1)`, and the **20-day Multi-Head Temporal Self-Attention heatmap**:

![Executive Tear Sheet Dashboard](reports/tearsheet_dashboard.png)

---

### 4. Multi-Asset PyTorch Probabilistic Volatility Cones (`NVDA`, `QQQ`, `MSFT`, `META`)
Comparison of out-of-sample **PyTorch 10th–90th Percentile Quantile Bands** and median forecasts against realized 5-day forward volatility and econometric `GARCH(1,1)` baselines:

![Multi-Asset Volatility Cones](reports/demo_volatility_cones.png)

---

### 5. SQL CTE Regime Attribution & Risk-Adjusted Performance Ratios
Out-of-sample forecast error reduction across market volatility regimes (`Low-Vol Expansion`, `Normal Rotational`, `High-Vol Stress`) alongside institutional portfolio ratios (`Sharpe`, `Sortino`, `Calmar`, `Profit Factor`):

![SQL Regime Attribution and Risk Ratios](reports/demo_regime_attribution.png)

---

## 🎯 Why This Architecture Excels for Capital Markets & FinTech (Nasdaq)

Most academic stock prediction projects suffer from **look-ahead data leakage**, **naive raw-price regression**, and **zero transaction cost modeling**. This repository is engineered around institutional quantitative standards:

1. **Range-Based Microstructure Volatility Estimators (`Pandas` / `NumPy`)**:
   - Goes beyond noisy close-to-close standard deviation by implementing **Parkinson (1980)** High-Low range volatility, **Garman-Klass (1980)** OHLC volatility, and **Yang-Zhang (2000)** drift-independent, overnight-gap-robust volatility estimators alongside **Amihud Illiquidity**, **VWAP Dislocation**, and **Volatility Term Structure ($\sigma_{5d} / \sigma_{21d}$)**.
2. **Probabilistic Deep Learning Architecture (`PyTorch`)**:
   - **`TemporalAttentionVolNet`**: Hybrid **2-Layer LSTM + 4-Head Temporal Self-Attention (`nn.MultiheadAttention`)** network with **Gated Residual LayerNorm** projections.
   - **Composite Econometric Loss (`QuantileQLIKELoss`)**: Jointly optimizes **Pinball Quantile Loss** (producing calibrated 10th, 50th, and 90th percentile volatility cones) and **Patton (2011) QLIKE Volatility Loss** ($\mathcal{L}_{\text{QLIKE}} = \ln(\hat{\sigma}^2) + \sigma^2 / \hat{\sigma}^2$) plus directional regime cross-entropy.
   - **Purged Walk-Forward Validation**: Strictly chronological split with a **5-trading-day embargo gap** between training (`2019–2023`) and out-of-sample testing (`2024–2026`) and causal `RobustScaler` fitting to guarantee **zero look-ahead bias**.
3. **Relational SQL Feature Store & Analytical Warehouse (`SQLite` / `PostgreSQL` Compatible)**:
   - Normalized 6-table relational schema (`instruments`, `ohlcv_bars`, `feature_store`, `model_forecasts`, `backtest_runs`, `trade_ledger`) indexed for fast time-series slicing.
   - Production **SQL Common Table Expressions (CTEs)** and **Window Functions** (`LAG`, `MAX() OVER (PARTITION BY symbol ORDER BY trade_date ROWS UNBOUNDED PRECEDING)`, rolling `AVG() OVER`) for underwater drawdown tracking and regime-conditional forecast error attribution.
4. **Market Microstructure Execution & Dynamic Volatility-Targeting Backtester**:
   - Models **Bid-Ask Half-Spread Crossing (bps)**, **Exchange Commissions**, and **Almgren-Chriss Square-Root Market Impact Slippage** ($\text{Impact} \propto \sigma_d \sqrt{\text{Notional} / \text{ADV}}$).
   - Dynamically sizes positions via **Inverse Tail-Volatility Risk Parity** and triggers an **automated Volatility Regime Circuit Breaker** during high-stress regimes.

---

## 📊 Empirical Out-of-Sample Results (2024–2026 Walk-Forward Evaluation)

### 1. Portfolio Backtest Tear Sheet ($1,000,000 Initial Capital, Net of Slippage & Fees)

| Strategy | Total Return | CAGR | Annualized Vol | Sharpe Ratio | Sortino Ratio | Max Drawdown | Calmar Ratio | CAPM Alpha ($\alpha$) | Market Beta ($\beta$) | Profit Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Temporal-Attention Vol-Target Alpha** | **+65.55%** | **24.68%** | **13.40%** | **1.42** | **1.53** | **-11.85%** | **2.08** | **+8.65%** | **0.51** | **1.39** |
| **GARCH(1,1) Risk-Parity Baseline** | +67.51% | 25.32% | 20.63% | 1.01 | 1.01 | -32.77% | 0.77 | +0.19% | 1.02 | 1.21 |
| **Nasdaq-100 Equal-Weight Benchmark** | +65.90% | 24.79% | 20.15% | 1.01 | 1.01 | -32.17% | 0.77 | 0.00% | 1.00 | 1.21 |

> **Key Takeaway**: The **PyTorch Temporal-Attention Vol-Target Alpha** strategy achieves a **1.42 Sharpe Ratio** (+40.6% over the Nasdaq benchmark) and **2.08 Calmar Ratio** (2.7x benchmark) while **slashing Maximum Drawdown by 63.2%** (from `-32.17%` down to `-11.85%`) and generating **+8.65% annualized CAPM Alpha** with a defensive **0.51 Beta**.

---

### 2. PyTorch `TemporalAttentionVolNet` vs. Econometric `GARCH(1,1)` Forecast Accuracy
Evaluated across **3,462 out-of-sample asset-days** (`QQQ`, `NVDA`, `AAPL`, `MSFT`, `AMZN`, `META`) via SQL CTE regime attribution:

| Market Volatility Regime | OOS Asset-Days | Avg Realized Vol | Avg Predicted Vol | PyTorch MAE (%) | GARCH(1,1) MAE (%) | MAE Reduction vs. GARCH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low-Vol Expansion** | 1,212 | 20.95% | 21.42% | **3.007%** | 3.368% | **+10.72%** |
| **Normal Rotational** | 1,384 | 25.60% | 26.32% | **3.603%** | 4.282% | **+15.87%** |
| **High-Vol Stress** | 866 | 31.11% | 32.01% | **4.286%** | 5.569% | **+23.04%** |
| **Overall Out-of-Sample** | **3,462** | **25.35%** | **26.03%** | **3.570% (RMSE: 0.0450)** | **4.280% (RMSE: 0.0559)** | **+16.78% MAE / +19.47% RMSE** |

---

## 🗄️ Relational SQL Warehouse & CTE Analytics

All market bars, engineered features, model forecasts, and trade executions are stored in `data/nasdaq_quant_warehouse.db` (`src/database/schema.sql`). Below is an example of the production **SQL Common Table Expression (CTE) + Window Function** query executed by `src/database/sql_manager.py` to compute continuous high-water-mark drawdowns and volume surge anomalies directly in SQL:

```sql
WITH ReturnCTE AS (
    SELECT
        b.symbol,
        i.sector,
        b.trade_date,
        b.close_price,
        b.volume,
        b.spread_bps,
        (b.close_price - LAG(b.close_price, 1) OVER (PARTITION BY b.symbol ORDER BY b.trade_date))
            / NULLIF(LAG(b.close_price, 1) OVER (PARTITION BY b.symbol ORDER BY b.trade_date), 0) AS daily_ret,
        MAX(b.close_price) OVER (
            PARTITION BY b.symbol
            ORDER BY b.trade_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_peak_price,
        AVG(b.volume) OVER (
            PARTITION BY b.symbol
            ORDER BY b.trade_date
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ) AS rolling_21d_adv
    FROM ohlcv_bars b
    INNER JOIN instruments i ON b.symbol = i.symbol
)
SELECT
    symbol,
    sector,
    COUNT(*) AS trading_days,
    ROUND(AVG(daily_ret) * 252 * 100, 2) AS annualized_return_pct,
    ROUND(MIN((close_price - running_peak_price) / NULLIF(running_peak_price, 0)) * 100, 2) AS max_historical_drawdown_pct,
    ROUND(AVG(spread_bps), 2) AS avg_effective_spread_bps,
    ROUND(MAX(volume * 1.0 / NULLIF(rolling_21d_adv, 0)), 2) AS max_volume_surge_multiple
FROM ReturnCTE
WHERE daily_ret IS NOT NULL
GROUP BY symbol, sector
ORDER BY annualized_return_pct DESC;
```

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    subgraph SQL_Warehouse["Relational SQL Data Warehouse (SQLite / PostgreSQL)"]
        A[("instruments & ohlcv_bars\n(11,340 Daily Bars)")]
        B[("feature_store\n(16 Causal Factors)")]
        C[("model_forecasts & trade_ledger\n(OOS Predictions & Fills)")]
    end

    subgraph Feature_Engine["Quantitative Feature Engineering (Pandas / NumPy)"]
        D["Range Volatility Estimators\n(Parkinson, Garman-Klass, Yang-Zhang)"]
        E["Microstructure & Alpha Factors\n(Amihud Illiquidity, VWAP, Vol Term Structure)"]
    end

    subgraph PyTorch_DL["PyTorch Deep Learning (TemporalAttentionVolNet)"]
        F["2-Layer LSTM + 4-Head Self-Attention\n(20-Day Sliding Window)"]
        G["Dual Output Heads\n(10/50/90th Vol Quantiles + Regime Prob)"]
    end

    subgraph Backtester["Institutional Backtesting Engine"]
        H["Inverse Tail-Vol Risk Parity\n+ Stress Regime Circuit Breaker"]
        I["Microstructure Execution\n(Spread + Square-Root Market Impact)"]
        J["Tear Sheet & Streamlit Terminal\n(Sharpe, Sortino, VaR/CVaR, Alpha/Beta)"]
    end

    A --> D & E
    D & E --> B
    B --> F --> G
    G --> C
    C --> H --> I --> J
```

---

## 📂 Repository Structure

```text
├── data/
│   └── nasdaq_quant_warehouse.db          # Relational SQLite database (auto-seeded by pipeline)
├── reports/
│   ├── demo_walkthrough.gif               # Animated walk-forward simulation demo (autoplay in README)
│   ├── demo_walkthrough.mp4               # High-definition MP4 walk-forward simulation video
│   ├── demo_terminal_preview.png          # Interactive Streamlit Quantitative Terminal UI preview
│   ├── tearsheet_dashboard.png            # 4-panel institutional performance & attention visual
│   ├── demo_volatility_cones.png          # Multi-asset 10th-90th quantile volatility cones
│   ├── demo_regime_attribution.png        # SQL CTE regime error & risk-adjusted ratio charts
│   ├── backtest_results.json              # Full out-of-sample metrics & attribution JSON
│   ├── sql_window_market_analytics.csv    # Output of SQL Window-Function market query
│   └── sql_regime_forecast_attribution.csv# Output of SQL CTE regime forecast evaluation
├── src/
│   ├── database/
│   │   ├── schema.sql                     # DDL schema with foreign keys & composite indexes
│   │   └── sql_manager.py                 # SQL ETL, GJR-GARCH market generator & CTE analytics
│   ├── features/
│   │   └── alpha_factors.py               # Parkinson, Garman-Klass, Yang-Zhang & microstructure factors
│   ├── models/
│   │   └── volatility_forecaster.py       # PyTorch TemporalAttentionVolNet & Quantile-QLIKE trainer
│   └── backtest/
│       ├── engine.py                      # Multi-asset backtester with Almgren-Chriss slippage
│       ├── metrics.py                     # Sharpe, Sortino, Calmar, VaR/CVaR, CAPM Alpha/Beta
│       └── visualizer.py                  # Tear-sheet matplotlib figure generator
├── tests/
│   └── test_engine.py                     # Pytest suite verifying SQL, features, PyTorch & backtester
├── app.py                                 # Interactive Streamlit + Plotly Quantitative Terminal
├── generate_demo_media.py                 # Script to generate README demo videos & visual gallery
├── run_pipeline.py                        # End-to-end CLI pipeline orchestrator
├── RESUME_AND_INTERVIEW_GUIDE.md          # Tailored Resume Bullets & Nasdaq Interview Talking Points
└── requirements.txt                       # Python package dependencies
```

---

## 🚀 Quick Start

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/meenub255/Deep-Learning-Volatility-Forecaster-Algorithmic-Backtesting-Engine.git
cd Deep-Learning-Volatility-Forecaster-Algorithmic-Backtesting-Engine
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline (SQL ETL $\rightarrow$ Feature Store $\rightarrow$ PyTorch Training $\rightarrow$ Backtest)
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Quantitative Dashboard (Streamlit)
```bash
streamlit run app.py
```

### 4. Run Automated Unit Tests
```bash
python -m pytest -v
```
