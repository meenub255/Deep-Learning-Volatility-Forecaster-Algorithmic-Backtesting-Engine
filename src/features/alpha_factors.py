"""
Institutional Quantitative Feature Engineering & Range-Based Volatility Estimators.

Implements strictly causal (zero look-ahead bias) time-series features:
- Close-to-Close Realized Volatility (5d, 10d, 21d)
- Parkinson High-Low Range Volatility Estimator (1980)
- Garman-Klass OHLC Volatility Estimator (1980)
- Yang-Zhang Drift-Independent & Jump-Robust Volatility Estimator (2000)
- Volatility Term Structure Ratio (Short-Term / Long-Term Vol)
- Market Microstructure Proxies: Amihud Illiquidity Ratio, VWAP Dislocation, Spread Regime
- Momentum & Tail-Risk Factors: RSI-14, Normalized MACD, Bollinger Bandwidth, Rolling Skewness
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


FEATURE_COLUMNS: List[str] = [
    "log_return",
    "realized_vol_5d",
    "realized_vol_21d",
    "parkinson_vol_21d",
    "garman_klass_vol_21d",
    "yang_zhang_vol_21d",
    "vol_term_structure",
    "rsi_14",
    "macd_hist",
    "bb_width",
    "atr_pct",
    "vwap_dislocation",
    "amihud_illiquidity",
    "volume_zscore",
    "return_skew_21d",
    "momentum_10d",
]


class QuantFeatureEngineer:
    """Computes institutional volatility estimators and microstructure alpha factors."""

    def __init__(self, annualization_factor: int = 252, forecast_horizon: int = 5) -> None:
        self.ann_factor = annualization_factor
        self.forecast_horizon = forecast_horizon

    def _compute_single_asset_features(self, df_asset: pd.DataFrame) -> pd.DataFrame:
        df = df_asset.sort_values("trade_date").copy()

        o = df["open_price"].astype(float)
        h = df["high_price"].astype(float)
        l = df["low_price"].astype(float)
        c = df["close_price"].astype(float)
        v = df["volume"].astype(float)
        vwap = df["vwap"].astype(float)

        # 1. Log returns & multi-horizon Close-to-Close Realized Volatility
        df["log_return"] = np.log(c / c.shift(1))
        df["realized_vol_5d"] = df["log_return"].rolling(5).std() * np.sqrt(self.ann_factor)
        df["realized_vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(self.ann_factor)

        # 2. Parkinson (1980) High-Low Range Volatility Estimator
        log_hl = np.log(h / l)
        parkinson_var = (1.0 / (4.0 * np.log(2.0))) * (log_hl ** 2)
        df["parkinson_vol_21d"] = np.sqrt(parkinson_var.rolling(21).mean() * self.ann_factor)

        # 3. Garman-Klass (1980) OHLC Volatility Estimator
        log_co = np.log(c / o)
        gk_daily_var = np.maximum(1e-10, 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2))
        df["garman_klass_vol_21d"] = np.sqrt(gk_daily_var.rolling(21).mean() * self.ann_factor)

        # 4. Yang-Zhang (2000) Jump- & Drift-Adjusted Volatility Estimator
        window_yz = 21
        log_oc_overnight = np.log(o / c.shift(1))
        log_ho = np.log(h / o)
        log_hc = np.log(h / c)
        log_lo = np.log(l / o)
        log_lc = np.log(l / c)

        rs_daily_var = np.maximum(0.0, log_ho * log_hc + log_lo * log_lc)
        overnight_var = log_oc_overnight.rolling(window_yz).var()
        open_close_var = log_co.rolling(window_yz).var()
        rs_var = rs_daily_var.rolling(window_yz).mean()

        k_yz = 0.34 / (1.34 + (window_yz + 1.0) / (window_yz - 1.0))
        yz_var = np.maximum(1e-10, overnight_var + k_yz * open_close_var + (1.0 - k_yz) * rs_var)
        df["yang_zhang_vol_21d"] = np.sqrt(yz_var * self.ann_factor)

        # 5. Volatility Term Structure (Short-term 5d vs 21d Yang-Zhang)
        df["vol_term_structure"] = df["realized_vol_5d"] / (df["yang_zhang_vol_21d"] + 1e-8)

        # 6. RSI-14 (Wilder's exponential smoothing)
        delta = c.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # 7. Normalized MACD Histogram
        ema_12 = c.ewm(span=12, adjust=False).mean()
        ema_26 = c.ewm(span=26, adjust=False).mean()
        macd_line = (ema_12 - ema_26) / (c + 1e-8)
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = (macd_line - macd_signal) * 100.0

        # 8. Bollinger Band Width & Normalized ATR
        sma_20 = c.rolling(20).mean()
        std_20 = c.rolling(20).std()
        df["bb_width"] = (4.0 * std_20) / (sma_20 + 1e-8)

        tr = pd.concat([
            (h - l),
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_pct"] = tr.rolling(14).mean() / (c + 1e-8)

        # 9. Microstructure & Order Flow Proxies
        df["vwap_dislocation"] = (c - vwap) / (vwap + 1e-8)
        dollar_vol_millions = (c * v) / 1e6
        df["amihud_illiquidity"] = (df["log_return"].abs() / (dollar_vol_millions + 1e-8)).rolling(21).mean() * 1e4

        vol_mean = v.rolling(21).mean()
        vol_std = v.rolling(21).std()
        df["volume_zscore"] = (v - vol_mean) / (vol_std + 1e-8)

        # 10. Higher-moment Return Skewness & 10-Day Momentum
        df["return_skew_21d"] = df["log_return"].rolling(21).skew().fillna(0.0)
        df["momentum_10d"] = c / c.shift(10) - 1.0

        # 11. Forward Targets (Strictly shifted backward ONLY for supervised training targets)
        # Forward 5-day realized volatility (using Yang-Zhang / Garman-Klass range + close returns)
        fwd_gk_sq = 0.5 * (gk_daily_var + df["log_return"] ** 2) * self.ann_factor
        fwd_mean = fwd_gk_sq.iloc[::-1].rolling(self.forecast_horizon).mean().iloc[::-1].shift(-1)
        df["target_fwd_vol_5d"] = np.sqrt(fwd_mean)
        df["target_fwd_ret_5d"] = np.log(c.shift(-self.forecast_horizon) / c)

        return df

    def build_feature_matrix(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Processes all symbols in the universe and returns clean feature matrix."""
        frames = []
        for _, group in df_ohlcv.groupby("symbol", sort=False):
            frames.append(self._compute_single_asset_features(group))
        df_all = pd.concat(frames, ignore_index=True)
        # Drop warm-up rows where rolling 21d features are NaN
        df_clean = df_all.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        return df_clean
