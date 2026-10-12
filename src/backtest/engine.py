"""
Institutional Multi-Asset Backtesting Engine with Market Microstructure Cost Modeling.

Simulates portfolio construction, inverse-volatility Kelly/Risk-Parity sizing,
regime-based drawdown protection, and realistic execution slippage (bid-ask spread
crossing + Almgren-Chriss square-root market impact).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_performance_tearsheet, compute_underwater_drawdown


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    target_annual_vol: float = 0.165
    max_leverage: float = 1.35
    commission_bps: float = 1.0
    impact_coeff: float = 0.08
    rebalance_every_n_days: int = 5
    min_trade_weight_delta: float = 0.025
    risk_free_rate_annual: float = 0.04


class InstitutionalBacktester:
    """
    Executes multi-asset algorithmic trading strategies against out-of-sample
    historical datasets with full trade ledger and slippage attribution.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def _compute_target_weights(
        self,
        day_slice: pd.DataFrame,
        mode: str,
    ) -> Dict[str, float]:
        """
        Computes cross-sectional portfolio weights for a given rebalance date.

        Modes:
        - 'pytorch_alpha': PyTorch TemporalAttentionVolNet inverse-vol + directional probability
                           + upper-quantile tail-risk regime circuit breaker.
        - 'garch_parity': Classical GARCH(1,1) inverse-volatility risk parity.
        - 'benchmark_ew': Equal-weight passive allocation across Nasdaq universe.
        """
        symbols = day_slice["symbol"].tolist()
        n_assets = len(symbols)
        if n_assets == 0:
            return {}

        if mode == "benchmark_ew":
            w = 1.0 / n_assets
            return {sym: w for sym in symbols}

        if mode == "garch_parity":
            garch_vols = np.maximum(day_slice["garch_baseline_vol"].to_numpy(), 0.08)
            inv_vol = 1.0 / garch_vols
            raw_weights = inv_vol / np.sum(inv_vol)
            port_vol_est = float(np.sum(raw_weights * garch_vols) * 0.78)
            scalar = np.clip(self.config.target_annual_vol / (port_vol_est + 1e-8), 0.35, 1.15)
            final_w = raw_weights * scalar
            return {sym: float(final_w[i]) for i, sym in enumerate(symbols)}

        # Mode: 'pytorch_alpha'
        pred_vol = np.maximum(day_slice["pred_vol_mean"].to_numpy(), 0.08)
        pred_q90 = np.maximum(day_slice["pred_vol_q90"].to_numpy(), 0.10)
        up_prob = day_slice["pred_up_prob"].to_numpy()
        mom_10 = day_slice["momentum_10d"].to_numpy()
        rsi = day_slice["rsi_14"].to_numpy()
        vol_term = day_slice["vol_term_structure"].to_numpy()

        # Composite alpha conviction score per asset
        conviction = np.zeros(n_assets)
        for i in range(n_assets):
            prob_edge = (up_prob[i] - 0.45) * 2.5
            trend_score = np.tanh(mom_10[i] * 10.0)
            rsi_filter = 0.70 if rsi[i] > 78.0 else (1.20 if 45.0 <= rsi[i] <= 70.0 else 0.90)
            # Reward volatility contraction (vol_term < 0.95) and penalize sharp vol backwardation spikes (> 1.25)
            term_mult = 1.20 if vol_term[i] < 0.95 else (0.45 if vol_term[i] > 1.25 else 1.0)

            score = max(0.05, (0.50 * prob_edge + 0.50 * max(-0.15, trend_score) + 0.45)) * rsi_filter * term_mult
            # Inverse predicted volatility risk-parity weighting
            conviction[i] = score / (0.70 * pred_vol[i] + 0.30 * pred_q90[i])

        if np.sum(conviction) <= 1e-8:
            return {sym: 0.0 for sym in symbols}

        norm_w = conviction / np.sum(conviction)
        # Allow high-conviction winners up to 38% weight
        norm_w = np.minimum(norm_w, 0.38)
        norm_w = norm_w / np.sum(norm_w)

        # Dynamic Portfolio Volatility Targeting & Regime Circuit Breaker
        mkt_pred_vol = float(np.mean(pred_vol))
        mkt_q90_vol = float(np.mean(pred_q90))
        avg_pred_vol = float(np.sum(norm_w * pred_vol))
        diversified_vol = avg_pred_vol * 0.75

        # De-lever aggressively when market predicted volatility spikes or cross-sectional trend turns negative
        avg_mom = float(np.mean(mom_10))
        avg_ret_today = float(np.mean(day_slice["log_return"]))
        regime_dampener = 1.08
        if mkt_pred_vol > 0.272 or mkt_q90_vol > 0.355 or avg_mom < -0.006:
            regime_dampener = 0.20 if avg_mom < 0.0 else 0.45
        elif avg_mom < 0.002 or avg_ret_today < -0.012:
            regime_dampener = 0.55
        elif mkt_pred_vol < 0.245 and avg_mom > 0.006:
            regime_dampener = 1.30

        gross_exposure = np.clip(
            (self.config.target_annual_vol / (diversified_vol + 1e-8)) * regime_dampener,
            0.12,
            self.config.max_leverage,
        )

        final_weights = norm_w * gross_exposure
        return {sym: float(final_weights[i]) for i, sym in enumerate(symbols)}

    def run_strategy(
        self,
        df_oos: pd.DataFrame,
        mode: str = "pytorch_alpha",
        run_id: str = "PYTORCH_VOL_ALPHA_V1",
        strategy_name: str = "PyTorch Temporal-Attention Vol-Target Alpha",
        benchmark_returns: pd.Series | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Simulates daily portfolio mark-to-market, rebalance execution, slippage, and cash interest.
        """
        df = df_oos.sort_values(["trade_date", "symbol"]).copy()
        unique_dates = sorted(df["trade_date"].unique())

        equity = self.config.initial_capital
        rf_daily = (1.0 + self.config.risk_free_rate_annual) ** (1.0 / 252.0) - 1.0

        current_weights: Dict[str, float] = {}
        equity_records: List[Dict] = []
        trade_records: List[Dict] = []

        for day_idx, dt in enumerate(unique_dates):
            day_slice = df[df["trade_date"] == dt].reset_index(drop=True)

            # 1. Realize daily mark-to-market return from positions held entering today
            daily_gross_ret = 0.0
            total_invested_weight = sum(current_weights.values())
            cash_weight = 1.0 - total_invested_weight

            for _, row in day_slice.iterrows():
                sym = str(row["symbol"])
                w_sym = current_weights.get(sym, 0.0)
                if w_sym > 0.0:
                    asset_simple_ret = float(np.exp(row["log_return"]) - 1.0)
                    daily_gross_ret += w_sym * asset_simple_ret

            # Cash earns daily risk-free interest (or pays financing cost if leveraged > 1.0)
            cash_ret = cash_weight * rf_daily
            equity_before_trades = equity * (1.0 + daily_gross_ret + cash_ret)

            # 2. Check if rebalance is triggered today (scheduled 5-day or immediate volatility circuit breaker)
            mkt_vol_today = float(day_slice["pred_vol_mean"].mean())
            mkt_mom_today = float(day_slice["momentum_10d"].mean())
            stress_trigger = (
                mode == "pytorch_alpha"
                and total_invested_weight > 0.55
                and (mkt_vol_today > 0.272 or mkt_mom_today < -0.008)
            )
            is_rebalance_day = (
                day_idx == 0
                or stress_trigger
                or (mode != "benchmark_ew" and day_idx % self.config.rebalance_every_n_days == 0)
            )

            day_commission = 0.0
            day_slippage = 0.0

            if is_rebalance_day:
                target_weights = self._compute_target_weights(day_slice, mode=mode)

                for _, row in day_slice.iterrows():
                    sym = str(row["symbol"])
                    old_w = current_weights.get(sym, 0.0)
                    new_w = target_weights.get(sym, 0.0)
                    delta_w = new_w - old_w

                    if day_idx > 0 and abs(delta_w) < self.config.min_trade_weight_delta:
                        continue

                    notional_traded = abs(delta_w) * equity_before_trades
                    if notional_traded < 100.0:
                        continue

                    ref_price = float(row["close_price"])
                    spread_bps = float(row["spread_bps"])
                    daily_vol = float(row["pred_vol_mean"]) / np.sqrt(252.0)
                    adv_usd = max(float(row["close_price"] * row["volume"]), 1e6)

                    # Microstructure Slippage = Half-Spread + Square-Root Market Impact
                    half_spread_rate = 0.5 * (spread_bps * 1e-4)
                    impact_rate = self.config.impact_coeff * daily_vol * np.sqrt(notional_traded / adv_usd)
                    slippage_rate = half_spread_rate + impact_rate
                    comm_rate = self.config.commission_bps * 1e-4

                    side = "BUY" if delta_w > 0 else "SELL"
                    exec_price = ref_price * (1.0 + slippage_rate if side == "BUY" else 1.0 - slippage_rate)
                    shares = notional_traded / ref_price

                    comm_usd = notional_traded * comm_rate
                    slip_usd = notional_traded * slippage_rate

                    day_commission += comm_usd
                    day_slippage += slip_usd
                    current_weights[sym] = new_w

                    trade_records.append({
                        "run_id": run_id,
                        "trade_date": dt,
                        "symbol": sym,
                        "side": side,
                        "target_weight": round(new_w, 4),
                        "shares_traded": round(shares, 2),
                        "reference_price": round(ref_price, 4),
                        "execution_price": round(exec_price, 4),
                        "notional_usd": round(notional_traded, 2),
                        "commission_usd": round(comm_usd, 2),
                        "slippage_usd": round(slip_usd, 2),
                        "pred_vol_signal": round(float(row["pred_vol_mean"]), 4),
                        "regime_label": str(row["volatility_regime"]),
                    })

            equity = equity_before_trades - (day_commission + day_slippage)
            equity_records.append({
                "trade_date": dt,
                "equity": equity,
                "gross_exposure": sum(current_weights.values()),
                "daily_costs_usd": day_commission + day_slippage,
                "avg_pred_vol": float(day_slice["pred_vol_mean"].mean()),
                "avg_realized_vol": float(day_slice["actual_fwd_vol"].mean()),
            })

        df_eq = pd.DataFrame(equity_records)
        df_eq["drawdown_pct"] = compute_underwater_drawdown(df_eq["equity"]) * 100.0
        df_trades = pd.DataFrame(trade_records)

        summary = compute_performance_tearsheet(
            run_id=run_id,
            strategy_name=strategy_name,
            df_equity=df_eq,
            df_trades=df_trades,
            initial_capital=self.config.initial_capital,
            risk_free_rate_annual=self.config.risk_free_rate_annual,
            benchmark_returns=benchmark_returns,
        )
        return df_eq, df_trades, summary
