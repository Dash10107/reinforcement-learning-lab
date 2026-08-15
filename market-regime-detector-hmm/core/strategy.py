"""
Regime-switching trading strategy backtester.

Strategy logic:
  🐂 Bull    → 100% invested
  ⚖️ Neutral → 50% invested
  🐻 Bear    → 0% (cash)
  💥 Crisis  → 0% (cash) or short (optional)

All signals are lagged by 1 day to avoid look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALLOCATION = {
    "🐂 Bull": 1.00,
    "⚖️ Neutral": 0.50,
    "🐻 Bear": 0.00,
    "💥 Crisis": 0.00,
}


def backtest_regime_strategy(
    feat: pd.DataFrame,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,  # 10bps per trade
) -> pd.DataFrame:
    """
    Back-test a regime-based allocation strategy against buy-and-hold.

    Returns a DataFrame with columns:
      date, close, log_ret, regime_label,
      allocation, strategy_ret, strategy_equity,
      bnh_equity, drawdown_strategy, drawdown_bnh
    """
    df = feat[["close", "log_ret", "regime_label"]].copy()
    df = df.dropna()

    # Lag allocation by 1 day (signal from yesterday's close)
    df["allocation"] = df["regime_label"].map(ALLOCATION).shift(1).fillna(0.0)

    # Transaction costs: charge when allocation changes
    df["alloc_change"] = df["allocation"].diff().abs().fillna(0)
    df["tc"] = df["alloc_change"] * transaction_cost

    # Strategy daily return
    df["strategy_ret"] = df["allocation"] * df["log_ret"] - df["tc"]

    # Equity curves
    df["strategy_equity"] = initial_capital * np.exp(df["strategy_ret"].cumsum())
    df["bnh_equity"] = initial_capital * np.exp(df["log_ret"].cumsum())

    # Drawdowns
    def _drawdown(equity: pd.Series) -> pd.Series:
        peak = equity.cummax()
        return equity / peak - 1

    df["drawdown_strategy"] = _drawdown(df["strategy_equity"])
    df["drawdown_bnh"] = _drawdown(df["bnh_equity"])

    return df.reset_index()


def strategy_metrics(bt: pd.DataFrame, trading_days: int = 252) -> dict:
    """Compute annualised performance metrics for strategy and buy-and-hold."""

    def _metrics(ret_col: str, eq_col: str, dd_col: str) -> dict:
        rets = bt[ret_col].dropna()
        eq = bt[eq_col]
        dd = bt[dd_col]
        n = len(rets)
        years = n / trading_days

        ann_ret = float(rets.sum() / max(years, 1e-9))
        ann_vol = float(rets.std() * np.sqrt(trading_days))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        max_dd = float(dd.min())
        total_r = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 0 else 0.0
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

        return {
            "Total Return": f"{total_r * 100:+.2f}%",
            "Ann. Return": f"{ann_ret * 100:+.2f}%",
            "Ann. Volatility": f"{ann_vol * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Calmar Ratio": f"{calmar:.2f}",
        }

    strategy_m = _metrics("strategy_ret", "strategy_equity", "drawdown_strategy")
    bnh_m = _metrics("log_ret", "bnh_equity", "drawdown_bnh")
    return {"Strategy": strategy_m, "Buy & Hold": bnh_m}
