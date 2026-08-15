"""
Data download, cleaning, and feature engineering for market regime detection.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

# ── Download ──────────────────────────────────────────────────────────────────


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data, flatten MultiIndex, return clean DataFrame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False
        )

    if raw.empty:
        raise ValueError(
            f"No data returned for '{ticker}' in {start}→{end}. "
            "Check the ticker symbol."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw[~raw.index.duplicated()]

    required = {"Close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} for ticker '{ticker}'.")

    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


# ── Feature engineering ───────────────────────────────────────────────────────


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)  # neutral fill


def build_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    Construct a feature DataFrame from OHLCV data.

    Features:
      log_ret      — daily log return
      volatility   — rolling std of log returns (annualised)
      momentum_5d  — 5-day cumulative log return
      rsi          — RSI(14) normalised to [-1, 1]
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    out["log_ret"] = np.log(close / close.shift(1))
    out["volatility"] = out["log_ret"].rolling(vol_window).std() * np.sqrt(252)
    out["momentum_5d"] = np.log(close / close.shift(5))
    out["rsi"] = (_rsi(close) - 50) / 50  # [-1, 1]

    out["close"] = close
    out["volume"] = df.get("Volume", pd.Series(dtype=float))

    return out.dropna()


# ── Regime statistics ─────────────────────────────────────────────────────────

REGIME_LABELS = {0: "🐂 Bull", 1: "⚖️ Neutral", 2: "🐻 Bear", 3: "💥 Crisis"}
REGIME_COLORS = {
    "🐂 Bull": "#00d4aa",
    "⚖️ Neutral": "#faad14",
    "🐻 Bear": "#ef4444",
    "💥 Crisis": "#7c3aed",
}
DEFAULT_COLOR = "#64748b"


def label_regimes(feat: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
    """
    Auto-label regimes by sorting on mean log return.
    Highest return → Bull; lowest → Bear; extremes in volatility → Crisis.
    """
    regimes = sorted(feat[regime_col].unique())
    n = len(regimes)

    means = {r: feat.loc[feat[regime_col] == r, "log_ret"].mean() for r in regimes}
    vols = {r: feat.loc[feat[regime_col] == r, "volatility"].mean() for r in regimes}

    # Sort by mean return
    sorted_by_ret = sorted(regimes, key=lambda r: means[r], reverse=True)

    label_map: dict[int, str] = {}
    if n == 2:
        label_map[sorted_by_ret[0]] = "🐂 Bull"
        label_map[sorted_by_ret[1]] = "🐻 Bear"
    elif n == 3:
        label_map[sorted_by_ret[0]] = "🐂 Bull"
        label_map[sorted_by_ret[1]] = "⚖️ Neutral"
        label_map[sorted_by_ret[2]] = "🐻 Bear"
    elif n == 4:
        label_map[sorted_by_ret[0]] = "🐂 Bull"
        label_map[sorted_by_ret[1]] = "⚖️ Neutral"
        label_map[sorted_by_ret[2]] = "🐻 Bear"
        # Most volatile remaining → Crisis
        remaining = [r for r in sorted_by_ret[2:] if r not in label_map]
        if remaining:
            most_vol = max(remaining, key=lambda r: vols[r])
            label_map[most_vol] = "💥 Crisis"
            for r in remaining:
                if r not in label_map:
                    label_map[r] = "🐻 Bear"
    else:
        for i, r in enumerate(sorted_by_ret):
            key = list(REGIME_LABELS.values())[min(i, len(REGIME_LABELS) - 1)]
            label_map[r] = key

    feat = feat.copy()
    feat["regime_label"] = feat[regime_col].map(label_map)
    feat["regime_color"] = feat["regime_label"].map(REGIME_COLORS).fillna(DEFAULT_COLOR)
    return feat, label_map


def regime_statistics(feat: pd.DataFrame) -> pd.DataFrame:
    """Per-regime stats: return, vol, Sharpe, max drawdown, time %."""
    rows = []
    for label in feat["regime_label"].unique():
        sub = feat[feat["regime_label"] == label]
        ann_ret = sub["log_ret"].mean() * 252
        ann_vol = sub["log_ret"].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        pct_time = len(sub) / len(feat) * 100

        # Max drawdown within regime
        cum = np.exp(sub["log_ret"].cumsum())
        if len(cum) > 0:
            roll_max = cum.expanding().max()
            dd = (cum / roll_max - 1).min()
        else:
            dd = 0.0

        # Avg duration
        rle = (feat["regime_label"] != feat["regime_label"].shift()).cumsum()
        durations = (
            feat[feat["regime_label"] == label]
            .groupby(rle[feat["regime_label"] == label])
            .size()
        )
        avg_dur = durations.mean() if len(durations) else 0

        rows.append(
            {
                "Regime": label,
                "Ann. Return": f"{ann_ret * 100:+.2f}%",
                "Ann. Volatility": f"{ann_vol * 100:.2f}%",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Max Drawdown": f"{dd * 100:.2f}%",
                "% of Time": f"{pct_time:.1f}%",
                "Avg Duration": f"{avg_dur:.1f} days",
                "_color": REGIME_COLORS.get(label, DEFAULT_COLOR),
            }
        )

    return pd.DataFrame(rows)


def transition_matrix(
    feat: pd.DataFrame, n_regimes: int, label_map: dict
) -> pd.DataFrame:
    """Compute observed regime transition matrix."""
    {v: k for k, v in label_map.items()}
    sorted(feat["regime_label"].unique())

    seq = feat["regime"].values
    T = np.zeros((n_regimes, n_regimes))
    for i in range(len(seq) - 1):
        T[seq[i], seq[i + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, T / row_sums, 0)

    label_order = sorted(label_map.items(), key=lambda x: x[0])
    row_labels = [label_map[k] for k, _ in label_order]

    return pd.DataFrame(T, index=row_labels, columns=row_labels)
