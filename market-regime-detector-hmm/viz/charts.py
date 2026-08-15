"""
All matplotlib charts for the Market Regime Detector.
Dark financial-terminal aesthetic throughout.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from core.data import DEFAULT_COLOR, REGIME_COLORS
from matplotlib import gridspec
from matplotlib.figure import Figure

# ── Design tokens ─────────────────────────────────────────────────────────────
BG = "#0a0d14"
BG2 = "#0f1520"
BG3 = "#161e2e"
GRID = "#1e2a3d"
TEXT = "#e2e8f0"
DIM = "#4a6080"
UP = "#00d4aa"
DOWN = "#ef4444"
NEUT = "#faad14"


def _style(ax, title="", xlabel="", ylabel="", fontsize=9):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=8, length=3)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(
            title,
            color=TEXT,
            fontsize=fontsize,
            pad=8,
            fontfamily="monospace",
            fontweight="bold",
        )
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=8)


def _color(label: str) -> str:
    return REGIME_COLORS.get(label, DEFAULT_COLOR)


# ── 1. Main regime chart ──────────────────────────────────────────────────────


def regime_price_chart(feat: pd.DataFrame, ticker: str) -> Figure:
    """
    3-panel chart:
      Top:    Price line with colored background bands per regime
      Middle: Log returns coloured by regime
      Bottom: Regime timeline bar
    """
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[4, 2, 0.4],
        hspace=0.08,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.06,
    )

    dates = feat.index
    close = feat["close"]
    rets = feat["log_ret"]
    labels = feat["regime_label"]
    colors = feat["regime_color"]

    # ── Panel 1: Price ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    _style(ax1, ylabel="Price ($)")

    # Regime background bands
    prev_label = None
    band_start = dates[0]
    for i, (d, lbl) in enumerate(zip(dates, labels)):
        if lbl != prev_label:
            if prev_label is not None:
                ax1.axvspan(
                    band_start, d, alpha=0.12, color=_color(prev_label), linewidth=0
                )
            band_start = d
            prev_label = lbl
    if prev_label:
        ax1.axvspan(
            band_start, dates[-1], alpha=0.12, color=_color(prev_label), linewidth=0
        )

    ax1.plot(dates, close, color="#94a3b8", linewidth=1.0, alpha=0.9)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.tick_params(labelbottom=False)

    # Regime legend
    seen = {}
    for lbl, col in zip(labels, colors):
        if lbl not in seen:
            seen[lbl] = col
    handles = [mpatches.Patch(color=c, label=l) for l, c in seen.items()]
    ax1.legend(
        handles=handles,
        facecolor=BG3,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8,
        loc="upper left",
    )

    # Price annotation
    ax1.set_title(
        f"{ticker.upper()} — Market Regime Detection  "
        f"({dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')})",
        color=TEXT,
        fontsize=10,
        pad=8,
        fontfamily="monospace",
        fontweight="bold",
    )

    # ── Panel 2: Returns ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _style(ax2, ylabel="Log Return")
    for i in range(len(dates)):
        col = _color(labels.iloc[i])
        ax2.bar(dates[i], rets.iloc[i], color=col, alpha=0.7, width=1.5)
    ax2.axhline(0, color=GRID, linewidth=0.8)
    ax2.tick_params(labelbottom=False)

    # ── Panel 3: Regime bar ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor(BG2)
    for i in range(len(dates)):
        ax3.axvspan(
            dates[i],
            dates[i] if i == len(dates) - 1 else dates[i + 1],
            facecolor=_color(labels.iloc[i]),
            alpha=0.9,
        )
    ax3.set_yticks([])
    ax3.tick_params(colors=DIM, labelsize=7, length=2)
    for sp in ax3.spines.values():
        sp.set_color(GRID)
    ax3.set_xlabel("Date", color=DIM, fontsize=8)

    return fig


# ── 2. Regime analytics dashboard ────────────────────────────────────────────


def regime_analytics(
    feat: pd.DataFrame,
    stats_df: pd.DataFrame,
    trans_df: pd.DataFrame,
    diagnostics: dict,
    bic_scores: list[tuple[int, float]],
) -> Figure:
    """4-panel: return distributions, transition heatmap, duration, BIC curve."""
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.52,
        wspace=0.35,
        left=0.08,
        right=0.97,
        top=0.91,
        bottom=0.08,
    )

    labels_present = feat["regime_label"].unique()

    # ── Return distributions ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1, "RETURN DISTRIBUTION BY REGIME", "Log Return", "Density")
    for lbl in labels_present:
        sub = feat.loc[feat["regime_label"] == lbl, "log_ret"]
        col = _color(lbl)
        sub.plot.kde(ax=ax1, color=col, linewidth=2.0, label=lbl, bw_method=0.4)
        ax1.axvline(sub.mean(), color=col, linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.axvline(0, color=GRID, linewidth=0.8)
    ax1.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=7)

    # ── Transition matrix heatmap ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2, "REGIME TRANSITION PROBABILITIES")
    if not trans_df.empty:
        T = trans_df.values.astype(float)
        im = ax2.imshow(T, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=DIM, labelsize=7)
        cbar.set_label("Probability", color=DIM, fontsize=7)
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                ax2.text(
                    j,
                    i,
                    f"{T[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if T[i, j] > 0.5 else TEXT,
                    fontsize=8,
                    fontweight="bold",
                )
        ax2.set_xticks(range(len(trans_df.columns)))
        ax2.set_yticks(range(len(trans_df.index)))
        ax2.set_xticklabels(trans_df.columns, color=DIM, fontsize=7, rotation=20)
        ax2.set_yticklabels(trans_df.index, color=DIM, fontsize=7)

    # ── Regime time share ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3, "TIME IN EACH REGIME  (%)", "Regime", "% of Time")
    if not stats_df.empty:
        names = stats_df["Regime"].tolist()
        pcts = [float(v.replace("%", "")) for v in stats_df["% of Time"].tolist()]
        cols = [_color(n) for n in names]
        bars = ax3.bar(names, pcts, color=cols, edgecolor=BG, linewidth=0.8)
        for bar, val in zip(bars, pcts):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                color=TEXT,
                fontsize=9,
                fontweight="bold",
            )
        ax3.tick_params(axis="x", colors=TEXT, labelsize=8)

    # ── BIC model selection ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4, "BIC MODEL SELECTION  (lower = better)", "# Regimes", "BIC Score")
    if bic_scores:
        ns = [b[0] for b in bic_scores]
        bics = [b[1] for b in bic_scores]
        ax4.plot(
            ns,
            bics,
            color=UP,
            linewidth=2.2,
            marker="o",
            markersize=7,
            markerfacecolor=BG2,
            markeredgecolor=UP,
            markeredgewidth=1.5,
        )
        best_n = ns[np.argmin(bics)]
        ax4.axvline(
            best_n, color=NEUT, linewidth=1.2, linestyle="--", label=f"Best n={best_n}"
        )
        ax4.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
        ax4.set_xticks(ns)
        ax4.tick_params(axis="x", colors=DIM, labelsize=9)

    n = diagnostics.get("n_observations", 0)
    ll = diagnostics.get("log_likelihood", 0)
    fig.suptitle(
        f"REGIME ANALYTICS  ·  {n} observations  ·  "
        f"Log-likelihood: {ll:.0f}  ·  "
        f"Converged: {'✓' if diagnostics.get('converged') else '✗'}",
        color=TEXT,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
        y=0.97,
    )
    return fig


# ── 3. Backtest chart ─────────────────────────────────────────────────────────


def backtest_chart(bt: pd.DataFrame, metrics: dict, ticker: str) -> Figure:
    """3-panel: equity curves, drawdowns, rolling Sharpe."""
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[4, 2, 2],
        hspace=0.45,
        left=0.08,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )

    dates = pd.to_datetime(bt["date"]) if "date" in bt.columns else pd.Series(bt.index)

    # ── Equity curves ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    _style(ax1, ylabel="Portfolio Value ($)")

    # Regime bands
    if "regime_label" in bt.columns:
        for i in range(len(bt) - 1):
            ax1.axvspan(
                dates.iloc[i],
                dates.iloc[i + 1],
                facecolor=_color(bt["regime_label"].iloc[i]),
                alpha=0.07,
                linewidth=0,
            )

    ax1.plot(
        dates, bt["strategy_equity"], color=UP, linewidth=2.2, label="Regime Strategy"
    )
    ax1.plot(
        dates,
        bt["bnh_equity"],
        color=NEUT,
        linewidth=1.8,
        linestyle="--",
        label="Buy & Hold",
    )
    ax1.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    ax1.axhline(bt["strategy_equity"].iloc[0], color=GRID, linewidth=0.8, linestyle=":")

    strat_m = metrics.get("Strategy", {})
    bnh_m = metrics.get("Buy & Hold", {})
    ax1.set_title(
        f"REGIME STRATEGY vs BUY & HOLD  ·  {ticker.upper()}  ·  "
        f"Strategy: {strat_m.get('Total Return', '—')}  ·  "
        f"B&H: {bnh_m.get('Total Return', '—')}",
        color=TEXT,
        fontsize=10,
        pad=8,
        fontfamily="monospace",
        fontweight="bold",
    )

    # ── Drawdowns ─────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _style(ax2, ylabel="Drawdown")
    ax2.fill_between(dates, bt["drawdown_strategy"], alpha=0.4, color=UP)
    ax2.fill_between(dates, bt["drawdown_bnh"], alpha=0.25, color=NEUT)
    ax2.plot(dates, bt["drawdown_strategy"], color=UP, linewidth=1.2, label="Strategy")
    ax2.plot(
        dates,
        bt["drawdown_bnh"],
        color=NEUT,
        linewidth=1.0,
        linestyle="--",
        label="B&H",
    )
    ax2.axhline(0, color=GRID, linewidth=0.8)
    ax2.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax2.tick_params(labelbottom=False)

    # ── Allocation over time ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _style(ax3, "ALLOCATION (%)", "Date", "%")
    ax3.fill_between(dates, bt["allocation"] * 100, alpha=0.5, color=UP)
    ax3.plot(dates, bt["allocation"] * 100, color=UP, linewidth=1.0)
    ax3.set_ylim(-5, 115)
    ax3.axhline(100, color=GRID, linewidth=0.5, linestyle=":")
    ax3.set_xlabel("Date", color=DIM, fontsize=8)

    return fig


# ── 4. Multi-asset regime comparison ─────────────────────────────────────────


def multi_asset_chart(results: dict[str, pd.DataFrame]) -> Figure:
    """
    Stacked regime-label timeline for multiple tickers.
    Each row = one ticker, color = regime.
    """
    tickers = list(results.keys())
    n = len(tickers)
    if n == 0:
        return empty_fig("No results.")

    fig, axes = plt.subplots(
        n + 1,
        1,
        figsize=(14, 2 + n * 1.5),
        facecolor=BG,
        gridspec_kw={"height_ratios": [3] * n + [0.3]},
    )
    fig.patch.set_facecolor(BG)

    if n == 1:
        axes = [axes[0], axes[1]]

    for ax, ticker in zip(axes[:n], tickers):
        feat = results[ticker]
        dates = feat.index
        labels = feat["regime_label"]

        ax.set_facecolor(BG2)
        for i in range(len(dates)):
            x0 = dates[i]
            x1 = dates[i + 1] if i < len(dates) - 1 else dates[i]
            ax.axvspan(
                x0, x1, facecolor=_color(labels.iloc[i]), alpha=0.85, linewidth=0
            )

        ax.set_yticks([])
        ax.set_ylabel(
            ticker.upper(),
            color=TEXT,
            fontsize=9,
            fontweight="bold",
            rotation=0,
            labelpad=36,
        )
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=DIM, labelsize=7, length=2, labelbottom=False)

    # Last row: x-axis labels
    axes[-1].set_facecolor(BG)
    axes[-1].axis("off")

    # Legend
    all_labels = set()
    for feat in results.values():
        all_labels.update(feat["regime_label"].unique())
    handles = [mpatches.Patch(color=_color(l), label=l) for l in sorted(all_labels)]
    fig.legend(
        handles=handles,
        facecolor=BG3,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.97),
    )

    fig.suptitle(
        "MULTI-ASSET REGIME SYNCHRONISATION",
        color=TEXT,
        fontsize=11,
        fontfamily="monospace",
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── 5. Current regime gauge ───────────────────────────────────────────────────


def regime_gauge(probs: np.ndarray, label_map: dict) -> Figure:
    """Horizontal bar chart showing current regime posterior probabilities."""
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    labels = [label_map.get(i, f"R{i}") for i in range(len(probs))]
    y_pos = range(len(labels))

    for i, (lbl, prob) in enumerate(zip(labels, probs)):
        col = _color(lbl)
        ax.barh(i, prob * 100, color=col, edgecolor=BG, linewidth=0.5, alpha=0.85)
        ax.text(
            prob * 100 + 0.5,
            i,
            f"{prob * 100:.1f}%",
            va="center",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, color=TEXT, fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)", color=DIM, fontsize=8)
    ax.tick_params(colors=DIM, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_title(
        "CURRENT REGIME PROBABILITY",
        color=TEXT,
        fontsize=9,
        pad=8,
        fontfamily="monospace",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ── 6. Empty placeholder ──────────────────────────────────────────────────────


def empty_fig(msg: str = "Run analysis to see charts.") -> Figure:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.text(
        0.5,
        0.5,
        msg,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=DIM,
        fontsize=12,
        fontfamily="monospace",
        wrap=True,
    )
    ax.axis("off")
    return fig
