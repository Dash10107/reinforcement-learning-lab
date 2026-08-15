"""
Analytics charts for the warehouse dashboard.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.figure import Figure

BG = "#0d1117"
BG2 = "#161b22"
GRID = "#21262d"
TEXT = "#e2e8f0"
DIM = "#4a5568"
C_REWARD = "#10b981"
C_DELIVERY = "#1890ff"
C_COLLISION = "#ef4444"
C_RANDOM = "#f59e0b"
C_TRAINED = "#10b981"


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=8, length=3)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(
            title,
            color=TEXT,
            fontsize=9,
            pad=7,
            fontfamily="monospace",
            fontweight="bold",
        )
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=8)


def _smooth(arr, k=10):
    if len(arr) <= k:
        return arr
    return np.convolve(arr, np.ones(k) / k, "valid")


def training_dashboard(state) -> Figure:
    fig = plt.figure(figsize=(13, 8), facecolor=BG)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.52,
        wspace=0.32,
        left=0.08,
        right=0.97,
        top=0.88,
        bottom=0.08,
    )

    eps = list(range(len(state.mean_rewards)))

    # Reward curve
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1, "TEAM REWARD PER EPISODE", "Episode", "Mean Agent Reward")
    if eps:
        ax1.plot(eps, state.mean_rewards, color=C_REWARD, alpha=0.25, linewidth=0.8)
        sm = _smooth(state.mean_rewards)
        ax1.plot(range(len(sm)), sm, color=C_REWARD, linewidth=2.2, label="Smoothed")
        ax1.axhline(
            state.rolling_mean,
            color=C_REWARD,
            linewidth=1,
            linestyle="--",
            alpha=0.5,
            label=f"Rolling: {state.rolling_mean:+.2f}",
        )
        ax1.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # Deliveries
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2, "DELIVERIES PER EPISODE", "Episode", "Count")
    if state.deliveries_per_ep:
        d_eps = list(range(len(state.deliveries_per_ep)))
        ax2.bar(d_eps, state.deliveries_per_ep, color=C_DELIVERY, alpha=0.5, width=1.0)
        sm_d = _smooth(state.deliveries_per_ep)
        ax2.plot(range(len(sm_d)), sm_d, color=C_DELIVERY, linewidth=2.0)

    # Collisions
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3, "COLLISIONS PER EPISODE", "Episode", "Count")
    if state.collisions_per_ep:
        c_eps = list(range(len(state.collisions_per_ep)))
        ax3.bar(c_eps, state.collisions_per_ep, color=C_COLLISION, alpha=0.5, width=1.0)
        sm_c = _smooth(state.collisions_per_ep)
        ax3.plot(range(len(sm_c)), sm_c, color=C_COLLISION, linewidth=2.0)

    n = len(eps)
    avg_d = np.mean(state.deliveries_per_ep[-20:]) if state.deliveries_per_ep else 0
    avg_c = np.mean(state.collisions_per_ep[-20:]) if state.collisions_per_ep else 0
    fig.suptitle(
        f"WAREHOUSE IPPO TRAINING  ·  {n} episodes  ·  "
        f"Avg deliveries: {avg_d:.1f}/ep  ·  Avg collisions: {avg_c:.1f}/ep",
        color=TEXT,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
        y=0.95,
    )
    return fig


def comparison_chart(results: dict[str, dict]) -> Figure:
    """Bar chart comparing strategies: deliveries, collisions, reward."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=BG)
    fig.patch.set_facecolor(BG)

    strategies = list(results.keys())
    s_colors = {
        "IPPO (trained)": C_TRAINED,
        "Random": C_RANDOM,
        "Greedy": "#8b5cf6",
        "No-op": "#4a5568",
    }

    metrics = [
        ("deliveries", "Deliveries", C_DELIVERY),
        ("collisions", "Collisions", C_COLLISION),
        ("reward", "Team Reward", C_REWARD),
    ]
    for ax, (key, label, _) in zip(axes, metrics):
        _style(ax, label.upper())
        vals = [results[s].get(key, 0) for s in strategies]
        cols = [s_colors.get(s, TEXT) for s in strategies]
        bars = ax.bar(
            strategies, vals, color=cols, edgecolor=BG, linewidth=0.8, width=0.5
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(min(vals + [0])) * 0.03,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                color=TEXT,
                fontsize=9,
                fontweight="bold",
            )
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.tick_params(axis="x", colors=TEXT, labelsize=8)

    fig.suptitle(
        "STRATEGY COMPARISON  ·  Same Warehouse, 100 Steps",
        color=TEXT,
        fontsize=11,
        fontfamily="monospace",
        fontweight="bold",
    )
    fig.tight_layout(pad=2)
    return fig


def empty_fig(msg: str = "") -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
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
    )
    ax.axis("off")
    return fig
