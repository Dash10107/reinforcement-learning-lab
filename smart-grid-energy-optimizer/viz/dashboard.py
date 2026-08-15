"""
Professional energy management dashboard charts.
All figures use a dark SCADA-inspired aesthetic.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.figure import Figure

# ── Design tokens ─────────────────────────────────────────────────────────────
BG = "#0a0f1e"
BG2 = "#0d1526"
BG3 = "#111e35"
GRID = "#1a2744"
TEXT = "#e8eaf0"
DIM = "#4a6080"

C_SOLAR = "#faad14"  # yellow — solar generation
C_GRID_I = "#1890ff"  # blue   — grid import
C_GRID_E = "#13c2c2"  # cyan   — grid export
C_BAT_C = "#52c41a"  # green  — battery charging
C_BAT_D = "#ff4d4f"  # red    — battery discharging
C_LOAD = "#722ed1"  # purple — building load
C_SOC = "#00d4aa"  # teal   — state of charge
C_PRICE = "#ff7a45"  # orange — price
C_DP = "#52c41a"
C_DQN = "#1890ff"
C_NAIVE = "#faad14"
C_NOBAT = "#ff4d4f"

HOURS = list(range(24))
H_LABELS = [f"{h:02d}:00" for h in HOURS]


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(
            title,
            color=TEXT,
            fontsize=10,
            pad=8,
            fontfamily="monospace",
            fontweight="bold",
        )
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=8)


def _set_hour_ticks(ax, step=4):
    ax.set_xticks(range(0, 24, step))
    ax.set_xticklabels(
        [H_LABELS[i] for i in range(0, 24, step)], color=DIM, fontsize=7, rotation=30
    )


def dispatch_dashboard(
    log: dict, scenario_name: str = "", strategy: str = ""
) -> Figure:
    """
    Main 24-hour dispatch dashboard — the centrepiece chart.
    Shows: price + SOC (top), energy flows stacked (middle), hourly P&L (bottom).
    """
    hours = log["hour"]
    if not hours:
        return _empty_fig("Run a simulation to see the dispatch dashboard.")

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.55,
        wspace=0.32,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )

    # ── Panel 1: Electricity Price ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(
        ax1, "⚡ ELECTRICITY PRICE  &  BATTERY STATE OF CHARGE", ylabel="Price (¢/kWh)"
    )
    ax1b = ax1.twinx()
    ax1.fill_between(hours, log["price"], alpha=0.15, color=C_PRICE)
    ax1.plot(hours, log["price"], color=C_PRICE, linewidth=2.0, label="Price (¢/kWh)")
    ax1b.plot(
        hours,
        [s / max(log["soc"] + [1e-9]) * 100 for s in log["soc"]],
        color=C_SOC,
        linewidth=2.5,
        linestyle="--",
        label="SOC (%)",
    )
    ax1b.set_ylabel("Battery SOC (%)", color=C_SOC, fontsize=8)
    ax1b.tick_params(colors=C_SOC, labelsize=8)
    ax1b.set_ylim(-5, 115)
    ax1.set_ylim(bottom=0)
    _set_hour_ticks(ax1)
    # Shade peak period
    peak_start, peak_end = 14, 20
    ax1.axvspan(peak_start, peak_end, alpha=0.07, color=C_PRICE, label="Peak period")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        facecolor=BG3,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8,
        loc="upper left",
    )

    # ── Panel 2: Energy Flow Stacked ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    _style(ax2, "⚡ ENERGY FLOWS  (24-HOUR DISPATCH)", ylabel="Power (kW)")
    _set_hour_ticks(ax2)

    solar = np.array(log["solar"])
    load = np.array(log["load"])
    action_kw = np.array(log["action_kw"])
    charge = np.where(action_kw > 0, action_kw, 0)
    discharge = np.where(action_kw < 0, -action_kw, 0)
    grid_import = np.array(log["grid_import"])
    grid_export = np.array(log["grid_export"])  # noqa: F841

    ax2.fill_between(
        hours, 0, solar, alpha=0.7, color=C_SOLAR, label="Solar generation"
    )
    ax2.fill_between(
        hours, 0, grid_import, alpha=0.5, color=C_GRID_I, label="Grid import"
    )
    ax2.fill_between(
        hours, 0, -charge, alpha=0.6, color=C_BAT_C, label="Battery charging"
    )
    ax2.fill_between(
        hours, 0, discharge, alpha=0.6, color=C_BAT_D, label="Battery discharge"
    )
    ax2.plot(
        hours,
        load,
        color=C_LOAD,
        linewidth=2.5,
        linestyle="--",
        label="Building load",
        zorder=5,
    )
    ax2.axhline(0, color=GRID, linewidth=0.8)
    ax2.legend(
        facecolor=BG3,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8,
        loc="upper right",
        ncol=2,
    )

    # ── Panel 3: Hourly P&L ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3, "💰 HOURLY PROFIT / LOSS", "Hour", "$ Revenue")
    rewards = np.array(log["reward"])
    colors = [C_BAT_C if r >= 0 else C_BAT_D for r in rewards]
    ax3.bar(hours, rewards, color=colors, edgecolor=BG, linewidth=0.5, width=0.8)
    ax3.axhline(0, color=GRID, linewidth=1)
    _set_hour_ticks(ax3)
    ax3.set_xlabel("Hour", color=DIM, fontsize=8)

    # ── Panel 4: KPI cards ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor(BG2)
    ax4.axis("off")

    total_r = sum(log["reward"])
    total_solar = sum(solar)
    total_load = sum(load)
    self_suff = min(100, total_solar / max(total_load, 1e-9) * 100)
    bat_cycles = (
        sum(abs(np.array(log["action_kw"]))) / 2 / max(max(log["soc"] + [1e-9]), 1)
    )
    peak_import = max(grid_import) if any(grid_import) else 0

    kpis = [
        ("Daily Net Revenue", f"${total_r:+.3f}", C_BAT_C if total_r >= 0 else C_BAT_D),
        ("Solar Self-Sufficiency", f"{self_suff:.1f}%", C_SOLAR),
        ("Battery Cycles Used", f"{bat_cycles:.2f}", C_SOC),
        ("Peak Grid Import", f"{peak_import:.1f} kW", C_GRID_I),
    ]
    for i, (label, val, col) in enumerate(kpis):
        y = 0.82 - i * 0.22
        ax4.text(
            0.05,
            y,
            label,
            transform=ax4.transAxes,
            color=DIM,
            fontsize=8,
            fontfamily="monospace",
        )
        ax4.text(
            0.05,
            y - 0.08,
            val,
            transform=ax4.transAxes,
            color=col,
            fontsize=16,
            fontweight="bold",
            fontfamily="monospace",
        )

    title = f"SMART GRID DISPATCH  ·  {scenario_name}  ·  Strategy: {strategy}"
    fig.suptitle(
        title,
        color=TEXT,
        fontsize=11,
        fontfamily="monospace",
        fontweight="bold",
        y=0.96,
    )
    return fig


def comparison_chart(results: dict[str, dict]) -> Figure:
    """
    Side-by-side comparison of all strategies.
    results: {"DP": {"log": ..., "reward": ...}, "DQN": ..., ...}
    """
    strategies = list(results.keys())
    colors = {
        "DP Optimal": C_DP,
        "DQN Agent": C_DQN,
        "Rule-Based": C_NAIVE,
        "No Storage": C_NOBAT,
    }

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.5,
        wspace=0.32,
        left=0.07,
        right=0.97,
        top=0.88,
        bottom=0.08,
    )

    # ── Panel 1: SOC trajectories ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1, "BATTERY STATE OF CHARGE", "Hour", "SOC (kWh)")
    for name, data in results.items():
        log = data.get("log")
        if log and log["soc"]:
            col = colors.get(name, TEXT)
            ax1.plot(
                log["hour"], log["soc"], color=col, linewidth=2, label=name, alpha=0.9
            )
    ax1.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    _set_hour_ticks(ax1)

    # ── Panel 2: Revenue bar chart ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2, "DAILY NET REVENUE COMPARISON")
    rewards = [results[s].get("reward", 0) for s in strategies]
    cols = [colors.get(s, TEXT) for s in strategies]
    bars = ax2.bar(
        strategies, rewards, color=cols, edgecolor=BG, linewidth=0.8, width=0.5
    )
    for bar, val in zip(bars, rewards):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.002 if val >= 0 else -0.004),
            f"${val:+.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )
    ax2.axhline(0, color=GRID, linewidth=1)
    ax2.tick_params(axis="x", colors=TEXT, labelsize=8)

    # ── Panel 3: Hourly price + SOC overlay per strategy ─────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3, "PRICE vs GRID IMPORT PROFILE", "Hour", "Grid Import (kW)")
    for name, data in results.items():
        log = data.get("log")
        if log and log["grid_import"]:
            col = colors.get(name, TEXT)
            ax3.plot(
                log["hour"],
                log["grid_import"],
                color=col,
                linewidth=1.8,
                label=name,
                alpha=0.85,
            )
    ax3.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    _set_hour_ticks(ax3)

    # ── Panel 4: Cumulative revenue ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4, "CUMULATIVE REVENUE", "Hour", "$ Cumulative")
    for name, data in results.items():
        log = data.get("log")
        if log and log["reward"]:
            cum = np.cumsum(log["reward"])
            col = colors.get(name, TEXT)
            ax4.plot(log["hour"], cum, color=col, linewidth=2, label=name, alpha=0.9)
    ax4.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax4.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    _set_hour_ticks(ax4)

    fig.suptitle(
        "STRATEGY BENCHMARKING  ·  Same Scenario, Same Day",
        color=TEXT,
        fontsize=12,
        fontfamily="monospace",
        fontweight="bold",
        y=0.95,
    )
    return fig


def training_chart(state) -> Figure:
    """Live DQN training dashboard."""
    fig = plt.figure(figsize=(14, 4), facecolor=BG)
    gs = gridspec.GridSpec(
        1, 3, figure=fig, wspace=0.35, left=0.07, right=0.97, top=0.82, bottom=0.15
    )

    # Reward history
    ax1 = fig.add_subplot(gs[0])
    _style(ax1, "EPISODE REWARD", "Episode", "$ Reward")
    if state.episode_rewards:
        eps = list(range(len(state.episode_rewards)))
        ax1.plot(eps, state.episode_rewards, color=C_DQN, alpha=0.3, linewidth=0.8)
        if len(eps) > 10:
            k = max(5, len(eps) // 30)
            smooth = np.convolve(state.episode_rewards, np.ones(k) / k, "valid")
            ax1.plot(range(k - 1, len(eps)), smooth, color=C_DQN, linewidth=2.2)
        ax1.axhline(
            state.best_reward,
            color=C_SOLAR,
            linestyle="--",
            linewidth=1,
            label=f"Best: ${state.best_reward:+.3f}",
        )
        ax1.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # Rolling mean
    ax2 = fig.add_subplot(gs[1])
    _style(ax2, "ROLLING MEAN (20 ep)", "Episode", "$ Avg Reward")
    if state.mean_rewards:
        eps = list(range(len(state.mean_rewards)))
        ax2.plot(eps, state.mean_rewards, color=C_SOC, linewidth=2.0)
        ax2.axhline(0, color=GRID, linewidth=0.8, linestyle="--")

    # Progress bar representation
    ax3 = fig.add_subplot(gs[2])
    _style(ax3, "TRAINING PROGRESS")
    ax3.axis("off")
    pct = state.timestep / max(state.total_timesteps, 1)
    n_ep = len(state.episode_rewards)
    roll = state.mean_rewards[-1] if state.mean_rewards else 0
    info = [
        ("Progress", f"{pct * 100:.1f}%"),
        ("Episodes", f"{n_ep}"),
        ("Rolling Reward", f"${roll:+.3f}"),
        ("Best Episode", f"${state.best_reward:+.3f}"),
    ]
    for i, (label, val) in enumerate(info):
        y = 0.8 - i * 0.2
        ax3.text(
            0.05,
            y,
            label,
            transform=ax3.transAxes,
            color=DIM,
            fontsize=9,
            fontfamily="monospace",
        )
        ax3.text(
            0.65,
            y,
            val,
            transform=ax3.transAxes,
            color=C_SOC,
            fontsize=9,
            fontweight="bold",
            fontfamily="monospace",
        )

    fig.suptitle(
        f"DQN TRAINING  ·  {state.timestep:,}/{state.total_timesteps:,} steps",
        color=TEXT,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
    )
    return fig


def scenario_price_chart(
    prices: np.ndarray, solar: np.ndarray, load: np.ndarray, scenario_name: str = ""
) -> Figure:
    """Preview chart for the Scenario Lab."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    data = [
        (prices, "PRICE (¢/kWh)", C_PRICE),
        (solar, "SOLAR GEN (kW)", C_SOLAR),
        (load, "BUILDING LOAD (kW)", C_LOAD),
    ]
    for ax, (vals, title, col) in zip(axes, data):
        _style(ax, title, ylabel="")
        ax.fill_between(HOURS, vals, alpha=0.3, color=col)
        ax.plot(HOURS, vals, color=col, linewidth=2.0)
        _set_hour_ticks(ax)

    fig.suptitle(
        f"SCENARIO PREVIEW  ·  {scenario_name}",
        color=TEXT,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
    )
    fig.tight_layout(pad=2)
    return fig


def _empty_fig(msg: str = "") -> Figure:
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
    )
    ax.axis("off")
    return fig
