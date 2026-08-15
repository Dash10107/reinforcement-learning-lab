"""
All matplotlib figure generation for the dashboard.
Every function returns a plt.Figure — caller closes or passes to Gradio.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from core.mission import MissionResult, EpisodeResult

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#030b1a"
BG2      = "#060f1e"
GRID     = "#0d2540"
ACCENT   = "#4fb3ff"
GREEN    = "#2ddb7c"
AMBER    = "#f5a623"
RED      = "#ff4d6d"
PURPLE   = "#c77dff"
TEXT     = "#c8ddf0"
DIM      = "#3a6080"

EP_COLORS = [ACCENT, GREEN, AMBER, PURPLE, "#ff9f1c", "#e9c46a", "#f4a261"]


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, pad=8, fontfamily="monospace")
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=8)


def mission_overview(mission: MissionResult) -> plt.Figure:
    """4-panel summary: bar chart, trajectory, reward curves, throttle."""
    n = len(mission.episodes)
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    # ── Panel 1: Episode rewards bar ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "EPISODE REWARDS", "Episode", "Score")
    labels = [f"#{e.episode_idx+1}" for e in mission.episodes]
    colors = [GREEN if r >= 150 else (AMBER if r >= 0 else RED) for r in mission.rewards]
    bars = ax1.bar(labels, mission.rewards, color=colors, edgecolor=BG, linewidth=0.8)
    ax1.axhline(200, color=GREEN, linestyle="--", linewidth=1, alpha=0.5, label="Perfect (200)")
    ax1.axhline(150, color=ACCENT, linestyle="--", linewidth=1, alpha=0.5, label="Success (150)")
    ax1.axhline(0, color=RED, linestyle="--", linewidth=1, alpha=0.3)
    ax1.legend(fontsize=7, facecolor=BG2, edgecolor=GRID, labelcolor=DIM)
    for bar, val in zip(bars, mission.rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f"{val:.0f}", ha="center", va="bottom", color=TEXT, fontsize=8)

    # ── Panel 2: 2-D flight trajectory ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "FLIGHT TRAJECTORIES", "X Position", "Altitude")
    for i, ep in enumerate(mission.episodes):
        col = EP_COLORS[i % len(EP_COLORS)]
        # colour-map by altitude for gradient effect
        points = np.array([ep.xs, ep.ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=col, linewidth=1.2, alpha=0.7)
        ax2.add_collection(lc)
        # landing marker
        ax2.scatter(ep.xs[-1], ep.ys[-1],
                    marker=("*" if ep.total_reward >= 150 else "x"),
                    s=80, color=col, zorder=5)
    ax2.autoscale()
    ax2.axhline(0, color=GRID, linewidth=1)
    # Legend patches
    patches = [mpatches.Patch(color=EP_COLORS[i % len(EP_COLORS)],
                               label=f"#{e.episode_idx+1} {e.status_emoji}")
               for i, e in enumerate(mission.episodes)]
    ax2.legend(handles=patches, fontsize=7, facecolor=BG2,
               edgecolor=GRID, labelcolor=DIM, loc="upper right")

    # ── Panel 3: Cumulative reward over steps ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "CUMULATIVE REWARD", "Step", "Reward")
    for i, ep in enumerate(mission.episodes):
        col = EP_COLORS[i % len(EP_COLORS)]
        ax3.plot(ep.cumulative_rewards, color=col, linewidth=1.5,
                 label=f"#{ep.episode_idx+1}", alpha=0.85)
    ax3.axhline(0, color=RED, linestyle="--", linewidth=0.8, alpha=0.4)
    ax3.legend(fontsize=7, facecolor=BG2, edgecolor=GRID, labelcolor=DIM)

    # ── Panel 4: Engine throttle timeline ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "ENGINE THROTTLE — BEST EPISODE", "Step", "Throttle")
    best = mission.best
    steps = range(len(best.steps))
    ax4.fill_between(steps, 0, best.main_throttle,
                     color=ACCENT, alpha=0.35, label="Main Engine")
    ax4.plot(steps, best.main_throttle, color=ACCENT, linewidth=1.2)
    ax4.fill_between(steps, 0, best.lateral_throttle,
                     color=AMBER, alpha=0.25, label="Lateral Thrusters")
    ax4.plot(steps, best.lateral_throttle, color=AMBER, linewidth=1.0)
    ax4.axhline(0, color=GRID, linewidth=0.8)
    ax4.set_ylim(-1.1, 1.1)
    ax4.legend(fontsize=7, facecolor=BG2, edgecolor=GRID, labelcolor=DIM)

    # ── Figure title ─────────────────────────────────────────────────────────
    sr = mission.success_rate * 100
    fig.suptitle(
        f"MISSION REPORT  ·  {n} episodes  ·  "
        f"Avg {mission.avg_reward:+.1f}  ·  Success {sr:.0f}%",
        color=TEXT, fontsize=12, fontfamily="monospace", y=0.96,
    )
    return fig


def single_episode_detail(ep: EpisodeResult) -> plt.Figure:
    """6-panel deep-dive for one episode."""
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38,
                           left=0.07, right=0.97, top=0.88, bottom=0.08)

    steps = list(range(len(ep.steps)))

    # Trajectory
    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax, "TRAJECTORY", "X", "Y")
    ax.plot(ep.xs, ep.ys, color=ACCENT, linewidth=1.5)
    ax.scatter(ep.xs[0], ep.ys[0], s=60, color=GREEN, zorder=5, label="Start")
    ax.scatter(ep.xs[-1], ep.ys[-1], s=80,
               marker="*" if ep.total_reward >= 150 else "x",
               color=GREEN if ep.total_reward >= 150 else RED, zorder=5, label="End")
    ax.axhline(0, color=GRID, linewidth=1)
    ax.legend(fontsize=7, facecolor=BG2, edgecolor=GRID, labelcolor=DIM)

    # Cumulative reward
    ax = fig.add_subplot(gs[0, 1])
    _style_ax(ax, "CUMULATIVE REWARD", "Step", "Reward")
    cum = ep.cumulative_rewards
    ax.fill_between(steps, 0, cum,
                    color=GREEN if ep.total_reward >= 150 else RED, alpha=0.2)
    ax.plot(steps, cum, color=GREEN if ep.total_reward >= 150 else RED, linewidth=1.5)
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")

    # Altitude over time
    ax = fig.add_subplot(gs[0, 2])
    _style_ax(ax, "ALTITUDE", "Step", "Y")
    ax.fill_between(steps, 0, ep.ys, color=ACCENT, alpha=0.15)
    ax.plot(steps, ep.ys, color=ACCENT, linewidth=1.5)
    ax.axhline(0, color=RED, linewidth=1, linestyle="--", alpha=0.5)

    # Angle
    ax = fig.add_subplot(gs[1, 0])
    _style_ax(ax, "BODY ANGLE", "Step", "Degrees")
    ax.fill_between(steps, 0, ep.angles, color=AMBER, alpha=0.2)
    ax.plot(steps, ep.angles, color=AMBER, linewidth=1.3)
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")

    # Main throttle
    ax = fig.add_subplot(gs[1, 1])
    _style_ax(ax, "MAIN ENGINE", "Step", "Throttle")
    ax.fill_between(steps, 0, ep.main_throttle, color=ACCENT, alpha=0.3)
    ax.plot(steps, ep.main_throttle, color=ACCENT, linewidth=1.2)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color=GRID, linewidth=0.8)

    # Lateral throttle
    ax = fig.add_subplot(gs[1, 2])
    _style_ax(ax, "LATERAL THRUSTERS", "Step", "Throttle")
    ax.fill_between(steps, 0, ep.lateral_throttle, color=PURPLE, alpha=0.3)
    ax.plot(steps, ep.lateral_throttle, color=PURPLE, linewidth=1.2)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color=GRID, linewidth=0.8)

    fig.suptitle(
        f"EPISODE {ep.episode_idx+1} DEEP-DIVE  ·  "
        f"{ep.status_emoji} {ep.status}  ·  Score: {ep.total_reward:+.1f}  ·  "
        f"{len(ep.steps)} steps",
        color=TEXT, fontsize=11, fontfamily="monospace", y=0.95,
    )
    return fig


def training_dashboard(state) -> plt.Figure:
    """Live training metrics: reward history + losses + entropy."""
    fig = plt.figure(figsize=(14, 5), facecolor=BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                           left=0.06, right=0.97, top=0.85, bottom=0.12)

    # Reward curve
    ax = fig.add_subplot(gs[0])
    _style_ax(ax, "EPISODE REWARD", "Episode", "Reward")
    if state.episode_rewards:
        eps = list(range(len(state.episode_rewards)))
        ax.plot(eps, state.episode_rewards, color=ACCENT, linewidth=0.8, alpha=0.4)
        if len(eps) > 20:
            k = max(5, len(eps) // 30)
            smooth = np.convolve(state.episode_rewards, np.ones(k)/k, "valid")
            ax.plot(range(k-1, len(eps)), smooth, color=ACCENT, linewidth=2)
        ax.axhline(200, color=GREEN, linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(150, color=AMBER, linestyle="--", linewidth=1, alpha=0.5)

    # Losses
    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2, "ACTOR / CRITIC LOSS", "Log Step", "Loss")
    if state.log_steps:
        ax2.plot(state.log_steps, state.actor_losses, color=ACCENT,
                 linewidth=1.5, label="Actor")
        ax2.plot(state.log_steps, state.critic_losses, color=AMBER,
                 linewidth=1.5, label="Critic")
        ax2.legend(fontsize=7, facecolor=BG2, edgecolor=GRID, labelcolor=DIM)

    # Entropy coef
    ax3 = fig.add_subplot(gs[2])
    _style_ax(ax3, "ENTROPY COEFFICIENT", "Log Step", "α")
    if state.log_steps:
        ax3.plot(state.log_steps, state.ent_coefs, color=PURPLE, linewidth=1.5)
        ax3.axhline(0, color=GRID, linewidth=0.8, linestyle="--")

    n_ep = len(state.episode_rewards)
    best = state.best_reward
    fig.suptitle(
        f"SAC TRAINING  ·  {state.timestep:,}/{state.total_timesteps:,} steps  ·  "
        f"{n_ep} episodes  ·  Best: {best:+.1f}",
        color=TEXT, fontsize=10, fontfamily="monospace",
    )
    return fig


def empty_figure(message: str = "Run a mission to see charts.") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", color=DIM,
            fontsize=13, fontfamily="monospace")
    ax.axis("off")
    return fig
