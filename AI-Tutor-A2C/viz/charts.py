"""
All matplotlib visualisations for the AI Tutor dashboard.
Dark glassmorphic theme consistent with the UI.
"""

from __future__ import annotations

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import PIL.Image
from core.environment import SUBJECT_COLORS, SUBJECTS
from matplotlib import gridspec
from matplotlib.figure import Figure

BG = "#0a0b10"
BG2 = "#111318"
GRID = "#1e2130"
TEXT = "#f8fafc"
DIM = "#64748b"
ACC = "#6366f1"


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=8, length=3)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(
            title,
            color=TEXT,
            fontsize=9,
            pad=8,
            fontfamily="monospace",
            fontweight="bold",
        )
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=8)


def _fig_to_pil(fig: Figure) -> PIL.Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return PIL.Image.open(buf).convert("RGB")


# ── 1. Learning path trajectory ───────────────────────────────────────────────


def trajectory_chart(history: list[dict]) -> PIL.Image.Image:
    """Line chart showing each subject's proficiency over simulation steps."""
    if not history:
        return _empty("Run a simulation to see the learning trajectory.")

    steps = [h["step"] for h in history]
    states = np.array([h["state"] for h in history])  # (T, 5)
    actions = [h["action"] for h in history]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}, facecolor=BG
    )
    fig.patch.set_facecolor(BG)

    # ── Panel 1: proficiency lines ─────────────────────────────────────────
    _style(ax1, "LEARNING TRAJECTORY  —  Proficiency Over Steps", "", "Proficiency (%)")
    for i, (name, col) in enumerate(zip(SUBJECTS, SUBJECT_COLORS)):
        ax1.plot(steps, states[:, i], color=col, linewidth=2.2, label=name, alpha=0.9)
        # Mark final value
        ax1.scatter(steps[-1], states[-1, i], color=col, s=40, zorder=5)

    ax1.axhline(
        98, color=TEXT, linewidth=0.7, linestyle="--", alpha=0.3, label="Mastery (98%)"
    )
    ax1.set_ylim(-2, 105)
    ax1.legend(
        facecolor=BG2,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=7,
        loc="lower right",
        ncol=2,
    )
    ax1.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
    ax1.tick_params(labelbottom=False)

    # ── Panel 2: action timeline ───────────────────────────────────────────
    ax2.set_facecolor(BG2)
    for sp in ax2.spines.values():
        sp.set_color(GRID)
    ax2.tick_params(colors=DIM, labelsize=7)

    for step, action in zip(steps, actions):
        ax2.bar(
            step,
            1,
            color=SUBJECT_COLORS[action],
            edgecolor=BG,
            linewidth=0.3,
            alpha=0.85,
            width=0.9,
        )

    ax2.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Step", color=DIM, fontsize=8)
    ax2.set_title(
        "AGENT FOCUS  (colour = subject chosen)",
        color=TEXT,
        fontsize=8,
        pad=5,
        fontfamily="monospace",
        fontweight="bold",
    )

    fig.tight_layout(pad=2.0)
    return _fig_to_pil(fig)


# ── 2. Policy probability bars ────────────────────────────────────────────────


def policy_bars(probs: list[float], current_state: list[float]) -> PIL.Image.Image:
    """Two-panel: current proficiency + action probabilities side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
    fig.patch.set_facecolor(BG)

    _style(ax1, "CURRENT PROFICIENCY", "", "%")
    _style(ax2, "A2C ACTION PROBABILITIES", "", "Probability")

    names = [s[:8] for s in SUBJECTS]
    y_pos = np.arange(len(SUBJECTS))

    # Proficiency
    bars1 = ax1.barh(
        y_pos,
        current_state,
        color=SUBJECT_COLORS,
        edgecolor=BG,
        linewidth=0.5,
        alpha=0.85,
    )
    ax1.axvline(98, color=TEXT, linewidth=0.8, linestyle="--", alpha=0.3)
    ax1.set_xlim(0, 108)
    for bar, val in zip(bars1, current_state):
        ax1.text(
            val + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            color=TEXT,
            fontsize=8,
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, color=TEXT, fontsize=8.5)

    # Action probs
    best = int(np.argmax(probs))
    bar_colors = [
        SUBJECT_COLORS[i] if i == best else "#334155" for i in range(len(SUBJECTS))
    ]
    bars2 = ax2.barh(
        y_pos,
        [p * 100 for p in probs],
        color=bar_colors,
        edgecolor=BG,
        linewidth=0.5,
        alpha=0.85,
    )
    ax2.set_xlim(0, 108)
    for bar, val in zip(bars2, probs):
        ax2.text(
            val * 100 + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            color=TEXT,
            fontsize=8,
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, color=TEXT, fontsize=8.5)

    fig.tight_layout(pad=2.0)
    return _fig_to_pil(fig)


# ── 3. Episode analytics (focus distribution + reward) ───────────────────────


def episode_analytics(history: list[dict]) -> PIL.Image.Image:
    """Pie chart of attention allocation + cumulative reward curve."""
    if not history:
        return _empty("Run a simulation to see episode analytics.")

    fig = plt.figure(figsize=(12, 5), facecolor=BG)
    gs = gridspec.GridSpec(
        1, 2, figure=fig, wspace=0.35, left=0.07, right=0.97, top=0.85, bottom=0.1
    )

    # ── Attention pie ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG)
    actions = [h["action"] for h in history]
    counts = np.bincount(actions, minlength=len(SUBJECTS))
    pcts = counts / counts.sum() * 100

    wedges, texts, autotexts = ax1.pie(  # noqa: RUF059
        pcts,
        labels=[f"{s[:8]}\n{p:.0f}%" for s, p in zip(SUBJECTS, pcts)],
        colors=SUBJECT_COLORS,
        autopct="",
        startangle=90,
        wedgeprops={"edgecolor": BG, "linewidth": 2},
    )
    for t in texts:
        t.set_color(TEXT)
        t.set_fontsize(7.5)
    ax1.set_title(
        "AGENT ATTENTION ALLOCATION",
        color=TEXT,
        fontsize=9,
        fontfamily="monospace",
        fontweight="bold",
        pad=10,
    )

    # ── Reward curve ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    _style(ax2, "CUMULATIVE REWARD", "Step", "$")
    rewards = [h["reward"] for h in history]
    cum_r = np.cumsum(rewards)
    steps = [h["step"] for h in history]
    ax2.fill_between(steps, cum_r, alpha=0.15, color=ACC)
    ax2.plot(steps, cum_r, color=ACC, linewidth=2.2)
    ax2.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)

    fig.suptitle(
        f"EPISODE ANALYSIS  ·  {len(history)} steps  ·  "
        f"Total reward: {sum(rewards):.2f}  ·  "
        f"Final avg proficiency: {np.mean(history[-1]['state']):.1f}%",
        color=TEXT,
        fontsize=9,
        fontfamily="monospace",
        fontweight="bold",
        y=0.98,
    )
    return _fig_to_pil(fig)


# ── 4. Training dashboard ─────────────────────────────────────────────────────


def training_chart(train_state) -> PIL.Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Reward history
    ax1 = axes[0]
    _style(ax1, "EPISODE REWARD", "Episode", "Reward")
    if train_state.ep_rewards:
        eps = list(range(len(train_state.ep_rewards)))
        ax1.plot(eps, train_state.ep_rewards, color=ACC, alpha=0.3, linewidth=0.8)
        if len(eps) > 20:
            k = max(5, len(eps) // 30)
            smooth = np.convolve(train_state.ep_rewards, np.ones(k) / k, "valid")
            ax1.plot(range(k - 1, len(eps)), smooth, color=ACC, linewidth=2.2)

    # Progress info
    ax2 = axes[1]
    _style(ax2, "TRAINING PROGRESS")
    ax2.axis("off")
    pct = train_state.timestep / max(train_state.total, 1) * 100
    n_ep = len(train_state.ep_rewards)
    roll = float(np.mean(train_state.ep_rewards[-20:])) if train_state.ep_rewards else 0
    info = [
        ("Progress", f"{pct:.1f}%"),
        ("Steps", f"{train_state.timestep:,}"),
        ("Episodes", f"{n_ep}"),
        ("Rolling Reward", f"{roll:.3f}"),
    ]
    for i, (label, val) in enumerate(info):
        y = 0.80 - i * 0.20
        ax2.text(
            0.05,
            y,
            label,
            transform=ax2.transAxes,
            color=DIM,
            fontsize=9,
            fontfamily="monospace",
        )
        ax2.text(
            0.60,
            y,
            val,
            transform=ax2.transAxes,
            color=ACC,
            fontsize=9,
            fontweight="bold",
            fontfamily="monospace",
        )

    fig.suptitle(
        "A2C TRAINING DASHBOARD",
        color=TEXT,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
    )
    fig.tight_layout()
    return _fig_to_pil(fig)


def _empty(msg: str) -> PIL.Image.Image:
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
        fontsize=11,
        fontfamily="monospace",
    )
    ax.axis("off")
    return _fig_to_pil(fig)
