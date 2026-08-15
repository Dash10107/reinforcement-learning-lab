"""
All visualisation for the MAB Banner Optimizer.
Dark ad-tech dashboard aesthetic.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bandits.agents import AGENT_COLORS, BanditAgent, ThompsonSampling
from matplotlib import gridspec
from matplotlib.figure import Figure
from scipy.stats import beta as beta_dist

# ── Design tokens ─────────────────────────────────────────────────────────────
BG = "#0a0d14"
BG2 = "#0f1520"
BG3 = "#161e2e"
GRID = "#1e2a3d"
TEXT = "#e2e8f0"
DIM = "#4a6080"
UP = "#00d4aa"
DOWN = "#ef4444"
GOLD = "#faad14"
PURP = "#a855f7"


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


def _agent_color(name: str) -> str:
    return AGENT_COLORS.get(name, TEXT)


# ── 1. Algorithm face-off (main comparison) ───────────────────────────────────


def comparison_dashboard(
    agents: dict[str, BanditAgent],
    arm_names: list[str],
    optimal_ev: float,
) -> Figure:
    """
    4-panel comparison:
      Top-left:   Cumulative revenue per algorithm
      Top-right:  Cumulative regret per algorithm
      Bottom-left:  Final arm pull distribution (grouped bar)
      Bottom-right: Final revenue & regret summary bar
    """
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.50,
        wspace=0.32,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.08,
    )

    steps = max(len(a.reward_history) for a in agents.values())
    xs = np.arange(1, steps + 1)

    # ── Cumulative revenue ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1, "CUMULATIVE REVENUE ($)", "Impression", "$")
    for name, agent in agents.items():
        cum = agent.cumulative_rewards
        ax1.plot(
            range(1, len(cum) + 1),
            cum,
            color=_agent_color(name),
            linewidth=2.0,
            label=name,
            alpha=0.9,
        )
    # Optimal line
    opt = np.arange(1, steps + 1) * optimal_ev
    ax1.plot(
        xs, opt, color=GRID, linewidth=1.2, linestyle="--", alpha=0.8, label="Optimal"
    )
    ax1.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=7)

    # ── Cumulative regret ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2, "CUMULATIVE REGRET ($)", "Impression", "$")
    for name, agent in agents.items():
        rh = agent.regret_history
        ax2.plot(
            range(1, len(rh) + 1),
            rh,
            color=_agent_color(name),
            linewidth=2.0,
            label=name,
            alpha=0.9,
        )
    ax2.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    ax2.set_ylim(bottom=0)

    # ── Arm pull distribution ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3, "ARM PULL DISTRIBUTION (%)", "Banner", "%")
    n_arms = len(arm_names)
    n_agents = len(agents)
    width = 0.8 / n_agents
    x_pos = np.arange(n_arms)

    for i, (name, agent) in enumerate(agents.items()):
        offset = (i - n_agents / 2 + 0.5) * width
        pcts = agent.arm_pull_pcts
        ax3.bar(
            x_pos + offset,
            pcts,
            width=width * 0.9,
            color=_agent_color(name),
            alpha=0.8,
            edgecolor=BG,
            linewidth=0.5,
            label=name,
        )

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(arm_names, color=DIM, fontsize=7, rotation=15, ha="right")
    ax3.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=7)

    # ── Final summary ─────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4, "FINAL PERFORMANCE SUMMARY")
    ax4.axis("off")

    names = list(agents.keys())
    revenues = [a.total_reward for a in agents.values()]
    regrets = [a.cumulative_regret for a in agents.values()]

    best_rev = max(revenues)
    y_base = 0.95
    ax4.text(
        0.03,
        y_base,
        "Algorithm",
        transform=ax4.transAxes,
        color=DIM,
        fontsize=8,
        fontfamily="monospace",
    )
    ax4.text(
        0.52,
        y_base,
        "Revenue",
        transform=ax4.transAxes,
        color=DIM,
        fontsize=8,
        fontfamily="monospace",
    )
    ax4.text(
        0.76,
        y_base,
        "Regret",
        transform=ax4.transAxes,
        color=DIM,
        fontsize=8,
        fontfamily="monospace",
    )

    for i, (name, rev, reg) in enumerate(zip(names, revenues, regrets)):
        y = y_base - 0.14 * (i + 1)
        col = _agent_color(name)
        crown = " 👑" if rev == best_rev else ""
        ax4.text(
            0.03,
            y,
            f"● {name}{crown}",
            transform=ax4.transAxes,
            color=col,
            fontsize=8,
            fontfamily="monospace",
            fontweight="bold",
        )
        ax4.text(
            0.52,
            y,
            f"${rev:,.1f}",
            transform=ax4.transAxes,
            color=UP if rev == best_rev else TEXT,
            fontsize=8,
            fontfamily="monospace",
        )
        ax4.text(
            0.76,
            y,
            f"${reg:,.1f}",
            transform=ax4.transAxes,
            color=DOWN if reg == max(regrets) else TEXT,
            fontsize=8,
            fontfamily="monospace",
        )

    n = steps
    fig.suptitle(
        f"ALGORITHM FACE-OFF  ·  {n:,} impressions  ·  "
        f"{n_arms} banners  ·  Optimal EV: ${optimal_ev:.3f}/impression",
        color=TEXT,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
        y=0.97,
    )
    return fig


# ── 2. Belief / state visualisation ──────────────────────────────────────────


def belief_chart(
    agent: BanditAgent,
    arm_names: list[str],
    true_ctrs: list[float],
    step: int,
) -> Figure:
    """
    For Thompson Sampling: Beta PDFs per arm.
    For others: Q-value bar chart with true CTR overlay.
    """
    n = len(arm_names)
    arm_colors = plt.cm.Set2(np.linspace(0, 1, n))

    if isinstance(agent, ThompsonSampling):
        fig, axes = plt.subplots(1, n, figsize=(min(14, n * 3), 4), facecolor=BG)
        fig.patch.set_facecolor(BG)
        if n == 1:
            axes = [axes]

        x = np.linspace(0, 1, 500)
        for i, (ax, name, col) in enumerate(zip(axes, arm_names, arm_colors)):
            _style(ax, name[:12], "CTR", "Density")
            a, b = agent.alpha[i], agent.beta[i]
            pdf = beta_dist.pdf(x, a, b)
            ax.fill_between(x, pdf, alpha=0.3, color=col)
            ax.plot(x, pdf, color=col, linewidth=2.0)
            ax.axvline(
                true_ctrs[i],
                color=DOWN,
                linewidth=1.5,
                linestyle="--",
                label=f"True: {true_ctrs[i]:.3f}",
            )
            ax.axvline(
                a / (a + b),
                color=UP,
                linewidth=1.5,
                linestyle=":",
                label=f"Est: {a / (a + b):.3f}",
            )
            pulls = agent.counts[i]
            ax.text(
                0.97,
                0.97,
                f"n={pulls}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=DIM,
                fontsize=7.5,
                fontfamily="monospace",
            )
            ax.legend(
                facecolor=BG3,
                edgecolor=GRID,
                labelcolor=TEXT,
                fontsize=7,
                loc="upper left",
            )

        fig.suptitle(
            f"THOMPSON SAMPLING — POSTERIOR BELIEFS  (Step {step})",
            color=TEXT,
            fontsize=9,
            fontfamily="monospace",
            fontweight="bold",
        )
        fig.tight_layout(pad=1.5)

    else:
        fig, ax = plt.subplots(figsize=(min(12, n * 1.8 + 2), 5), facecolor=BG)
        fig.patch.set_facecolor(BG)
        _style(
            ax,
            f"{agent.name.upper()}  —  Q-VALUES  (Step {step})",
            "Banner",
            "Estimated Value ($)",
        )

        x = np.arange(n)
        col_list = [arm_colors[i] for i in range(n)]
        bars = ax.bar(
            x, agent.values, color=col_list, edgecolor=BG, linewidth=0.8, alpha=0.85
        )

        # True expected values overlay
        [
            c * r
            for c, r in zip(
                true_ctrs, [agent.values[i] / max(true_ctrs[i], 1e-9) for i in range(n)]
            )
        ]
        # Use arm pulls to estimate revenue separately — just show true CTR as line
        for i, (ctr, bar) in enumerate(zip(true_ctrs, bars)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(agent.values) * 0.01,
                f"{agent.values[i]:.3f}\nn={agent.counts[i]}",
                ha="center",
                color=TEXT,
                fontsize=7.5,
                fontfamily="monospace",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(arm_names, color=DIM, fontsize=8, rotation=15, ha="right")
        fig.tight_layout(pad=1.5)

    return fig


# ── 3. Campaign analytics (single agent deep-dive) ────────────────────────────


def campaign_analytics(
    agent: BanditAgent,
    arm_names: list[str],
    true_ctrs: list[float],
    revenues: list[float],
) -> Figure:
    """3-panel: per-arm stats, reward timeline, arm selection heatmap."""
    n = len(arm_names)
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.52,
        wspace=0.32,
        left=0.08,
        right=0.97,
        top=0.90,
        bottom=0.08,
    )
    arm_colors = plt.cm.Set2(np.linspace(0, 1, n))

    # ── Per-arm performance table ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1, "PER-BANNER STATS")
    ax1.axis("off")

    headers = ["Banner", "Pulls", "Est. CTR", "True CTR", "Revenue"]
    y0 = 0.93
    ax1.text(
        0.02,
        y0,
        "  ".join(headers),
        transform=ax1.transAxes,
        color=DIM,
        fontsize=7.5,
        fontfamily="monospace",
    )
    ax1.axhline(y0 - 0.04, color=GRID, linewidth=0.8)

    for i, name in enumerate(arm_names):
        y = y0 - 0.11 * (i + 1)
        pull = agent.counts[i]
        est = agent.values[i] / max(revenues[i], 1e-9) if revenues[i] > 0 else 0
        rev = agent.counts[i] * agent.values[i]
        col = arm_colors[i]
        is_best = true_ctrs[i] * revenues[i] == max(
            c * r for c, r in zip(true_ctrs, revenues)
        )
        ax1.text(
            0.02,
            y,
            f"{'★ ' if is_best else '  '}{name[:14]}",
            transform=ax1.transAxes,
            color=col,
            fontsize=7.5,
            fontfamily="monospace",
        )
        ax1.text(
            0.36,
            y,
            f"{pull:>5}",
            transform=ax1.transAxes,
            color=TEXT,
            fontsize=7.5,
            fontfamily="monospace",
        )
        ax1.text(
            0.50,
            y,
            f"{est:.3f}",
            transform=ax1.transAxes,
            color=UP if abs(est - true_ctrs[i]) < 0.03 else TEXT,
            fontsize=7.5,
            fontfamily="monospace",
        )
        ax1.text(
            0.65,
            y,
            f"{true_ctrs[i]:.3f}",
            transform=ax1.transAxes,
            color=DIM,
            fontsize=7.5,
            fontfamily="monospace",
        )
        ax1.text(
            0.80,
            y,
            f"${rev:.1f}",
            transform=ax1.transAxes,
            color=UP,
            fontsize=7.5,
            fontfamily="monospace",
        )

    # ── Reward smoothed rolling average ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2, "ROLLING REWARD (window=50)", "Step", "Avg Reward ($)")
    rh = np.array(agent.reward_history)
    if len(rh) >= 50:
        rolling = np.convolve(rh, np.ones(50) / 50, "valid")
        ax2.fill_between(range(49, len(rh)), rolling, alpha=0.2, color=UP)
        ax2.plot(range(49, len(rh)), rolling, color=UP, linewidth=1.8)
    ax2.axhline(
        np.mean(rh) if len(rh) else 0,
        color=GOLD,
        linewidth=1,
        linestyle="--",
        alpha=0.6,
        label="Mean",
    )
    ax2.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # ── Arm selection over time (stacked area) ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    _style(ax3, "ARM SELECTION TIMELINE  (100-step windows)", "Window", "% Impressions")

    ah = np.array(agent.arm_history)
    wsize = max(1, len(ah) // 50)
    wins = len(ah) // wsize
    if wins > 0:
        fracs = np.zeros((n, wins))
        for w in range(wins):
            chunk = ah[w * wsize : (w + 1) * wsize]
            for arm_i in range(n):
                fracs[arm_i, w] = (chunk == arm_i).mean() * 100

        bottoms = np.zeros(wins)
        for i in range(n):
            ax3.bar(
                range(wins),
                fracs[i],
                bottom=bottoms,
                color=arm_colors[i],
                edgecolor=BG,
                linewidth=0,
                alpha=0.85,
                label=arm_names[i],
            )
            bottoms += fracs[i]

        ax3.set_xlim(-0.5, wins + 0.5)
        ax3.set_ylim(0, 101)
        ax3.legend(
            facecolor=BG3,
            edgecolor=GRID,
            labelcolor=TEXT,
            fontsize=7,
            loc="upper right",
            ncol=min(n, 4),
        )

    fig.suptitle(
        f"{agent.name.upper()}  ·  {len(agent.arm_history):,} impressions  ·  "
        f"Total Revenue: ${agent.total_reward:,.2f}  ·  "
        f"Regret: ${agent.cumulative_regret:,.2f}",
        color=TEXT,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
        y=0.97,
    )
    return fig


# ── 4. Scenario preview ───────────────────────────────────────────────────────


def scenario_preview(
    arm_names: list[str], true_ctrs: list[float], revenues: list[float]
) -> Figure:
    """Quick horizontal bar chart of expected value per banner."""
    n = len(arm_names)
    fig, ax = plt.subplots(figsize=(10, max(3, n * 0.7 + 1)), facecolor=BG)
    fig.patch.set_facecolor(BG)
    _style(
        ax, "BANNER EXPECTED VALUE  (CTR × Revenue)", "Expected Value ($/impression)"
    )

    evs = [c * r for c, r in zip(true_ctrs, revenues)]
    best = max(evs)
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    bars = ax.barh(
        arm_names, evs, color=colors, edgecolor=BG, linewidth=0.5, alpha=0.85
    )
    for bar, ev, name in zip(bars, evs, arm_names):
        ax.text(
            bar.get_width() + best * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"${ev:.4f}",
            va="center",
            color=UP if ev == best else TEXT,
            fontsize=8.5,
            fontweight="bold" if ev == best else "normal",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(arm_names, color=TEXT, fontsize=8.5)
    ax.tick_params(axis="x", colors=DIM)
    ax.set_xlim(0, best * 1.25)

    # Mark optimal
    opt_idx = np.argmax(evs)
    ax.get_yticklabels()[opt_idx].set_color(UP)
    ax.get_yticklabels()[opt_idx].set_fontweight("bold")

    fig.tight_layout(pad=1.5)
    return fig


# ── 5. Learner mode step chart ────────────────────────────────────────────────


def learner_step_chart(
    agent: BanditAgent,
    arm_names: list[str],
    true_ctrs: list[float],
    revenues: list[float],
    last_arm: int,
    last_reward: float,
) -> Figure:
    """Compact chart for the interactive learner — beliefs + pull history."""
    n = len(arm_names)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    fig.patch.set_facecolor(BG)
    arm_colors = plt.cm.Set2(np.linspace(0, 1, n))

    # Left: beliefs
    if isinstance(agent, ThompsonSampling):
        _style(ax1, "POSTERIOR BELIEFS  (Beta PDFs)", "CTR", "Density")
        x = np.linspace(0, 1, 300)
        for i, name in enumerate(arm_names):
            a, b = agent.alpha[i], agent.beta[i]
            pdf = beta_dist.pdf(x, a, b)
            alpha_v = 0.9 if i == last_arm else 0.45
            ax1.plot(
                x,
                pdf,
                color=arm_colors[i],
                linewidth=2.0 if i == last_arm else 1.2,
                alpha=alpha_v,
                label=name[:10],
            )
            ax1.axvline(
                true_ctrs[i],
                color=arm_colors[i],
                linewidth=0.8,
                linestyle=":",
                alpha=0.5,
            )
        ax1.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    else:
        _style(ax1, f"{agent.name}  —  Q-VALUES", "Banner", "Value ($)")
        cols = [arm_colors[i] for i in range(n)]
        cols[last_arm] = UP  # highlight last chosen
        ax1.bar(arm_names, agent.values, color=cols, edgecolor=BG, alpha=0.85)
        ax1.tick_params(axis="x", colors=DIM, labelsize=7)
        for i, v in enumerate(agent.values):
            ax1.text(
                i,
                v + max(agent.values) * 0.01,
                f"{v:.3f}",
                ha="center",
                color=TEXT,
                fontsize=7,
            )

    # Right: pull counts
    _style(ax2, "PULL COUNTS PER BANNER", "Banner", "# Pulls")
    cols = [arm_colors[i] for i in range(n)]
    cols[last_arm] = UP
    ax2.bar(arm_names, agent.counts, color=cols, edgecolor=BG, alpha=0.85)
    ax2.tick_params(axis="x", colors=DIM, labelsize=7)
    for i, c in enumerate(agent.counts):
        ax2.text(i, c + 0.3, str(c), ha="center", color=TEXT, fontsize=7.5)

    fig.suptitle(
        f"Step {agent.t}  ·  Chose: {arm_names[last_arm]}  ·  "
        f"Reward: ${last_reward:.2f}  ·  "
        f"Total: ${agent.total_reward:.2f}  ·  Regret: ${agent.cumulative_regret:.2f}",
        color=UP if last_reward > 0 else DOWN,
        fontsize=9,
        fontfamily="monospace",
        fontweight="bold",
    )
    fig.tight_layout(pad=1.5)
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
        wrap=True,
    )
    ax.axis("off")
    return fig
