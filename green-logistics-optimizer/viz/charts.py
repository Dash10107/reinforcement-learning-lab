"""
All matplotlib visualisations for the Green Logistics Optimizer.
Dark cyberpunk-industrial aesthetic.
"""

from __future__ import annotations
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import PIL.Image

from core.agents import RouteResult
from core.env import GreenCityEnv

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d1117"
BG2     = "#161b22"
GRID_L  = "#21262d"
TEXT    = "#e6edf3"
DIM     = "#8b949e"
ACCENT  = "#00e676"   # DQN  — neon green
RED     = "#f44336"   # Greedy — red
BLUE    = "#29b6f6"   # A*   — blue
WARN    = "#ff6d00"   # congestion

AGENT_COLORS = {
    "DQN Agent":              ACCENT,
    "DQN (→ Greedy fallback)": ACCENT,
    "Greedy":                 RED,
    "A* Optimal":             BLUE,
}


def _pil(fig: Figure) -> PIL.Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return PIL.Image.open(buf).copy()


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_L)
    ax.grid(color=GRID_L, lw=0.6, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, pad=8,
                     fontfamily="monospace", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=8)


# ── 1. City heatmap with routes ───────────────────────────────────────────────

def city_map(env: GreenCityEnv, results: list[RouteResult]) -> PIL.Image.Image:
    """Carbon cost heatmap with all agent routes overlaid."""
    size = env.size
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    # Carbon cost surface
    cost = np.ones((size, size)) * 0.3
    for z in env.congestion:
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = int(z[0]) + di, int(z[1]) + dj
                if 0 <= ni < size and 0 <= nj < size:
                    cost[ni][nj] += 1.4 if (di == 0 and dj == 0) else 0.5

    im = ax.imshow(cost, cmap="YlOrRd", alpha=0.45, origin="upper",
                   extent=[-0.5, size-0.5, size-0.5, -0.5], vmin=0, vmax=2.5)

    # Grid
    for i in range(size + 1):
        ax.axhline(i - 0.5, color=GRID_L, lw=0.8, zorder=1)
        ax.axvline(i - 0.5, color=GRID_L, lw=0.8, zorder=1)

    # Congestion zones
    for z in env.congestion:
        rect = mpatches.FancyBboxPatch(
            (int(z[1]) - 0.44, int(z[0]) - 0.44), 0.88, 0.88,
            boxstyle="round,pad=0.04",
            linewidth=1.5, edgecolor=WARN, facecolor="#ff6d0020", zorder=2,
        )
        ax.add_patch(rect)
        ax.text(int(z[1]), int(z[0]), "⚡", ha="center", va="center",
                fontsize=13, zorder=3, color=WARN)

    # Agent routes
    for r in results:
        col = AGENT_COLORS.get(r.agent_name, TEXT)
        if len(r.path) > 1:
            px = [p[1] for p in r.path]
            py = [p[0] for p in r.path]
            ax.plot(px, py, color=col, lw=2.2, alpha=0.8, zorder=4,
                    solid_capstyle="round", label=f"{r.agent_name} ({len(r.path)-1}steps)")

    # Start & goal
    start = results[0].path[0] if results else [0, 0]
    ax.scatter(start[1], start[0], s=200, marker="s", color=BLUE, zorder=7, linewidths=0)
    ax.text(start[1], start[0], "S", ha="center", va="center",
            fontsize=9, fontweight="bold", color=BG, zorder=8)

    g = env.goal
    ax.scatter(g[1], g[0], s=200, marker="*", color="#ffd600", zorder=7)
    ax.text(g[1], g[0], "G", ha="center", va="center",
            fontsize=7, fontweight="bold", color=BG, zorder=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Carbon cost", color=DIM, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=DIM)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=DIM)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.tick_params(colors=DIM, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_L)

    ax.set_title("City Carbon Map  ·  All Agent Routes", color=TEXT,
                 fontsize=11, pad=10, fontfamily="monospace")
    legend_items = [mpatches.Patch(color=AGENT_COLORS.get(r.agent_name, TEXT),
                                   label=r.agent_name) for r in results]
    legend_items += [
        mpatches.Patch(color=BLUE,    label="Start"),
        mpatches.Patch(color="#ffd600", label="Goal"),
        mpatches.Patch(color=WARN,    label="Congestion"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              facecolor=BG2, edgecolor=GRID_L, labelcolor=TEXT,
              fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    return _pil(fig)


# ── 2. Carbon trace comparison ────────────────────────────────────────────────

def carbon_trace(results: list[RouteResult]) -> PIL.Image.Image:
    """Cumulative carbon cost over steps for each agent."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for r in results:
        col  = AGENT_COLORS.get(r.agent_name, TEXT)
        cum  = np.cumsum(r.carbon_per_step)
        steps = list(range(1, len(cum) + 1))
        _style(ax1, "CUMULATIVE CARBON  (lower = greener)", "Step", "kg CO₂")
        ax1.fill_between(steps, cum, alpha=0.1, color=col)
        ax1.plot(steps, cum, color=col, lw=2.0, label=r.agent_name, alpha=0.9)

    ax1.legend(facecolor=BG2, edgecolor=GRID_L, labelcolor=TEXT, fontsize=8)

    # Bar chart: final totals
    _style(ax2, "TOTAL CARBON COMPARISON", "Strategy", "kg CO₂")
    names  = [r.agent_name.split("(")[0].strip() for r in results]
    totals = [r.total_carbon for r in results]
    cols   = [AGENT_COLORS.get(r.agent_name, TEXT) for r in results]
    bars   = ax2.bar(names, totals, color=cols, edgecolor=BG, linewidth=0.5,
                     alpha=0.85, width=0.5)
    for bar, val in zip(bars, totals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(totals)*0.01,
                 f"{val:.2f}", ha="center", color=TEXT, fontsize=9, fontweight="bold")
    ax2.tick_params(axis="x", colors=TEXT, labelsize=8)

    fig.suptitle("CARBON EMISSION ANALYSIS", color=TEXT, fontsize=10,
                 fontfamily="monospace", fontweight="bold")
    fig.tight_layout(pad=2)
    return _pil(fig)


# ── 3. Performance radar ──────────────────────────────────────────────────────

def performance_radar(results: list[RouteResult], env_size: int) -> PIL.Image.Image:
    labels = ["Eco Score", "Speed", "Safety", "Efficiency", "Delivery"]
    n = len(labels)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles_c = angles + [angles[0]]

    ncols = min(len(results), 3)
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4),
                             subplot_kw={"polar": True}, facecolor=BG)
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)

    for ax, r in zip(axes, results[:ncols]):
        col = AGENT_COLORS.get(r.agent_name, TEXT)
        ax.set_facecolor(BG2)

        eco   = max(0, min(10, 10 - r.total_carbon / 5))
        speed = max(0, min(10, 10 - r.steps * 0.4))
        safe  = max(0, min(10, 10 - r.congestion_hits * 2.5))
        eff   = max(0, min(10, 10 / (r.steps / (env_size * 1.2) + 0.1)))
        deliv = 10.0 if r.delivered else 2.0
        vals  = [eco, speed, safe, eff, deliv]
        vals_c = vals + [vals[0]]

        ax.plot(angles_c, vals_c, color=col, lw=2.0)
        ax.fill(angles_c, vals_c, color=col, alpha=0.18)
        ax.set_thetagrids([a * 180 / np.pi for a in angles],
                          labels, fontsize=8, color=TEXT)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(["2","4","6","8","10"], color=DIM, fontsize=6.5)
        ax.grid(color=GRID_L, lw=0.7)
        ax.spines["polar"].set_color(GRID_L)
        short = r.agent_name.split("(")[0].strip()
        ax.set_title(short, color=col, fontsize=9, pad=14,
                     fontfamily="monospace", fontweight="bold")

    fig.suptitle("PERFORMANCE RADAR  ·  All Strategies",
                 color=TEXT, fontsize=10, fontfamily="monospace",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _pil(fig)


# ── 4. Training curve ─────────────────────────────────────────────────────────

def training_curve(state) -> PIL.Image.Image:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG)
    fig.patch.set_facecolor(BG)

    _style(ax1, "DQN EPISODE REWARD", "Episode", "Reward")
    if state.ep_rewards:
        eps = list(range(len(state.ep_rewards)))
        ax1.plot(eps, state.ep_rewards, color=ACCENT, alpha=0.25, lw=0.8)
        if len(eps) > 20:
            k = max(5, len(eps)//30)
            s = np.convolve(state.ep_rewards, np.ones(k)/k, "valid")
            ax1.plot(range(k-1, len(eps)), s, color=ACCENT, lw=2.2)

    _style(ax2, "TRAINING PROGRESS")
    ax2.axis("off")
    pct  = state.timestep / max(state.total, 1) * 100
    n_ep = len(state.ep_rewards)
    roll = float(np.mean(state.ep_rewards[-20:])) if state.ep_rewards else 0
    info = [("Progress",f"{pct:.1f}%"),("Steps",f"{state.timestep:,}"),
            ("Episodes",str(n_ep)),("Rolling Reward",f"{roll:.2f}")]
    for i, (lbl, val) in enumerate(info):
        y = 0.80 - i*0.20
        ax2.text(0.05, y, lbl, transform=ax2.transAxes, color=DIM, fontsize=9,
                 fontfamily="monospace")
        ax2.text(0.62, y, val, transform=ax2.transAxes, color=ACCENT, fontsize=9,
                 fontweight="bold", fontfamily="monospace")

    fig.suptitle("DQN TRAINING DASHBOARD", color=TEXT, fontsize=10,
                 fontfamily="monospace", fontweight="bold")
    fig.tight_layout()
    return _pil(fig)


def empty_fig(msg: str = "") -> PIL.Image.Image:
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            color=DIM, fontsize=11, fontfamily="monospace")
    ax.axis("off")
    return _pil(fig)
