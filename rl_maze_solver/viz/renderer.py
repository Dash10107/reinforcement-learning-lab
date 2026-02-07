"""
All visual output: animated GIF of the agent solving the maze,
training curve chart, Q-value heatmap, algorithm race chart.
"""

from __future__ import annotations
import io
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as mpl_cm
from matplotlib.figure import Figure as MplFigure
from matplotlib.axes import Axes as MplAxes
from matplotlib.colors import LinearSegmentedColormap
import PIL.Image
import PIL.ImageDraw

from maze.env import MazeEnv
from agents.base import TabularAgent

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0d1117"
BG2      = "#161b22"
GRID_C   = "#21262d"
WALL_C   = "#1f6feb"
OPEN_C   = "#0d1117"
PATH_C   = "#3fb950"
START_C  = "#58a6ff"
GOAL_C   = "#f78166"
AGENT_C  = "#ffa657"
TEXT_C   = "#c9d1d9"
DIM_C    = "#484f58"


def _fig_to_pil(fig: MplFigure) -> PIL.Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return PIL.Image.open(buf).convert("RGB")


def _draw_maze_frame(
    ax: MplAxes,
    grid: np.ndarray,
    path: list[tuple[int, int]],
    current_step: int,
    H: int, W: int,
    show_heatmap: np.ndarray | None = None,
) -> None:
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw cells
    for r in range(H):
        for c in range(W):
            is_wall = grid[r, c] == 1
            if show_heatmap is not None and not is_wall:
                val = float(show_heatmap[r, c])
                intensity = np.clip(val, 0, 1)
                color = mpl_cm.get_cmap("YlOrRd")(0.2 + 0.8 * intensity)
            else:
                color = "#2d333b" if is_wall else BG2
            rect = patches.FancyBboxPatch(
                (c - 0.45, r - 0.45), 0.90, 0.90,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor=GRID_C, linewidth=0.4,
            )
            ax.add_patch(rect)

    # Start & goal
    ax.add_patch(patches.Circle((0, 0), 0.35, color=START_C, zorder=3))
    ax.add_patch(patches.Circle((W-1, H-1), 0.35, color=GOAL_C, zorder=3))
    ax.text(0, 0, "S", ha="center", va="center", color="white",
            fontsize=7, fontweight="bold", zorder=4)
    ax.text(W-1, H-1, "G", ha="center", va="center", color="white",
            fontsize=7, fontweight="bold", zorder=4)

    # Walked path (up to current_step)
    walked = path[:current_step]
    if len(walked) > 1:
        xs = [c for _, c in walked]
        ys = [r for r, _ in walked]
        ax.plot(xs, ys, color=PATH_C, linewidth=2.5, alpha=0.7,
                zorder=2, solid_capstyle="round")

    # Current agent position
    if walked:
        ar, ac = walked[-1]
        ax.add_patch(patches.Circle((ac, ar), 0.32, color=AGENT_C, zorder=5))
        ax.text(ac, ar, "●", ha="center", va="center",
                color="white", fontsize=8, zorder=6)


def make_solution_gif(
    env: MazeEnv,
    agent: TabularAgent,
    fps: int = 8,
    label: str = "",
) -> str:
    """Greedy rollout → animated GIF of agent walking the maze."""
    # Collect path
    state, _ = env.reset()
    path: list[tuple[int, int]] = [env.start]
    for _ in range(env.n_states * 3):
        action = agent.greedy_action(state)
        state, _, done, _, _ = env.step(action)
        path.append(env._from_state(state))
        if done:
            break

    H, W = env.shape
    pil_frames: list[PIL.Image.Image] = []

    for step in range(1, len(path) + 1):
        fig, ax = plt.subplots(figsize=(max(4, W * 0.55), max(4, H * 0.55)))
        fig.patch.set_facecolor(BG)
        _draw_maze_frame(ax, env.grid, path, step, H, W)

        status = f"Step {step}/{len(path)-1}"
        if step == len(path) and path[-1] == env.goal:
            status = f"🎉 Solved in {len(path)-1} steps!"
        ax.set_title(f"{label}  {status}", color=TEXT_C, fontsize=9,
                     pad=6, fontfamily="monospace")
        pil_frames.append(_fig_to_pil(fig))

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    pil_frames[0].save(
        tmp.name, save_all=True, append_images=pil_frames[1:],
        duration=int(1000 / fps), loop=0, optimize=False,
    )
    return tmp.name


def make_training_chart(
    rewards_dict: dict[str, list[float]],
    colors: dict[str, str] | None = None,
) -> MplFigure:
    """Reward-per-episode curves for one or more algorithms."""
    default_colors = {"Q-Learning": "#3fb950", "SARSA": "#58a6ff", "Monte Carlo": "#ffa657"}
    colors = colors or default_colors

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    for name, rewards in rewards_dict.items():
        col = colors.get(name, "#ffffff")
        eps = list(range(len(rewards)))
        ax.plot(eps, rewards, color=col, linewidth=0.8, alpha=0.3)
        if len(eps) > 20:
            k = max(5, len(eps) // 40)
            smooth = np.convolve(rewards, np.ones(k) / k, "valid")
            ax.plot(range(k - 1, len(eps)), smooth, color=col, linewidth=2.2, label=name)
        else:
            ax.plot(eps, rewards, color=col, linewidth=2.0, label=name)

    ax.set_xlabel("Episode", color=DIM_C, fontsize=9)
    ax.set_ylabel("Total Reward", color=DIM_C, fontsize=9)
    ax.set_title("Training Progress — Reward per Episode", color=TEXT_C,
                 fontsize=11, pad=10, fontfamily="monospace")
    ax.tick_params(colors=DIM_C, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_C)
    ax.grid(color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.5)
    if len(rewards_dict) > 1:
        ax.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9)
    fig.tight_layout()
    return fig


def make_qvalue_heatmap(env: MazeEnv, agent: TabularAgent, title: str = "") -> MplFigure:
    """Q-value heatmap overlaid on the maze grid."""
    H, W = env.shape
    max_q = np.max(agent.Q, axis=1).reshape(H, W)

    # Normalise to [0, 1] for colour mapping
    qmin, qmax = max_q.min(), max_q.max()
    norm_q = (max_q - qmin) / max(qmax - qmin, 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, H * 0.6)))
    fig.patch.set_facecolor(BG)

    # Left: maze with learned path
    ax1.set_facecolor(BG)
    state, _ = env.reset()
    path: list[tuple[int, int]] = [env.start]
    for _ in range(env.n_states * 3):
        action = agent.greedy_action(state)
        state, _, done, _, _ = env.step(action)
        path.append(env._from_state(state))
        if done:
            break
    _draw_maze_frame(ax1, env.grid, path, len(path), H, W)
    solved = path[-1] == env.goal
    ax1.set_title(
        f"{'Solved' if solved else 'Not solved'} — {len(path)-1} steps",
        color=TEXT_C, fontsize=10, pad=8, fontfamily="monospace",
    )

    # Right: Q-value heatmap
    ax2.set_facecolor(BG)
    ax2.set_xlim(-0.5, W - 0.5)
    ax2.set_ylim(H - 0.5, -0.5)
    ax2.set_aspect("equal")
    ax2.axis("off")

    cmap = LinearSegmentedColormap.from_list("qmap", ["#0d1117", "#1f6feb", "#3fb950"])
    for r in range(H):
        for c in range(W):
            if env.grid[r, c] == 1:
                color = "#2d333b"
            else:
                color = cmap(norm_q[r, c])
            rect = patches.FancyBboxPatch(
                (c - 0.45, r - 0.45), 0.90, 0.90,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor=GRID_C, linewidth=0.4,
            )
            ax2.add_patch(rect)
            if env.grid[r, c] == 0 and H <= 11:
                ax2.text(c, r, f"{max_q[r,c]:.0f}",
                         ha="center", va="center", fontsize=5.5,
                         color="white" if norm_q[r, c] > 0.4 else DIM_C)

    ax2.set_title("Q-Value Map (brighter = agent prefers this cell)",
                  color=TEXT_C, fontsize=10, pad=8, fontfamily="monospace")

    if title:
        fig.suptitle(title, color=TEXT_C, fontsize=12, fontfamily="monospace", y=1.02)
    fig.tight_layout()
    return fig


def make_race_chart(
    rewards_a: list[float], name_a: str,
    rewards_b: list[float], name_b: str,
    rewards_c: list[float] | None = None, name_c: str = "",
) -> MplFigure:
    """Head-to-head convergence race chart."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(BG)

    pairs = [(rewards_a, name_a, "#3fb950"), (rewards_b, name_b, "#58a6ff")]
    if rewards_c:
        pairs.append((rewards_c, name_c, "#ffa657"))

    # Left: raw + smoothed reward curves
    ax = axes[0]
    ax.set_facecolor(BG2)
    for rewards, name, col in pairs:
        eps = list(range(len(rewards)))
        ax.plot(eps, rewards, color=col, linewidth=0.6, alpha=0.25)
        if len(eps) > 20:
            k = max(5, len(eps) // 40)
            smooth = np.convolve(rewards, np.ones(k) / k, "valid")
            ax.plot(range(k-1, len(eps)), smooth, color=col, linewidth=2.2, label=name)
    ax.set_title("Reward Convergence", color=TEXT_C, fontsize=10, fontfamily="monospace")
    ax.set_xlabel("Episode", color=DIM_C, fontsize=8)
    ax.set_ylabel("Reward", color=DIM_C, fontsize=8)
    ax.tick_params(colors=DIM_C, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID_C)
    ax.grid(color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

    # Right: bar chart of final performance
    ax2 = axes[1]
    ax2.set_facecolor(BG2)
    names = [p[1] for p in pairs]
    finals = [float(np.mean(p[0][-max(1, len(p[0])//5):])) for p in pairs]
    cols = [p[2] for p in pairs]
    bars = ax2.bar(names, finals, color=cols, edgecolor=BG, linewidth=0.8, width=0.5)
    for bar, val in zip(bars, finals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + abs(min(finals)) * 0.02,
                 f"{val:.1f}", ha="center", va="bottom",
                 color=TEXT_C, fontsize=9, fontweight="bold")
    ax2.set_title("Final Performance (avg last 20%)", color=TEXT_C,
                  fontsize=10, fontfamily="monospace")
    ax2.tick_params(colors=DIM_C, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color(GRID_C)
    ax2.grid(axis="y", color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_facecolor(BG2)

    fig.tight_layout(pad=2.0)
    return fig


def score_run(path: list[tuple[int, ...]], goal: tuple[int, int],
              rewards: list[float], n_states: int) -> dict:
    solved = bool(path and path[-1] == goal)
    steps = len(path) - 1
    optimal_approx = n_states ** 0.5  # rough floor
    efficiency = min(100, int(optimal_approx / max(steps, 1) * 100)) if solved else 0
    avg_r = float(np.mean(rewards[-max(1, len(rewards)//5):])) if rewards else 0

    if not solved:
        grade, verdict = "F", "Bot got lost — try more episodes!"
    elif efficiency >= 80:
        grade, verdict = "S", "Perfect pathfinding! Optimal route."
    elif efficiency >= 60:
        grade, verdict = "A", "Excellent! Near-optimal path."
    elif efficiency >= 40:
        grade, verdict = "B", "Good solve — some detours taken."
    else:
        grade, verdict = "C", "Solved but took the scenic route!"

    return {"solved": solved, "steps": steps, "grade": grade,
            "verdict": verdict, "efficiency": efficiency, "avg_reward": avg_r}
