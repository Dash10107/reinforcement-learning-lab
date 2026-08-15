"""
Warehouse visualisation — renders the grid + robots to numpy RGB frames.
"""

from __future__ import annotations
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
import PIL.Image

from warehouse.layout import OPEN, SHELF, PICKUP, DELIVERY

# ── Colour scheme ─────────────────────────────────────────────────────────────
C_BG       = "#0d1117"
C_FLOOR    = "#161b22"
C_SHELF    = "#21262d"
C_SHELF_FG = "#30363d"
C_PICKUP   = "#0d2d17"
C_PICKUP_B = "#2ea043"
C_DELIVERY = "#0d1a2e"
C_DELIVERY_B = "#1890ff"

ROBOT_COLORS = [
    "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4",
    "#10b981", "#f97316", "#ec4899", "#6366f1",
]
TRAIL_ALPHA = 0.35


def render_frame(
    grid: np.ndarray,
    robots: list,
    trails: list[list[tuple[int, int]]] | None = None,
    step: int = 0,
    deliveries: int = 0,
    collisions: int = 0,
    title: str = "",
    cell_size: float = 0.55,
) -> np.ndarray:
    H, W = grid.shape
    fig_w = W * cell_size
    fig_h = H * cell_size + 0.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)   # flip y so row 0 is at top
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw grid cells
    for r in range(H):
        for c in range(W):
            ct = int(grid[r, c])
            if ct == SHELF:
                facecolor = C_SHELF
                ec = C_SHELF_FG
            elif ct == PICKUP:
                facecolor = C_PICKUP
                ec = C_PICKUP_B
            elif ct == DELIVERY:
                facecolor = C_DELIVERY
                ec = C_DELIVERY_B
            else:
                facecolor = C_FLOOR
                ec = "#1c2333"

            rect = patches.FancyBboxPatch(
                (c - 0.46, r - 0.46), 0.92, 0.92,
                boxstyle="round,pad=0.03",
                facecolor=facecolor, edgecolor=ec, linewidth=0.5, zorder=1,
            )
            ax.add_patch(rect)

    # Labels for pickup / delivery
    for r in range(H):
        for c in range(W):
            ct = int(grid[r, c])
            if ct == PICKUP:
                ax.text(c, r, "P", ha="center", va="center",
                        color=C_PICKUP_B, fontsize=5.5, fontweight="bold", zorder=2)
            elif ct == DELIVERY:
                ax.text(c, r, "D", ha="center", va="center",
                        color=C_DELIVERY_B, fontsize=5.5, fontweight="bold", zorder=2)

    # Draw trails
    if trails:
        for i, trail in enumerate(trails):
            if len(trail) < 2:
                continue
            col = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            xs = [c for r, c in trail]
            ys = [r for r, c in trail]
            ax.plot(xs, ys, color=col, linewidth=1.2, alpha=TRAIL_ALPHA, zorder=3,
                    solid_capstyle="round")

    # Draw task lines (robot → goal)
    for i, robot in enumerate(robots):
        col = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax.plot([robot.col, robot.goal_col], [robot.row, robot.goal_row],
                color=col, linewidth=0.7, alpha=0.35, linestyle=":", zorder=3)

    # Draw robots
    for i, robot in enumerate(robots):
        col = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        circle = patches.Circle(
            (robot.col, robot.row), 0.30,
            facecolor=col, edgecolor="white", linewidth=0.8, zorder=5,
        )
        ax.add_patch(circle)

        # Package indicator
        if robot.has_package:
            pkg = patches.Rectangle(
                (robot.col - 0.13, robot.row - 0.13), 0.26, 0.26,
                facecolor="white", edgecolor=col, linewidth=0.6, zorder=6,
            )
            ax.add_patch(pkg)

        ax.text(robot.col, robot.row, str(i + 1),
                ha="center", va="center", color="white" if not robot.has_package else col,
                fontsize=5, fontweight="bold", zorder=7)

    # HUD
    hud = f"Step {step:03d}  |  Deliveries: {deliveries}  |  Collisions: {collisions}"
    if title:
        hud = f"[{title}]  " + hud
    ax.text(W / 2 - 0.5, H - 0.08, hud,
            ha="center", va="top", color="#4a5568",
            fontsize=5.5, fontfamily="monospace", zorder=10)

    plt.tight_layout(pad=0.2)
    buf = __import__("io").BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=C_BG)
    buf.seek(0)
    plt.close(fig)
    return np.array(PIL.Image.open(buf).convert("RGB"))


def frames_to_gif(frames: list[np.ndarray], fps: int = 10) -> str:
    pil_frames = [PIL.Image.fromarray(f) for f in frames]
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    pil_frames[0].save(
        tmp.name, save_all=True, append_images=pil_frames[1:],
        duration=int(1000 / fps), loop=0, optimize=False,
    )
    return tmp.name
