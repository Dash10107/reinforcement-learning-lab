"""
All visualisation for the MBRL Pendulum Playground.
"""

from __future__ import annotations

import io
import tempfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import PIL.Image
from matplotlib import gridspec
from matplotlib.figure import Figure
from model.dynamics import STATE_NAMES

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#0e1117"
BG2 = "#161b27"
BG3 = "#1c2333"
GRID = "#252d40"
TEXT = "#e2e8f0"
DIM = "#4a5568"
REAL = "#7c3aed"  # purple — real trajectory
IMAG = "#06b6d4"  # cyan   — imagined trajectory
ERR = "#f59e0b"  # amber  — error
MPC = "#10b981"  # green  — MPC
RAND = "#ef4444"  # red    — random baseline
UNC = "#8b5cf6"  # violet — uncertainty band


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_color(GRID)
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


def _fig_to_pil(fig: Figure) -> PIL.Image.Image:
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    buf.seek(0)
    plt.close(fig)
    return PIL.Image.open(buf).convert("RGB")


# ── 1. Single-step explorer ───────────────────────────────────────────────────


def single_step_comparison(
    real_next: np.ndarray,
    pred_next: np.ndarray,
    pred_std: np.ndarray | None = None,
) -> Figure:
    """Bar chart comparing real vs predicted next state, with optional uncertainty."""
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG)
    _style(ax, "REAL  vs  PREDICTED NEXT STATE", ylabel="Value")
    fig.patch.set_facecolor(BG)

    x = np.arange(3)
    w = 0.32
    ax.bar(
        x - w / 2,
        real_next,
        width=w,
        color=REAL,
        label="Real (environment)",
        edgecolor=BG,
    )
    ax.bar(
        x + w / 2,
        pred_next,
        width=w,
        color=IMAG,
        label="Predicted (model)",
        edgecolor=BG,
    )

    if pred_std is not None:
        ax.errorbar(
            x + w / 2,
            pred_next,
            yerr=pred_std * 1.96,
            fmt="none",
            color=UNC,
            linewidth=1.5,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES, color=DIM, fontsize=9)
    ax.axhline(0, color=GRID, linewidth=0.8)

    # Annotate error
    for i, (r, p) in enumerate(zip(real_next, pred_next)):
        err = abs(r - p)
        ax.text(
            i,
            max(abs(r), abs(p)) + 0.05,
            f"Δ{err:.3f}",
            ha="center",
            color=ERR,
            fontsize=7.5,
            fontfamily="monospace",
        )

    ax.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


# ── 2. Imagination rollout comparison ────────────────────────────────────────


def imagination_comparison(
    real_traj: np.ndarray,  # (T, 3) real states
    imag_traj: np.ndarray,  # (T, 3) imagined states
    imag_std: np.ndarray | None,  # (T, 3) uncertainty
) -> Figure:
    """4-panel: 3 state dims + cumulative MSE error."""
    T = len(real_traj)
    steps = np.arange(T)

    fig = plt.figure(figsize=(13, 8), facecolor=BG)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        left=0.08,
        right=0.97,
        top=0.88,
        bottom=0.08,
    )

    # State panels
    for i, name in enumerate(STATE_NAMES):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        _style(ax, name.upper(), "Step →", "Value")

        ax.plot(steps, real_traj[:, i], color=REAL, linewidth=2.2, label="Real")
        ax.plot(
            steps,
            imag_traj[:, i],
            color=IMAG,
            linewidth=2.0,
            linestyle="--",
            label="Imagined",
        )

        if imag_std is not None:
            ax.fill_between(
                steps,
                imag_traj[:, i] - 2 * imag_std[:, i],
                imag_traj[:, i] + 2 * imag_std[:, i],
                color=IMAG,
                alpha=0.15,
                label="95% CI",
            )
        ax.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=7)

    # MSE panel
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4, "PREDICTION ERROR  (cumulative)", "Step →", "MSE")
    mse_per_step = np.mean((real_traj - imag_traj) ** 2, axis=1)
    cum_mse = np.cumsum(mse_per_step)
    ax4.fill_between(steps, 0, mse_per_step, color=ERR, alpha=0.3)
    ax4.plot(steps, mse_per_step, color=ERR, linewidth=1.5, label="Step MSE")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(
        steps, cum_mse, color=UNC, linewidth=2.0, linestyle=":", label="Cumulative"
    )
    ax4_twin.tick_params(colors=DIM, labelsize=7)
    ax4_twin.set_ylabel("Cumulative MSE", color=DIM, fontsize=7)
    ax4.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=7)

    fig.suptitle(
        "IMAGINATION ROLLOUT  ·  Real Trajectory vs Model Prediction",
        color=TEXT,
        fontsize=11,
        fontfamily="monospace",
        fontweight="bold",
        y=0.95,
    )
    return fig


# ── 3. Training loss curve ────────────────────────────────────────────────────


def training_curve(train_losses: list[float], val_losses: list[float]) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Loss curves
    ax = axes[0]
    _style(ax, "DYNAMICS MODEL TRAINING", "Epoch", "MSE Loss")
    eps = list(range(len(train_losses)))
    ax.semilogy(eps, train_losses, color=MPC, linewidth=2.0, label="Train")
    ax.semilogy(
        eps, val_losses, color=REAL, linewidth=2.0, linestyle="--", label="Validation"
    )
    ax.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # Per-state breakdown indicator
    ax2 = axes[1]
    _style(ax2, "TRAINING PROGRESS", "Epoch", "")
    ax2.axis("off")

    if train_losses:
        info = [
            ("Final Train MSE", f"{train_losses[-1]:.5f}"),
            ("Final Val MSE", f"{val_losses[-1]:.5f}" if val_losses else "—"),
            ("Best Val MSE", f"{min(val_losses):.5f}" if val_losses else "—"),
            ("Epochs", str(len(train_losses))),
            (
                "Improvement",
                f"{(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%"
                if train_losses[0] > 0
                else "—",
            ),
        ]
        for i, (label, val) in enumerate(info):
            y = 0.85 - i * 0.16
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
                0.65,
                y,
                val,
                transform=ax2.transAxes,
                color=MPC,
                fontsize=9,
                fontweight="bold",
                fontfamily="monospace",
            )

    fig.tight_layout(pad=2.0)
    return fig


# ── 4. MPC episode results ────────────────────────────────────────────────────


def mpc_episode_chart(
    mpc_result: dict,
    random_result: dict | None = None,
) -> Figure:
    fig = plt.figure(figsize=(13, 7), facecolor=BG)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        left=0.08,
        right=0.97,
        top=0.88,
        bottom=0.08,
    )

    mpc_r = mpc_result["rewards"]
    mpc_a = mpc_result["actions"]
    mpc_cum = np.cumsum(mpc_r)
    steps_m = list(range(len(mpc_r)))

    # Cumulative reward
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1, "CUMULATIVE EPISODE REWARD", "Step", "Reward")
    ax1.fill_between(steps_m, mpc_cum, alpha=0.2, color=MPC)
    ax1.plot(steps_m, mpc_cum, color=MPC, linewidth=2.2, label="MPC (MBRL)")
    if random_result:
        rand_r = random_result["rewards"]
        rand_cum = np.cumsum(rand_r[: len(mpc_r)])
        ax1.fill_between(range(len(rand_cum)), rand_cum, alpha=0.15, color=RAND)
        ax1.plot(
            range(len(rand_cum)),
            rand_cum,
            color=RAND,
            linewidth=1.8,
            linestyle="--",
            label="Random policy",
        )
    ax1.axhline(0, color=GRID, linewidth=0.8)
    ax1.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # Action sequence
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2, "TORQUE APPLIED  (MPC)", "Step", "Torque (N·m)")
    ax2.fill_between(steps_m, 0, mpc_a, alpha=0.4, color=MPC)
    ax2.plot(steps_m, mpc_a, color=MPC, linewidth=1.5)
    ax2.axhline(0, color=GRID, linewidth=0.8)
    ax2.set_ylim(-2.2, 2.2)

    # State: cos_theta (upright = 1)
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3, "cos θ  (1 = upright)", "Step", "cos θ")
    cos_vals = [s[0] for s in mpc_result["states"]]
    ax3.fill_between(steps_m, 0, cos_vals, alpha=0.2, color=IMAG)
    ax3.plot(steps_m, cos_vals, color=IMAG, linewidth=2.0)
    ax3.axhline(
        1.0, color=MPC, linewidth=1, linestyle="--", alpha=0.5, label="Upright target"
    )
    ax3.set_ylim(-1.1, 1.15)
    ax3.legend(facecolor=BG3, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    total_mpc = mpc_result["total_reward"]
    total_r = random_result["total_reward"] if random_result else 0
    fig.suptitle(
        f"MPC EPISODE  ·  Total Reward: {total_mpc:.1f}  "
        f"(Random: {total_r:.1f}  ·  Improvement: {total_mpc - total_r:+.1f})",
        color=TEXT,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
        y=0.96,
    )
    return fig


# ── 5. GIF ────────────────────────────────────────────────────────────────────


def make_mpc_gif(frames: list[np.ndarray], fps: int = 20) -> str:
    """Convert RGB frames to animated GIF. Returns temp file path."""
    if not frames:
        return ""
    pil_frames = [PIL.Image.fromarray(f) for f in frames]
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)  # noqa: SIM115
    pil_frames[0].save(
        tmp.name,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
    return tmp.name


# ── 6. Uncertainty heatmap ────────────────────────────────────────────────────


def uncertainty_heatmap(
    ensemble,
    vel_range: tuple[float, float] = (-8, 8),
    n_grid: int = 30,
) -> Figure:
    """
    Show ensemble uncertainty over (theta, angular_velocity) state space
    at a fixed action = 0.
    """
    thetas = np.linspace(-np.pi, np.pi, n_grid)
    vels = np.linspace(vel_range[0], vel_range[1], n_grid)

    unc_map = np.zeros((n_grid, n_grid))
    with __import__("torch").no_grad():
        import torch

        for i, theta in enumerate(thetas):
            for j, vel in enumerate(vels):
                state = torch.FloatTensor([[np.cos(theta), np.sin(theta), vel]])
                action = torch.zeros(1, 1)
                _, std = ensemble.forward_all(state, action)
                unc_map[j, i] = std.mean().item()

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    im = ax.contourf(
        np.degrees(thetas),
        vels,
        unc_map,
        levels=20,
        cmap="plasma",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=DIM, labelsize=7)
    cbar.set_label("Ensemble Std (uncertainty)", color=DIM, fontsize=8)

    ax.set_xlabel("θ (degrees)", color=DIM, fontsize=9)
    ax.set_ylabel("Angular velocity (rad/s)", color=DIM, fontsize=9)
    ax.tick_params(colors=DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.axvline(0, color=TEXT, linewidth=0.8, linestyle="--", alpha=0.4)
    ax.text(3, vel_range[1] * 0.85, "← upright", color=TEXT, fontsize=7.5)
    ax.text(-175, vel_range[1] * 0.85, "hanging →", color=TEXT, fontsize=7.5)

    _style(ax, "ENSEMBLE UNCERTAINTY MAP  ·  Action = 0 (hold)")
    fig.tight_layout()
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
