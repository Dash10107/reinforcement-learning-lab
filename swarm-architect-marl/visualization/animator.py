import io

import matplotlib
import numpy as np
import PIL.Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt

AGENT_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
GOAL_COLOR = "#DDA0DD"


def frames_to_gif(frames: list[np.ndarray], fps: int = 12, loop: int = 0) -> str:
    """Convert RGB frames to a GIF saved as a temp file, return the path."""
    import tempfile

    pil_frames = [PIL.Image.fromarray(f) for f in frames]
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)  # noqa: SIM115
    pil_frames[0].save(
        tmp.name,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=loop,
        optimize=True,
    )
    return tmp.name


def make_training_plot(log_data: dict) -> plt.Figure:
    eps = log_data["episodes"]
    if not eps:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(
            0.5,
            0.5,
            "No training data yet.\nStart training in the Training Lab tab.",
            ha="center",
            va="center",
            fontsize=14,
            color="#888",
        )
        ax.set_facecolor("#0f0f1a")
        fig.patch.set_facecolor("#0f0f1a")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#0f0f1a")

    panels = [
        ("mean_rewards", "Episode Reward", "#4ECDC4"),
        ("actor_losses", "Actor Loss", "#FF6B6B"),
        ("entropies", "Policy Entropy", "#FFEAA7"),
    ]
    for ax, (key, title, color) in zip(axes, panels):
        ax.set_facecolor("#16213e")
        ax.plot(eps, log_data[key], color=color, linewidth=1.2, alpha=0.5)
        # Smoothed curve
        if len(eps) > 10:
            kernel = np.ones(10) / 10
            smoothed = np.convolve(log_data[key], kernel, mode="valid")
            ax.plot(eps[9:], smoothed, color=color, linewidth=2.2)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.set_xlabel("Episode", color="#aaa", fontsize=9)

    fig.tight_layout(pad=2.0)
    return fig


def make_reward_history_plot(log_data: dict) -> plt.Figure:
    eps = log_data["episodes"]
    rewards = log_data["mean_rewards"]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#16213e")

    if eps:
        ax.fill_between(eps, rewards, alpha=0.15, color="#4ECDC4")
        ax.plot(eps, rewards, color="#4ECDC4", linewidth=1, alpha=0.4)
        if len(eps) > 20:
            k = max(5, len(eps) // 20)
            kernel = np.ones(k) / k
            smoothed = np.convolve(rewards, kernel, mode="valid")
            ax.plot(
                eps[k - 1 :], smoothed, color="#4ECDC4", linewidth=2.5, label="Smoothed"
            )
        ax.axhline(
            max(rewards),
            color="#FFEAA7",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"Best: {max(rewards):.3f}",
        )
        ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="white")

    ax.set_title("Team Reward over Training", color="white", fontsize=13, pad=10)
    ax.set_xlabel("Episode", color="#aaa")
    ax.set_ylabel("Mean Agent Reward", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")

    fig.tight_layout()
    return fig


def overlay_metrics_on_frame(
    frame: np.ndarray, step: int, total_steps: int, rewards: dict | None = None
) -> np.ndarray:
    """Burn step counter and reward info into a frame using matplotlib."""
    fig, ax = plt.subplots(
        figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100
    )
    ax.imshow(frame)
    ax.axis("off")

    ax.text(
        5,
        12,
        f"Step {step}/{total_steps}",
        fontsize=8,
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
    )  # noqa: C408

    if rewards:
        team_r = sum(rewards.values())
        ax.text(
            5,
            frame.shape[0] - 8,
            f"Team Reward: {team_r:.3f}",
            fontsize=7,
            color="#4ECDC4",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )  # noqa: C408

    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = PIL.Image.open(buf).convert("RGB")
    return np.array(img.resize((frame.shape[1], frame.shape[0])))
