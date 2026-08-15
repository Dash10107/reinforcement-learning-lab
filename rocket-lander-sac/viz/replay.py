"""
Animated GIF generation from raw RGB frames.
Adds HUD overlay (step, reward, throttle bars) using PIL drawing — no matplotlib overhead.
"""

from __future__ import annotations

import tempfile

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from core.mission import EpisodeResult

# ── HUD rendering ─────────────────────────────────────────────────────────────


def _draw_hud(
    img: PIL.Image.Image,
    step: int,
    cumulative_reward: float,
    main_throttle: float,
    lateral_throttle: float,
    status: str,
) -> PIL.Image.Image:
    draw = PIL.ImageDraw.Draw(img)
    W, H = img.size

    # Semi-transparent top bar
    draw.rectangle([(0, 0), (W, 22)], fill=(3, 11, 26, 200))

    # Step & reward text
    draw.text((6, 4), f"STEP {step:03d}", fill=(79, 179, 255), font=None)
    rcolor = (45, 219, 124) if cumulative_reward >= 0 else (255, 77, 109)
    draw.text(
        (W // 2 - 40, 4), f"REWARD {cumulative_reward:+.1f}", fill=rcolor, font=None
    )
    draw.text((W - 80, 4), status, fill=(248, 166, 35), font=None)

    # Throttle bars at bottom
    BAR_H = 6
    BAR_Y = H - BAR_H - 4

    # Main engine bar (blue)
    bar_max = W // 2 - 20
    bar_w = int(abs(main_throttle) * bar_max)
    draw.rectangle([(10, BAR_Y), (10 + bar_max, BAR_Y + BAR_H)], fill=(13, 37, 64))
    draw.rectangle([(10, BAR_Y), (10 + bar_w, BAR_Y + BAR_H)], fill=(79, 179, 255))
    draw.text((10, BAR_Y - 11), "MAIN", fill=(79, 179, 255), font=None)

    # Lateral bar (amber)
    lx = W // 2 + 10
    lat_w = int(abs(lateral_throttle) * bar_max)
    draw.rectangle([(lx, BAR_Y), (lx + bar_max, BAR_Y + BAR_H)], fill=(13, 37, 64))
    col = (245, 166, 35) if lateral_throttle >= 0 else (255, 77, 109)
    draw.rectangle([(lx, BAR_Y), (lx + lat_w, BAR_Y + BAR_H)], fill=col)
    draw.text((lx, BAR_Y - 11), "LATERAL", fill=(245, 166, 35), font=None)

    return img


def make_episode_gif(
    frames: list[np.ndarray],
    episode: EpisodeResult,
    fps: int = 15,
) -> str:
    """Overlay HUD on every frame, save as animated GIF. Returns temp file path."""
    if not frames:
        return ""

    cum_rewards = episode.cumulative_rewards
    pil_frames: list[PIL.Image.Image] = []

    for i, frame in enumerate(frames):
        img = PIL.Image.fromarray(frame).convert("RGBA")
        cum_r = cum_rewards[i] if i < len(cum_rewards) else cum_rewards[-1]
        step_data = episode.steps[i] if i < len(episode.steps) else episode.steps[-1]
        img = _draw_hud(
            img.convert("RGB"),
            step=i + 1,
            cumulative_reward=cum_r,
            main_throttle=step_data.action_main,
            lateral_throttle=step_data.action_lateral,
            status=episode.status,
        )
        pil_frames.append(img)

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    pil_frames[0].save(
        tmp.name,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
    return tmp.name


def make_comparison_gif(
    all_frames: list[list[np.ndarray]],
    episodes: list[EpisodeResult],
    fps: int = 12,
    max_episodes: int = 4,
) -> str:
    """
    Side-by-side grid GIF comparing up to `max_episodes` episodes.
    Pads shorter episodes with their last frame.
    """
    n = min(len(all_frames), max_episodes)
    if n == 0:
        return ""

    frame_lists = [all_frames[i] for i in range(n)]
    ep_list = [episodes[i] for i in range(n)]

    max_len = max(len(fl) for fl in frame_lists)
    # Pad each episode to max_len
    padded = [fl + [fl[-1]] * (max_len - len(fl)) if fl else [] for fl in frame_lists]

    if not padded[0]:
        return ""

    h, w = padded[0][0].shape[:2]
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    grid_w, grid_h = cols * w, rows * h

    pil_frames: list[PIL.Image.Image] = []
    for step_i in range(max_len):
        canvas = PIL.Image.new("RGB", (grid_w, grid_h), (3, 11, 26))
        for ep_i in range(n):
            if step_i < len(padded[ep_i]):
                cell = PIL.Image.fromarray(padded[ep_i][step_i])
            else:
                continue
            # label
            draw = PIL.ImageDraw.Draw(cell)
            ep = ep_list[ep_i]
            draw.rectangle([(0, 0), (cell.width, 16)], fill=(3, 11, 26))
            col = (45, 219, 124) if ep.total_reward >= 150 else (255, 77, 109)
            draw.text(
                (4, 2),
                f"#{ep.episode_idx + 1} {ep.status} {ep.total_reward:+.0f}",
                fill=col,
                font=None,
            )
            cx = (ep_i % cols) * w
            cy = (ep_i // cols) * h
            canvas.paste(cell, (cx, cy))
        pil_frames.append(canvas)

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    pil_frames[0].save(
        tmp.name,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
    return tmp.name
