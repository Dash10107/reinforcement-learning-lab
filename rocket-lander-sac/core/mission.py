"""
Mission runner — executes SAC agent episodes, collects full telemetry.
Returns structured data for both the UI and the visualization layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

# ── Telemetry data structures ─────────────────────────────────────────────────


@dataclass
class StepData:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_vel: float
    left_leg: bool
    right_leg: bool
    reward: float
    action_main: float  # main engine throttle [-1, 1]
    action_lateral: float  # lateral thruster [-1, 1]


@dataclass
class EpisodeResult:
    episode_idx: int
    steps: list[StepData] = field(default_factory=list)
    total_reward: float = 0.0
    landed: bool = False
    crashed: bool = False

    @property
    def status(self) -> str:
        if self.total_reward >= 200:
            return "PERFECT"
        if self.total_reward >= 150:
            return "LANDED"
        if self.total_reward >= 0:
            return "PARTIAL"
        return "CRASHED"

    @property
    def status_emoji(self) -> str:
        return {"PERFECT": "🏆", "LANDED": "✅", "PARTIAL": "⚠️", "CRASHED": "💥"}[
            self.status
        ]

    @property
    def xs(self) -> list[float]:
        return [s.x for s in self.steps]

    @property
    def ys(self) -> list[float]:
        return [s.y for s in self.steps]

    @property
    def cumulative_rewards(self) -> list[float]:
        total = 0.0
        out = []
        for s in self.steps:
            total += s.reward
            out.append(total)
        return out

    @property
    def main_throttle(self) -> list[float]:
        return [s.action_main for s in self.steps]

    @property
    def lateral_throttle(self) -> list[float]:
        return [s.action_lateral for s in self.steps]

    @property
    def angles(self) -> list[float]:
        return [np.degrees(s.angle) for s in self.steps]


@dataclass
class MissionResult:
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def rewards(self) -> list[float]:
        return [e.total_reward for e in self.episodes]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.total_reward >= 150) / len(
            self.episodes
        )

    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def best(self) -> EpisodeResult:
        return max(self.episodes, key=lambda e: e.total_reward)

    @property
    def worst(self) -> EpisodeResult:
        return min(self.episodes, key=lambda e: e.total_reward)


# ── Runner ────────────────────────────────────────────────────────────────────


def run_mission(
    model: SAC,
    n_episodes: int = 5,
    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 5.0,
    turbulence_power: float = 0.5,
    render: bool = True,
    progress_cb=None,
) -> tuple[MissionResult, list[list[np.ndarray]]]:
    """
    Run `n_episodes` of the lander.
    Returns (MissionResult, list_of_frame_lists) — one frame list per episode.
    """
    mission = MissionResult()
    all_frames: list[list[np.ndarray]] = []

    env_kwargs = {
        "continuous": True,
        "gravity": gravity,
        "enable_wind": enable_wind,
        "wind_power": wind_power if enable_wind else 0.0,
        "turbulence_power": turbulence_power if enable_wind else 0.0,
        "render_mode": "rgb_array" if render else None,
    }

    for ep_idx in range(n_episodes):
        if progress_cb:
            progress_cb(
                ep_idx / n_episodes, f"Running mission {ep_idx + 1}/{n_episodes}…"
            )

        env = gym.make("LunarLander-v3", **env_kwargs)
        obs, _ = env.reset()

        result = EpisodeResult(episode_idx=ep_idx)
        frames: list[np.ndarray] = []

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            result.steps.append(
                StepData(
                    x=float(obs[0]),
                    y=float(obs[1]),
                    vx=float(obs[2]),
                    vy=float(obs[3]),
                    angle=float(obs[4]),
                    angular_vel=float(obs[5]),
                    left_leg=bool(obs[6]),
                    right_leg=bool(obs[7]),
                    reward=float(reward),
                    action_main=float(action[0]),
                    action_lateral=float(action[1]),
                )
            )
            result.total_reward += float(reward)

            if render:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            obs = next_obs
            done = terminated or truncated

        env.close()
        mission.episodes.append(result)
        all_frames.append(frames)

    if progress_cb:
        progress_cb(1.0, "Mission complete.")

    return mission, all_frames
