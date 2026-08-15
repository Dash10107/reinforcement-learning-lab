"""
DQN agent using Stable-Baselines3.
Wraps training and rollout for the SmartGridEnv.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import numpy as np
from env.grid_env import SmartGridEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


@dataclass
class TrainingState:
    running: bool = False
    timestep: int = 0
    total_timesteps: int = 0
    episode_rewards: list[float] = field(default_factory=list)
    mean_rewards: list[float] = field(default_factory=list)  # rolling mean
    status: str = "idle"
    best_reward: float = float("-inf")


class _LiveCallback(BaseCallback):
    def __init__(self, state: TrainingState, log_interval: int = 24):
        super().__init__()
        self._state = state
        self._log_interval = log_interval

    def _on_step(self) -> bool:
        if not self._state.running:
            return False
        self._state.timestep = self.num_timesteps
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = float(info["episode"]["r"])
                self._state.episode_rewards.append(r)
                n = len(self._state.episode_rewards)
                roll = float(np.mean(self._state.episode_rewards[max(0, n - 20) :]))
                self._state.mean_rewards.append(roll)
                self._state.best_reward = max(self._state.best_reward, r)
        pct = self._state.timestep / max(self._state.total_timesteps, 1) * 100
        n_ep = len(self._state.episode_rewards)
        roll = self._state.mean_rewards[-1] if self._state.mean_rewards else 0
        self._state.status = (
            f"{pct:.0f}% | Episode {n_ep} | "
            f"Rolling reward: ${roll:+.3f} | Best: ${self._state.best_reward:+.3f}"
        )
        return True

    def _on_training_end(self) -> None:
        self._state.running = False
        self._state.status = (
            f"Training complete — {self._state.timestep:,} steps | "
            f"Best: ${self._state.best_reward:+.3f}"
        )


def start_training(
    env_kwargs: dict,
    total_timesteps: int,
    learning_rate: float,
    batch_size: int,
    state: TrainingState,
    save_path: str = "dqn_smartgrid.zip",
) -> threading.Thread:

    def _train():
        state.running = True
        state.total_timesteps = total_timesteps

        env = Monitor(SmartGridEnv(**env_kwargs, stochastic=True))
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=min(50_000, total_timesteps),
            learning_starts=min(500, total_timesteps // 4),
            gamma=0.97,
            train_freq=24,  # train every episode
            target_update_interval=240,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            policy_kwargs={"net_arch": [128, 128, 64]},
            verbose=0,
        )
        cb = _LiveCallback(state)
        model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)
        model.save(save_path)
        env.close()

    t = threading.Thread(target=_train, daemon=True)
    t.start()
    return t


def rollout_dqn(env: SmartGridEnv, model_path: str) -> float | None:
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        return None
    model = DQN.load(model_path)
    obs, _ = env.reset()
    env.soc = 0.0
    env.hour = 0
    env.log = env._empty_log()

    for _ in range(24):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))
        if done:
            break

    return env.total_reward
