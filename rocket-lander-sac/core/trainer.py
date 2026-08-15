"""
SAC training pipeline — fine-tune or train from scratch with live callbacks.
"""

from __future__ import annotations
import os
import threading
from dataclasses import dataclass, field
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np


@dataclass
class TrainingState:
    running: bool = False
    timestep: int = 0
    total_timesteps: int = 0
    episode_rewards: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    ent_coefs: list[float] = field(default_factory=list)
    log_steps: list[int] = field(default_factory=list)
    status: str = "idle"
    best_reward: float = float("-inf")


class _LiveCallback(BaseCallback):
    def __init__(self, state: TrainingState, log_interval: int = 500):
        super().__init__()
        self._state = state
        self._log_interval = log_interval
        self._ep_rewards: list[float] = []

    def _on_step(self) -> bool:
        if not self._state.running:
            return False  # abort training

        self._state.timestep = self.num_timesteps

        # Collect episode rewards from monitor wrapper
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                r = float(info["episode"]["r"])
                self._ep_rewards.append(r)
                self._state.episode_rewards.append(r)
                if r > self._state.best_reward:
                    self._state.best_reward = r

        if self.num_timesteps % self._log_interval == 0:
            losses = self.model.logger.name_to_value
            self._state.actor_losses.append(float(losses.get("train/actor_loss", 0)))
            self._state.critic_losses.append(float(losses.get("train/critic_loss", 0)))
            self._state.ent_coefs.append(float(losses.get("train/ent_coef", 0)))
            self._state.log_steps.append(self.num_timesteps)

        pct = self.num_timesteps / max(self._state.total_timesteps, 1)
        rolling = float(np.mean(self._ep_rewards[-20:])) if self._ep_rewards else 0.0
        self._state.status = (
            f"Step {self.num_timesteps:,}/{self._state.total_timesteps:,} "
            f"({pct*100:.1f}%)  |  Rolling reward: {rolling:+.1f}  |  "
            f"Best: {self._state.best_reward:+.1f}"
        )
        return True

    def _on_training_end(self) -> None:
        self._state.status = (
            f"Training complete — {self.num_timesteps:,} steps.  "
            f"Best reward: {self._state.best_reward:+.1f}"
        )
        self._state.running = False


def start_training(
    base_model_path: str,
    total_timesteps: int,
    learning_rate: float,
    batch_size: int,
    state: TrainingState,
    save_path: str = "sac_finetuned.zip",
) -> threading.Thread:
    """Launches training in a daemon thread. Progress written to `state`."""

    def _train():
        from stable_baselines3.common.monitor import Monitor

        state.running = True
        state.total_timesteps = total_timesteps
        state.status = "Initialising environment…"

        env = Monitor(gym.make("LunarLander-v3", continuous=True))

        if os.path.exists(base_model_path):
            model = SAC.load(base_model_path, env=env)
            model.learning_rate = learning_rate
            model.batch_size = batch_size
        else:
            model = SAC(
                "MlpPolicy", env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                verbose=0,
            )

        cb = _LiveCallback(state, log_interval=max(total_timesteps // 200, 200))
        model.learn(
            total_timesteps=total_timesteps,
            callback=cb,
            reset_num_timesteps=False,
            progress_bar=False,
            log_interval=1,
        )
        model.save(save_path)
        env.close()

    thread = threading.Thread(target=_train, daemon=True)
    thread.start()
    return thread
