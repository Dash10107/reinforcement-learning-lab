"""
A2C agent loading, inference, and training utilities.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from core.environment import AITutorEnv

MODEL_PATH = "tutor_model"


@dataclass
class TrainingState:
    running: bool = False
    timestep: int = 0
    total: int = 0
    ep_rewards: list[float] = field(default_factory=list)
    status: str = "idle"
    model_ready: bool = False


class _LiveCallback(BaseCallback):
    def __init__(self, state: TrainingState, log_every: int = 200):
        super().__init__()
        self._s = state
        self._le = log_every

    def _on_step(self) -> bool:
        if not self._s.running:
            return False
        self._s.timestep = self.num_timesteps
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._s.ep_rewards.append(float(info["episode"]["r"]))
        pct = self._s.timestep / max(self._s.total, 1) * 100
        n_ep = len(self._s.ep_rewards)
        roll = float(np.mean(self._s.ep_rewards[-20:])) if self._s.ep_rewards else 0
        self._s.status = (
            f"{pct:.0f}% | Step {self._s.timestep:,}/{self._s.total:,} "
            f"| Episodes: {n_ep} | Rolling reward: {roll:.3f}"
        )
        return True

    def _on_training_end(self) -> None:
        self._s.running = False
        self._s.model_ready = True
        self._s.status = f"Training complete — {self._s.timestep:,} steps."


def load_model(path: str = MODEL_PATH) -> A2C:
    env = AITutorEnv()
    try:
        m = A2C.load(path, env=env)
        return m
    except Exception:
        m = A2C("MlpPolicy", env, verbose=0)
        m.learn(total_timesteps=10_000)
        m.save(path)
        return m


def get_policy_probs(model: A2C, proficiency_pct: list[float]) -> np.ndarray:
    """Return action probability distribution for a given state."""
    obs = np.array(proficiency_pct, dtype=np.float32) / 100.0
    tensor = model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
    dist = model.policy.get_distribution(tensor)
    return dist.distribution.probs.detach().cpu().numpy()[0]


def simulate_path(
    model: A2C,
    start_pct: list[float],
    n_steps: int = 20,
    deterministic: bool = True,
) -> list[dict]:
    """
    Roll out the policy for n_steps.
    Returns a list of step-dicts with state, action, reward, probs.
    """
    env = AITutorEnv()
    env.set_state(start_pct)
    history = []

    for step in range(n_steps):
        obs = env.state.copy()
        probs = get_policy_probs(model, (obs * 100).tolist())
        action, _ = model.predict(obs, deterministic=deterministic)
        _next_obs, reward, done, trunc, _info = env.step(int(action))

        history.append(
            {
                "step": step + 1,
                "action": int(action),
                "state": (obs * 100).tolist(),
                "reward": reward,
                "probs": probs.tolist(),
                "done": done or trunc,
            }
        )
        if done or trunc:
            break

    return history


def start_training(
    total_steps: int,
    state: TrainingState,
    save_path: str = MODEL_PATH,
) -> threading.Thread:

    def _run():
        state.running = True
        state.total = total_steps
        env = Monitor(AITutorEnv())
        model = A2C(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=7e-4,
            gamma=0.99,
            n_steps=5,
            ent_coef=0.01,
        )
        cb = _LiveCallback(state)
        model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False)
        model.save(save_path)
        env.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
