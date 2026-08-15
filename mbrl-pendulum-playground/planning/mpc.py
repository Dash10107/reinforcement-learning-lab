"""
Model Predictive Control via Random Shooting.

At every step:
  1. Sample N random action sequences of length H
  2. Roll each sequence out in the learned dynamics model
  3. Sum predicted rewards (Pendulum-v1 objective)
  4. Execute the first action of the best sequence
  5. Repeat from new state (receding horizon)

This is the simplest, most transparent form of MBRL planning.
"""

from __future__ import annotations

import numpy as np
import torch
from model.dynamics import pendulum_reward_batch


def random_shooting_mpc(
    current_state: np.ndarray,
    dynamics_model,
    horizon: int = 20,
    n_samples: int = 512,
    action_low: float = -2.0,
    action_high: float = 2.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns:
      best_action  — scalar torque to apply now
      all_rewards  — (n_samples,) total reward per candidate sequence
      best_actions — (horizon,) full planned action sequence
    """
    # Sample action sequences: (n_samples, horizon, 1)
    actions = torch.FloatTensor(n_samples, horizon, 1).uniform_(action_low, action_high)

    # Expand starting state to all candidates: (n_samples, 3)
    states = torch.FloatTensor(current_state).unsqueeze(0).expand(n_samples, -1).clone()

    total_rewards = torch.zeros(n_samples)

    with torch.no_grad():
        for h in range(horizon):
            a_h = actions[:, h, :]  # (n_samples, 1)
            r_h = pendulum_reward_batch(states, a_h)  # (n_samples,)
            total_rewards += r_h * (0.99**h)  # discounted
            states = dynamics_model(states, a_h)  # (n_samples, 3)

    best_idx = total_rewards.argmax().item()
    best_action = float(actions[best_idx, 0, 0].item())
    return best_action, total_rewards.numpy(), actions[best_idx, :, 0].numpy()


def run_mpc_episode(
    dynamics_model,
    horizon: int = 20,
    n_samples: int = 512,
    max_steps: int = 200,
    render: bool = True,
    seed: int = 0,
) -> dict:
    """
    Run a full MPC-controlled episode in the real Pendulum-v1 environment.
    Returns a dict with frames, rewards, actions, states, planned_actions.
    """
    import gymnasium as gym

    env = gym.make("Pendulum-v1", render_mode="rgb_array" if render else None)
    obs, _ = env.reset(seed=seed)
    dynamics_model.eval()

    frames, rewards, actions_taken, states_log, planned_seqs = [], [], [], [], []

    for step in range(max_steps):
        best_action, all_rewards, planned = random_shooting_mpc(  # noqa: RUF059
            obs, dynamics_model, horizon=horizon, n_samples=n_samples
        )

        next_obs, reward, term, trunc, _ = env.step(np.array([best_action]))

        if render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        rewards.append(float(reward))
        actions_taken.append(best_action)
        states_log.append(obs.copy())
        planned_seqs.append(planned.copy())
        obs = next_obs

        if term or trunc:
            break

    env.close()
    return {
        "frames": frames,
        "rewards": rewards,
        "actions": actions_taken,
        "states": states_log,
        "planned": planned_seqs,
        "total_reward": sum(rewards),
        "steps": len(rewards),
    }


def run_random_episode(max_steps: int = 200, seed: int = 0) -> dict:
    """Baseline: random torque at every step."""
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    rewards, actions = [], []

    for _ in range(max_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)  # noqa: RUF059
        rewards.append(float(r))
        actions.append(float(a[0]))
        if term or trunc:
            break

    env.close()
    return {"rewards": rewards, "actions": actions, "total_reward": sum(rewards)}
