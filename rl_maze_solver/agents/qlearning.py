from __future__ import annotations

import numpy as np
from maze.env import MazeEnv

from agents.base import TabularAgent


def train_qlearning(
    env: MazeEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    decay: float,
    seed: int = 0,
) -> tuple[TabularAgent, list[float]]:
    agent = TabularAgent(env.n_states, env.action_space.n, alpha, gamma)
    rng = np.random.default_rng(seed)
    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        total = 0.0
        for _ in range(env.n_states * 4):
            action = agent.choose_action(state, rng)
            next_state, reward, done, _, _ = env.step(action)
            # Q-Learning: off-policy TD update
            td_target = reward + gamma * np.max(agent.Q[next_state]) * (1 - done)
            agent.Q[state, action] += alpha * (td_target - agent.Q[state, action])
            state = next_state
            total += reward
            if done:
                break
        agent.decay_epsilon(decay)
        rewards.append(total)

    return agent, rewards
