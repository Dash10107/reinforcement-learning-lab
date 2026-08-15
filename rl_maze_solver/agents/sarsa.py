from __future__ import annotations
import numpy as np
from agents.base import TabularAgent
from maze.env import MazeEnv


def train_sarsa(
    env: MazeEnv, episodes: int, alpha: float, gamma: float,
    decay: float, seed: int = 0,
) -> tuple[TabularAgent, list[float]]:
    agent = TabularAgent(env.n_states, env.action_space.n, alpha, gamma)
    rng = np.random.default_rng(seed)
    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        action = agent.choose_action(state, rng)
        total = 0.0
        for _ in range(env.n_states * 4):
            next_state, reward, done, _, _ = env.step(action)
            next_action = agent.choose_action(next_state, rng)
            # SARSA: on-policy TD update (uses next chosen action, not greedy)
            td_target = reward + gamma * agent.Q[next_state, next_action] * (1 - done)
            agent.Q[state, action] += alpha * (td_target - agent.Q[state, action])
            state, action = next_state, next_action
            total += reward
            if done:
                break
        agent.decay_epsilon(decay)
        rewards.append(total)

    return agent, rewards
