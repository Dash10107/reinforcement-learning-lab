from __future__ import annotations
import numpy as np


class TabularAgent:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def choose_action(self, state: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self, rate: float):
        self.epsilon = max(0.01, self.epsilon * rate)
