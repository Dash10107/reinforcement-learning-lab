"""
Enhanced MazeEnv — works with both DFS corridor mazes and open random mazes.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid: np.ndarray):
        super().__init__()
        self.grid = grid.copy()
        self.H, self.W = grid.shape
        self.n_states = self.H * self.W
        self.start = (0, 0)
        self.goal = (self.H - 1, self.W - 1)

        # Ensure start/goal are open
        self.grid[self.start] = 0
        self.grid[self.goal] = 0

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)  # up down left right
        self._MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.agent_pos = self.start

    def _to_state(self, r: int, c: int) -> int:
        return r * self.W + c

    def _from_state(self, s: int) -> tuple[int, int]:
        return divmod(s, self.W)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start
        return self._to_state(*self.agent_pos), {}

    def step(self, action: int):
        dr, dc = self._MOVES[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # Boundary / wall check
        if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] == 0:
            self.agent_pos = (nr, nc)
            reward = -1
        else:
            reward = -5  # bump penalty (less harsh than before so agent explores)

        done = self.agent_pos == self.goal
        if done:
            reward = 100

        return self._to_state(*self.agent_pos), reward, done, False, {}

    @property
    def shape(self) -> tuple[int, int]:
        return self.H, self.W
