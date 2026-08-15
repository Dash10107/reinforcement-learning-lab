"""
GreenCityEnv — Urban logistics grid environment.

State:   [agent_y, agent_x]  — 2-D position on grid
Action:  Discrete(4)         — Up / Down / Left / Right
Reward:  negative carbon cost per step; +20 on delivery
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

VEHICLE_BASE_COST = {"Diesel": 1.0, "EV": 0.2}
CONGESTION_MULT = 4.0
DELIVERY_BONUS = 20.0

PRESETS = {
    "🏙️ Downtown Rush": {
        "size": 7,
        "congestion": [[2, 2], [2, 3], [3, 2], [4, 4], [4, 5]],
        "start": [0, 0],
        "goal": [6, 6],
        "description": "Dense urban core — heavy congestion cluster in the middle.",
    },
    "🏭 Industrial Port": {
        "size": 8,
        "congestion": [[1, 6], [2, 5], [2, 6], [3, 6], [6, 1], [6, 2]],
        "start": [0, 0],
        "goal": [7, 7],
        "description": "Port logistics — congestion near loading docks.",
    },
    "🛣️ Ring Road": {
        "size": 7,
        "congestion": [[1, 1], [1, 5], [5, 1], [5, 5], [3, 3]],
        "start": [0, 3],
        "goal": [6, 3],
        "description": "Congestion at ring-road junctions — clear central corridor.",
    },
    "🏕️ Open Field": {
        "size": 6,
        "congestion": [],
        "start": [0, 0],
        "goal": [5, 5],
        "description": "Minimal congestion — baseline comparison scenario.",
    },
}


class GreenCityEnv(gym.Env):
    """Fixed and complete urban logistics environment."""

    metadata = {"render_modes": []}

    def __init__(self, size: int = 7):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)
        self._moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        # Will be set on reset
        self.agent_pos = np.zeros(2, dtype=np.int32)
        self.goal = np.array([size - 1, size - 1], dtype=np.int32)
        self.congestion = []
        self.vehicle = "Diesel"
        self.steps = 0

    def reset(
        self,
        seed: int | None = None,
        start_pos: list[int] | None = None,
        goal_pos: list[int] | None = None,
        congestion_map: list | None = None,
        vehicle: str = "Diesel",
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self.agent_pos = np.array(start_pos or [0, 0], dtype=np.int32)
        self.goal = np.array(goal_pos or [self.size - 1, self.size - 1], dtype=np.int32)
        self.congestion = [np.array(c, dtype=np.int32) for c in (congestion_map or [])]
        self.vehicle = vehicle
        self.steps = 0
        return self.agent_pos.copy(), {}

    def step(self, action: int):
        new_pos = np.clip(self.agent_pos + self._moves[action], 0, self.size - 1)
        self.agent_pos = new_pos
        self.steps += 1

        in_congestion = any(np.array_equal(new_pos, c) for c in self.congestion)
        base_cost = VEHICLE_BASE_COST.get(self.vehicle, 1.0)
        multiplier = CONGESTION_MULT if in_congestion else 1.0
        reward = -(base_cost * multiplier)

        done = bool(np.array_equal(self.agent_pos, self.goal))
        if done:
            reward += DELIVERY_BONUS

        truncated = self.steps >= self.size * 5
        info = {"in_congestion": in_congestion, "carbon_step": base_cost * multiplier}
        return self.agent_pos.copy(), float(reward), done, truncated, info


def parse_congestion(text: str, size: int) -> list[list[int]]:
    """Parse 'y,x; y,x; ...' string into list of [y,x] coords."""
    zones = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals = [
                int(v) for v in part.split() if v.isdigit() or (v.lstrip("-").isdigit())
            ]
            if len(vals) >= 2:
                y, x = vals[0], vals[1]
                if 0 <= y < size and 0 <= x < size:
                    zones.append([y, x])
        except ValueError:
            pass
    return zones
