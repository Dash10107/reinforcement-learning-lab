"""
WarehouseEnv — Custom multi-agent grid environment.

N robots coordinate to fulfil delivery tasks:
  1. Navigate to a pickup zone (load package)
  2. Navigate to a delivery zone (unload package, earn reward)
  3. Receive a new task immediately

Observations per robot (normalised to [-1,1] or [0,1]):
  [own_row, own_col,                       # 2
   goal_row, goal_col,                     # 2
   has_package,                            # 1
   rel_robot_1_row, rel_robot_1_col, ...   # 2*(N-1) capped at 4 neighbours
  ]  →  total = 5 + 2*min(N-1,4) dims

Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
"""

from __future__ import annotations
import numpy as np
from warehouse.layout import (
    get_layout, get_open_cells, get_pickup_cells, get_delivery_cells,
    OPEN, SHELF, PICKUP, DELIVERY,
)

ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = 5
MAX_NEIGHBOURS = 4  # how many other robots each agent observes


class RobotState:
    def __init__(self, row: int, col: int, robot_id: int):
        self.row = row
        self.col = col
        self.id = robot_id
        self.has_package = False
        self.goal_row = 0
        self.goal_col = 0
        self.deliveries = 0
        self.collisions = 0


class WarehouseEnv:
    def __init__(
        self,
        n_robots: int = 4,
        max_steps: int = 100,
        seed: int | None = None,
    ):
        self.n_robots = n_robots
        self.max_steps = max_steps
        self.grid = get_layout()
        self.H, self.W = self.grid.shape
        self._open = get_open_cells(self.grid)
        self._pickup = get_pickup_cells(self.grid)
        self._delivery = get_delivery_cells(self.grid)
        self._rng = np.random.default_rng(seed)

        self.obs_dim = 5 + 2 * MAX_NEIGHBOURS  # always padded to 4 neighbours
        self.robots: list[RobotState] = []
        self.step_count = 0
        self.episode_rewards: list[float] = []

    # ── Gymnasium-style API ───────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        starts = self._rng.choice(len(self._open), size=self.n_robots, replace=False)
        self.robots = []
        for i, idx in enumerate(starts):
            r, c = self._open[int(idx)]
            robot = RobotState(r, c, i)
            self._assign_task(robot)
            self.robots.append(robot)

        self.step_count = 0
        self.episode_rewards = [0.0] * self.n_robots
        return self._observations()

    def step(self, actions: list[int]) -> tuple[dict, list[float], bool, dict]:
        rewards = [0.0] * self.n_robots

        # Compute intended next positions
        next_positions: list[tuple[int, int]] = []
        for i, (robot, action) in enumerate(zip(self.robots, actions)):
            dr, dc = ACTIONS[action]
            nr = np.clip(robot.row + dr, 0, self.H - 1)
            nc = np.clip(robot.col + dc, 0, self.W - 1)
            if self.grid[nr, nc] in (SHELF,):  # wall/shelf blocks
                nr, nc = robot.row, robot.col  # stay
            next_positions.append((nr, nc))

        # Collision detection — if two robots want same cell, both stay
        pos_count: dict[tuple, list[int]] = {}
        for i, pos in enumerate(next_positions):
            pos_count.setdefault(pos, []).append(i)

        for pos, robots_at in pos_count.items():
            if len(robots_at) > 1:
                for ri in robots_at:
                    next_positions[ri] = (self.robots[ri].row, self.robots[ri].col)
                    rewards[ri] -= 0.3
                    self.robots[ri].collisions += 1

        # Apply movement
        for i, robot in enumerate(self.robots):
            robot.row, robot.col = next_positions[i]

        # Task logic
        for i, robot in enumerate(self.robots):
            cell = self.grid[robot.row, robot.col]
            gr, gc = robot.goal_row, robot.goal_col

            if not robot.has_package and cell == PICKUP and (robot.row, robot.col) == (gr, gc):
                robot.has_package = True
                rewards[i] += 0.5
                self._assign_delivery(robot)

            elif robot.has_package and cell == DELIVERY and (robot.row, robot.col) == (gr, gc):
                robot.has_package = False
                robot.deliveries += 1
                rewards[i] += 2.0
                self._assign_task(robot)

        # Step penalty (efficiency)
        for i in range(self.n_robots):
            rewards[i] -= 0.01

        for i in range(self.n_robots):
            self.episode_rewards[i] += rewards[i]

        self.step_count += 1
        done = self.step_count >= self.max_steps
        info = {
            "deliveries": sum(r.deliveries for r in self.robots),
            "collisions": sum(r.collisions for r in self.robots),
            "step": self.step_count,
        }
        return self._observations(), rewards, done, info

    # ── Task assignment ───────────────────────────────────────────────────────

    def _assign_task(self, robot: RobotState):
        """Assign robot to pick up from a random pickup cell."""
        idx = int(self._rng.integers(len(self._pickup)))
        robot.goal_row, robot.goal_col = self._pickup[idx]
        robot.has_package = False

    def _assign_delivery(self, robot: RobotState):
        """Assign robot to deliver to a random delivery cell."""
        idx = int(self._rng.integers(len(self._delivery)))
        robot.goal_row, robot.goal_col = self._delivery[idx]

    # ── Observations ─────────────────────────────────────────────────────────

    def _observations(self) -> dict[str, np.ndarray]:
        obs = {}
        for i, robot in enumerate(self.robots):
            own = np.array([
                robot.row / (self.H - 1) * 2 - 1,
                robot.col / (self.W - 1) * 2 - 1,
                robot.goal_row / (self.H - 1) * 2 - 1,
                robot.goal_col / (self.W - 1) * 2 - 1,
                float(robot.has_package),
            ], dtype=np.float32)

            others = []
            for j, other in enumerate(self.robots):
                if j == i:
                    continue
                rel_r = (other.row - robot.row) / (self.H - 1)
                rel_c = (other.col - robot.col) / (self.W - 1)
                others.extend([rel_r, rel_c])
                if len(others) >= 2 * MAX_NEIGHBOURS:
                    break

            # Pad if fewer robots than MAX_NEIGHBOURS
            while len(others) < 2 * MAX_NEIGHBOURS:
                others.extend([0.0, 0.0])

            obs[f"robot_{i}"] = np.concatenate([own, others[:2 * MAX_NEIGHBOURS]]).astype(np.float32)

        return obs

    # ── Episode stats ─────────────────────────────────────────────────────────

    @property
    def total_deliveries(self) -> int:
        return sum(r.deliveries for r in self.robots)

    @property
    def total_collisions(self) -> int:
        return sum(r.collisions for r in self.robots)

    @property
    def agent_ids(self) -> list[str]:
        return [f"robot_{i}" for i in range(self.n_robots)]
