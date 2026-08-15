"""
Three routing strategies — all implement the same interface.

DQNAgent      — trained Deep Q-Network (falls back to greedy if no model)
GreedyAgent   — always moves toward goal (Manhattan heuristic)
AStarAgent    — true optimal path ignoring emission cost (distance upper bound)
"""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field

import numpy as np

from core.env import CONGESTION_MULT, VEHICLE_BASE_COST, GreenCityEnv

MODEL_PATH = "green_dqn_model"

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    SB3 = True
except ImportError:
    SB3 = False


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class RouteResult:
    agent_name: str
    path: list[list[int]] = field(default_factory=list)
    carbon_per_step: list[float] = field(default_factory=list)
    total_carbon: float = 0.0
    congestion_hits: int = 0
    delivered: bool = False
    steps: int = 0

    @property
    def total_reward(self) -> float:
        return -self.total_carbon + (20.0 if self.delivered else 0.0)


# ── Base rollout ──────────────────────────────────────────────────────────────


def _rollout(env: GreenCityEnv, action_fn) -> RouteResult:
    """Generic rollout using a provided action function."""
    result = RouteResult(agent_name="")
    result.path.append(env.agent_pos.tolist())

    for _ in range(env.size * 5):
        action = action_fn(env.agent_pos.copy(), env)
        obs, _, done, trunc, info = env.step(int(action))
        result.path.append(obs.tolist())
        result.carbon_per_step.append(info["carbon_step"])
        result.total_carbon += info["carbon_step"]
        if info["in_congestion"]:
            result.congestion_hits += 1
        if done:
            result.delivered = True
            break
        if trunc:
            break

    result.steps = len(result.path) - 1
    return result


# ── Greedy agent ──────────────────────────────────────────────────────────────


def _greedy_action(pos: np.ndarray, env: GreenCityEnv) -> int:
    dy = env.goal[0] - pos[0]
    dx = env.goal[1] - pos[1]
    if abs(dy) >= abs(dx):
        return 1 if dy > 0 else 0
    return 3 if dx > 0 else 2


def run_greedy(env: GreenCityEnv) -> RouteResult:
    result = _rollout(env, _greedy_action)
    result.agent_name = "Greedy"
    return result


# ── A* optimal (distance-only) ────────────────────────────────────────────────


def _astar_path(env: GreenCityEnv) -> list[tuple[int, int]]:
    """A* search minimising step count (not carbon)."""
    start = tuple(env.agent_pos.tolist())
    goal = tuple(env.goal.tolist())
    size = env.size

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_heap: list = [(h(start), 0, start, [start])]
    visited: set = set()

    while open_heap:
        _, g, cur, path = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            return path
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (np.clip(cur[0] + dy, 0, size - 1), np.clip(cur[1] + dx, 0, size - 1))
            if nb not in visited:
                heapq.heappush(open_heap, (g + 1 + h(nb), g + 1, nb, path + [nb]))

    return [start, goal]


def run_astar(env: GreenCityEnv) -> RouteResult:
    path_tuples = _astar_path(env)
    path = [list(p) for p in path_tuples]
    base = VEHICLE_BASE_COST.get(env.vehicle, 1.0)
    carbon_steps: list[float] = []
    total_c = 0.0
    hits = 0

    for p in path[1:]:
        in_c = any(np.array_equal(p, c) for c in env.congestion)
        cost = base * (CONGESTION_MULT if in_c else 1.0)
        carbon_steps.append(cost)
        total_c += cost
        if in_c:
            hits += 1

    return RouteResult(
        agent_name="A* Optimal",
        path=path,
        carbon_per_step=carbon_steps,
        total_carbon=total_c,
        congestion_hits=hits,
        delivered=True,
        steps=len(path) - 1,
    )


# ── DQN agent ─────────────────────────────────────────────────────────────────


def _load_dqn(env: GreenCityEnv):
    if not SB3:
        return None
    try:
        return DQN.load(MODEL_PATH, env=env)
    except Exception:
        return None


def run_dqn(env: GreenCityEnv) -> RouteResult:
    model = _load_dqn(env)
    if model is None:
        result = run_greedy(env)
        result.agent_name = "DQN (→ Greedy fallback)"
        return result

    def _dqn_action(pos, e):
        action, _ = model.predict(pos.astype(np.float32), deterministic=True)
        return int(action)

    result = _rollout(env, _dqn_action)
    result.agent_name = "DQN Agent"
    return result


# ── Training state ────────────────────────────────────────────────────────────


@dataclass
class TrainingState:
    running: bool = False
    timestep: int = 0
    total: int = 0
    ep_rewards: list[float] = field(default_factory=list)
    status: str = "idle"
    model_ready: bool = False


class _LiveCB(BaseCallback):
    def __init__(self, state: TrainingState):
        super().__init__()
        self._s = state

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
            f"| Eps: {n_ep} | Avg reward: {roll:.2f}"
        )
        return True

    def _on_training_end(self) -> None:
        self._s.running = False
        self._s.model_ready = True
        self._s.status = f"Training complete — {self._s.timestep:,} steps."


def start_training(total_steps: int, state: TrainingState) -> threading.Thread:
    def _run():
        state.running = True
        state.total = total_steps
        train_env = Monitor(GreenCityEnv(size=7))
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=500,
            batch_size=64,
            gamma=0.95,
            train_freq=4,
            target_update_interval=200,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=0,
        )
        model.learn(
            total_timesteps=total_steps, callback=_LiveCB(state), progress_bar=False
        )
        model.save(MODEL_PATH)
        train_env.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
