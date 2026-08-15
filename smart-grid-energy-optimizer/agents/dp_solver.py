"""
Dynamic Programming (Value Iteration) solver.
Requires deterministic price/solar/load knowledge — acts as a theoretical upper bound.
"""

from __future__ import annotations

import numpy as np
from env.grid_env import ACTION_TO_KW, SmartGridEnv


def solve_dp(
    env: SmartGridEnv, n_soc_levels: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward induction on discretised SOC states.
    Returns (policy, value) arrays of shape (24, n_soc_levels).
    """
    soc_grid = np.linspace(0, env.cap, n_soc_levels)
    V = np.zeros((25, n_soc_levels))  # terminal value = 0
    policy = np.zeros((24, n_soc_levels), dtype=int)

    prices = env._prices
    solar = env._solar
    load = env._load

    for t in reversed(range(24)):
        for si, soc in enumerate(soc_grid):
            best_val = -1e9
            best_act = 3  # hold
            for ai, kw in enumerate(ACTION_TO_KW):
                # Apply action
                if kw > 0:
                    kw_actual = min(kw, env.max_rate, (env.cap - soc))
                    new_soc = soc + kw_actual
                elif kw < 0:
                    kw_actual = max(kw, -env.max_rate, -soc / env.eff)
                    new_soc = soc + kw_actual * env.eff
                else:
                    kw_actual = 0.0
                    new_soc = soc

                new_soc = np.clip(new_soc, 0, env.cap)

                # Reward (deterministic)
                price = prices[t]
                solar_now = solar[t]
                load_now = load[t]
                solar_to_load = min(solar_now, load_now)
                solar_excess = solar_now - solar_to_load
                load_after_solar = load_now - solar_to_load

                r = 0.0
                if kw_actual > 0:
                    from_solar = min(kw_actual, solar_excess)
                    from_grid = kw_actual - from_solar
                    r -= from_grid * price / 100.0
                elif kw_actual < 0:
                    discharge = -kw_actual * env.eff
                    to_load = min(discharge, load_after_solar)
                    to_grid = discharge - to_load
                    r += (to_load + to_grid) * price / 100.0

                if kw_actual <= 0:
                    residual = max(0, load_after_solar - max(0, -kw_actual * env.eff))
                else:
                    residual = load_after_solar
                r -= residual * price / 100.0

                if kw_actual >= 0:
                    solar_to_battery = min(solar_excess, kw_actual)
                    remaining_solar = solar_excess - solar_to_battery
                    r += remaining_solar * price / 100.0 * 0.5

                # Lookup next value
                next_si = np.searchsorted(soc_grid, new_soc)
                next_si = np.clip(next_si, 0, n_soc_levels - 1)
                total = r + V[t + 1, next_si]

                if total > best_val:
                    best_val = total
                    best_act = ai

            V[t, si] = best_val
            policy[t, si] = best_act

    return policy, V, soc_grid


def rollout_dp(env: SmartGridEnv, policy: np.ndarray, soc_grid: np.ndarray) -> float:
    """Execute the DP policy on the environment."""
    obs, _ = env.reset()  # noqa: RUF059
    env.soc = 0.0
    env.hour = 0
    env.log = env._empty_log()

    for t in range(24):
        si = np.searchsorted(soc_grid, env.soc)
        si = np.clip(si, 0, len(soc_grid) - 1)
        action = int(policy[t, si])
        env.step(action)

    return env.total_reward
