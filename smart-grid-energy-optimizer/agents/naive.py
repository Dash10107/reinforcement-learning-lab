"""
Rule-based baselines for comparison with RL.
"""

from __future__ import annotations

import numpy as np
from env.grid_env import SmartGridEnv


def rollout_naive(env: SmartGridEnv, threshold_pct: float = 0.5) -> float:
    """
    Price-threshold heuristic:
    - Charge at max when price < threshold_pct × median price
    - Discharge at max when price > (1 + threshold_pct) × median price
    - Hold otherwise
    """
    prices = env._prices
    median_p = np.median(prices)
    low_thresh = median_p * (1 - threshold_pct * 0.5)
    high_thresh = median_p * (1 + threshold_pct * 0.5)

    obs, _ = env.reset()  # noqa: RUF059
    env.soc = 0.0
    env.hour = 0
    env.log = env._empty_log()

    for t in range(24):
        p = prices[t]
        if p <= low_thresh and env.soc < env.cap * 0.9:
            action = 6  # charge max
        elif p >= high_thresh and env.soc > env.cap * 0.1:
            action = 0  # discharge max
        else:
            action = 3  # hold

        env.step(action)

    return env.total_reward


def rollout_no_battery(env: SmartGridEnv) -> float:
    """Pure grid + solar (no battery storage) — shows value of storage."""
    obs, _ = env.reset()  # noqa: RUF059
    env.soc = 0.0
    env.hour = 0
    env.log = env._empty_log()

    for t in range(24):
        env.step(3)  # always hold (no storage action)

    return env.total_reward
