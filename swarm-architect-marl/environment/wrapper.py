from mpe2 import simple_spread_v3
from config import EnvConfig


def make_env(cfg: EnvConfig, render: bool = False) -> object:
    return simple_spread_v3.parallel_env(
        N=cfg.n_agents,
        local_ratio=cfg.local_ratio,
        max_cycles=cfg.max_cycles,
        render_mode="rgb_array" if render else None,
    )


def get_agent_ids(env) -> list[str]:
    env.reset()
    return list(env.agents)
