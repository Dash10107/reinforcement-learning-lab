from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    n_agents: int = 5
    local_ratio: float = 0.5
    max_cycles: int = 50
    obs_dim: int = 30
    act_dim: int = 5


@dataclass
class NetworkConfig:
    hidden_sizes: list = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"


@dataclass
class PPOConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 128


@dataclass
class TrainConfig:
    total_episodes: int = 500
    eval_every: int = 25
    save_every: int = 50
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    model_path: str = "shared_actor.pth"


ENV = EnvConfig()
NET = NetworkConfig()
PPO = PPOConfig()
TRAIN = TrainConfig()
