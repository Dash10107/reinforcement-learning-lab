# Starter Templates

Three minimal implementations — one algorithm each, one file each. No framework. No boilerplate. Just the algorithm, clearly written, with every line annotated.

Fork one. Swap in your environment. You'll have a working RL agent in under an hour.

---

## Template 1: DQN

For discrete action spaces with large or visual state spaces.

```python
"""
DQN Starter Template
====================
Works with any Gymnasium environment that has a Discrete action space.
Usage: python dqn_starter.py --env CartPole-v1
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import argparse

# ─── Hyperparameters ─────────────────────────────────────────────────────────

LR = 1e-3  # learning rate
GAMMA = 0.99  # discount factor
BATCH_SIZE = 64  # samples per training step
BUFFER_SIZE = 10_000  # replay buffer capacity
EPSILON_START = 1.0  # initial exploration probability
EPSILON_END = 0.01  # minimum exploration probability
EPSILON_DECAY = 0.995  # multiplicative decay per episode
TARGET_UPDATE = 100  # update target network every N steps
N_EPISODES = 500  # total training episodes

# ─── Network ─────────────────────────────────────────────────────────────────


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),  # one Q-value per action
        )

    def forward(self, x):
        return self.net(x)


# ─── Replay Buffer ────────────────────────────────────────────────────────────


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Agent ───────────────────────────────────────────────────────────────────


class DQNAgent:
    def __init__(self, state_dim, n_actions):
        self.n_actions = n_actions
        self.online_net = QNetwork(state_dim, n_actions)
        self.target_net = QNetwork(state_dim, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)  # explore
        with torch.no_grad():
            q_values = self.online_net(torch.FloatTensor(state))
            return q_values.argmax().item()  # exploit

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Current Q-values for the actions that were taken
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values: r + γ * max Q(s', a') — computed using the target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1).values
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# ─── Training Loop ────────────────────────────────────────────────────────────


def train(env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(state_dim, n_actions)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.update()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if episode % 50 == 0:
            print(
                f"Ep {episode:4d} | Reward: {total_reward:7.1f} | ε: {agent.epsilon:.3f}"
            )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    args = parser.parse_args()
    train(args.env)
```

**Try it on:**
- `CartPole-v1` — balances in ~100 episodes
- `LunarLander-v2` — harder, needs ~500 episodes
- Any `Discrete` action space environment from Gymnasium

---

## Template 2: PPO

For both discrete and continuous action spaces. More stable than DQN in most settings.

```python
"""
PPO Starter Template
====================
Works with discrete or continuous Gymnasium environments.
Set CONTINUOUS = True/False based on your action space.
Usage: python ppo_starter.py --env CartPole-v1
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import argparse

# ─── Hyperparameters ─────────────────────────────────────────────────────────

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95  # λ for Generalised Advantage Estimation
CLIP_EPS = 0.2  # PPO clipping range [1-ε, 1+ε]
N_EPOCHS = 4  # gradient update passes per rollout
ROLLOUT_LEN = 2048  # steps to collect before each update
BATCH_SIZE = 64
CONTINUOUS = False  # set True for continuous action spaces

# ─── Actor-Critic Network ─────────────────────────────────────────────────────


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
        if CONTINUOUS:
            # Learnable log standard deviation (not state-dependent)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        features = self.shared(state)
        value = self.critic(features)
        if CONTINUOUS:
            mean = self.actor(features)
            std = self.log_std.exp().expand_as(mean)
            return Normal(mean, std), value
        else:
            logits = self.actor(features)
            return Categorical(logits=logits), value


# ─── Generalised Advantage Estimation ────────────────────────────────────────


def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    GAE smoothly combines TD(0) and Monte Carlo returns.
    Higher lam = more Monte Carlo (lower bias, higher variance).
    """
    advantages = []
    gae = 0
    values = values + [next_value]
    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        )
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


# ─── PPO Update ──────────────────────────────────────────────────────────────


def ppo_update(net, optimizer, states, actions, old_log_probs, advantages, returns):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(N_EPOCHS):
        # Shuffle data for each epoch
        indices = torch.randperm(len(states))
        for start in range(0, len(states), BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]

            dist, values = net(states[idx])
            new_log_probs = dist.log_prob(actions[idx])
            if CONTINUOUS:
                new_log_probs = new_log_probs.sum(-1)

            # PPO clipped objective
            ratio = (new_log_probs - old_log_probs[idx]).exp()
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            actor_loss = -torch.min(
                ratio * advantages[idx], clipped * advantages[idx]
            ).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns[idx])
            entropy_loss = -dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()


# ─── Training Loop ────────────────────────────────────────────────────────────


def train(env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if CONTINUOUS else env.action_space.n

    net = ActorCritic(state_dim, action_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    state, _ = env.reset()
    episode_rewards = []
    current_reward = 0

    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    for step in range(1_000_000):
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            dist, value = net(state_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if CONTINUOUS:
                log_prob = log_prob.sum()

        next_state, reward, terminated, truncated, _ = env.step(
            action.numpy() if CONTINUOUS else action.item()
        )
        done = terminated or truncated

        states.append(state_t)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        log_probs.append(log_prob)
        values.append(value.item())

        state = next_state
        current_reward += reward

        if done:
            episode_rewards.append(current_reward)
            current_reward = 0
            state, _ = env.reset()

        # Update every ROLLOUT_LEN steps
        if (step + 1) % ROLLOUT_LEN == 0:
            with torch.no_grad():
                _, next_value = net(torch.FloatTensor(state))
            advantages, returns = compute_gae(rewards, values, dones, next_value.item())

            ppo_update(
                net,
                optimizer,
                torch.stack(states),
                torch.stack(actions),
                torch.stack(log_probs).detach(),
                torch.FloatTensor(advantages),
                torch.FloatTensor(returns),
            )
            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

            if episode_rewards:
                print(
                    f"Step {step:7d} | Mean reward (last 10): "
                    f"{np.mean(episode_rewards[-10:]):7.1f}"
                )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    args = parser.parse_args()
    train(args.env)
```

**Try it on:**
- `CartPole-v1` — discrete, solves quickly
- `Pendulum-v1` — set `CONTINUOUS = True`, continuous control
- `LunarLanderContinuous-v2` — continuous, more realistic

---

## How to adapt these templates

**Change the environment:**
```python
env = gym.make("YourCustomEnv-v0")
```

**Change the network architecture:**
Replace the `nn.Sequential` in `QNetwork` or `ActorCritic` with anything — convolutional layers for pixel inputs, recurrent layers for partial observability, larger hidden sizes for more complex tasks.

**Add multi-agent support (IPPO):**
Create one `DQNAgent` or `ActorCritic` per agent. Run them independently. Each agent sees its own observation and updates from its own replay buffer or rollout.

**Plug in a custom reward:**
The only place reward enters is in the training loop: `reward = your_reward_function(state, action, next_state)`. Replace it with anything.

[← Back to Resources](../)
