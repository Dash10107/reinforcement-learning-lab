---
title: Swarm Architect Marl
emoji: 🤖
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Swarm Architect — Multi-Agent Cooperative Control with IPPO

<p align="center">
  <a href="https://dash10107.github.io/reinforcement-learning-lab/en/"><img src="https://img.shields.io/badge/Course_Chapter-Read-blue?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="Course Chapter"></a>
  <a href="https://huggingface.co/spaces/Dash10107/swarm-architect-marl"><img src="https://img.shields.io/badge/Live_Demo-Hugging_Face-yellow?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Demo"></a>
</p>

A multi-agent reinforcement learning system where five agents learn to coordinate and spread across a shared space to cover as many landmarks as possible. Each agent is trained independently using PPO and must figure out through experience alone how to claim a different landmark without colliding with its teammates. You can train the agents, watch them run, evaluate their performance, and explore the algorithm in detail.

**This project is part of the [Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab) — an interactive course and lab that bridges the gap between RL theory and practical implementation.**

---

## What problem does this solve?

Imagine five people trying to stand on five different marked spots on a floor, but they cannot communicate and cannot see the full picture — only their immediate surroundings. If they all rush toward the same spot, most of them miss. The optimal strategy requires each person to implicitly understand that others are heading somewhere and choose a different target.

This is a cooperative coverage problem. It appears in real applications: teams of drones covering a disaster area for search and rescue, sensor networks that need to spread to provide full coverage, or autonomous vehicles positioning themselves to monitor intersections.

The challenge for RL is that what one agent does changes what the others should do. This interdependency makes the learning problem much harder than single-agent RL.

---

## The Task: Simple Spread

The environment is the Simple Spread cooperative task from PettingZoo's Multi-Particle Environment suite. Five agents and five landmarks are placed in a continuous 2D space. The team earns a negative reward for each landmark that is not covered by at least one agent, and an additional penalty for agent-agent collisions.

The optimal strategy is for each agent to claim one distinct landmark and hold it. Getting there requires the agents to implicitly divide the landmarks among themselves without any direct communication — just by observing each other's positions and learning from shared reward signals.

---

## The Algorithm: Independent PPO (IPPO)

IPPO treats each agent as if it were the only agent in the world. Each of the five agents maintains its own Actor and Critic networks and trains on its own experience. There is no shared policy, no centralised critic, and no explicit communication between agents.

**The Actor** for each agent outputs a categorical probability distribution over five discrete actions: no movement, up, down, left, right. During training it samples from this distribution for exploration. During evaluation it picks the highest-probability action greedily.

**The Critic** estimates the value of the agent's current state — how much cumulative reward it expects to receive from here. This value estimate is used to compute advantages via Generalised Advantage Estimation (GAE):

```
delta_t = reward_t + gamma * V(next_state) - V(current_state)

advantage_t = delta_t + (gamma * lambda) * advantage_(t+1)
```

GAE with lambda near 1 produces low-bias, higher-variance advantage estimates. Lambda near 0 produces high-bias, low-variance estimates that rely more on the critic. The default lambda of 0.95 balances these.

**The PPO update** clips the ratio between the new and old policy to prevent large destabilising updates:

```
ratio = new_policy(action | state) / old_policy(action | state)

actor_loss = -min(ratio * advantage, clip(ratio, 1-0.2, 1+0.2) * advantage)
```

This clipping is the key PPO innovation. It means the gradient is ignored whenever the policy update would move the ratio outside the (0.8, 1.2) range, keeping each step small and stable.

**Entropy regularisation** adds a bonus for policies that are not too certain. An entropy coefficient of 0.01 discourages the agent from committing to one action too early in training, maintaining exploration throughout.

**Why does independent learning work here?** Simple Spread has a shared reward, partial observability, and no adversarial agents. From each agent's perspective the environment is relatively stationary (other agents are slow to change their policies), which is the condition IPPO needs to converge. In adversarial or highly competitive environments IPPO breaks down, but for cooperative coverage tasks it works well.

---

## The Environment Details

The environment is a continuous 2D world with no walls.

**Observation per agent (30-dimensional):** Each agent sees its own position and velocity, the relative positions and velocities of all four other agents, and the positions of all five landmarks. This gives 30 numbers in total, all in a continuous range.

**Action space:** 5 discrete actions — no-op, move up, move down, move left, move right.

**Reward structure:**

```
team_penalty = -(number of landmarks not covered by any agent)
collision_penalty = -(number of agent-agent collisions) * local_ratio
```

The local_ratio of 0.5 means half the reward comes from the global coverage score and half from avoiding local collisions. This balance encourages both good global coordination and respectful local behaviour.

**Episode length:** 50 steps maximum per episode.

---

## Project Structure

```
swarm-architect-marl/
├── app.py                    main Gradio application
├── config.py                 EnvConfig, NetworkConfig, PPOConfig, TrainConfig
├── train.py                  standalone CLI training script
├── agents/
│   ├── networks.py           Actor and Critic network architectures
│   ├── buffer.py             rollout buffer with GAE computation
│   └── ippo.py               IPPOAgent, IPPOTrainer, training loop
├── environment/
│   └── wrapper.py            PettingZoo environment factory
├── evaluation/
│   └── metrics.py            TrainingLog and evaluate_agents utilities
└── visualization/
    └── animator.py           episode GIF export and training plots
```

---

## Quick Setup

```bash
git clone https://github.com/Dash10107/reinforcement-learning-lab.git
cd swarm-architect-marl
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

**Suggested starting point:** Go to the Watch the Swarm tab and click Watch the Agents. You will see an animated GIF of a pre-trained episode with five coloured agents trying to cover five landmarks. Then go to Train Your Swarm and click Start Quick Training for a 200-episode run. After training finishes, go to Mission Report and click Refresh to see the reward and delivery curves.

You can also run training from the command line for longer sessions:

```bash
python train.py --episodes 500
```

---

## What each tab shows

**The Mission:** An introduction to the task, how agents and landmarks work, and a step-by-step guide to using the app. Explains what RLHF is in plain terms without assuming prior knowledge.

**Watch the Swarm:** Runs the current best policy for one episode and shows an animated GIF replay. A score card below shows the coordination grade (A through F), team reward, and number of steps.

**Train Your Swarm:** One-click Quick Start training with recommended settings, or an expandable Advanced Settings section for configuring episodes, learning rate, discount factor, exploration decay, and rollout size. All slider labels use plain English descriptions rather than Greek letters.

**Mission Report:** After training, click Refresh to see three charts: team reward per episode, deliveries per episode, and collisions per episode. A formal evaluation section runs a configurable number of test episodes and shows a per-agent reward breakdown.

---

## Key hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| Learning rate (actor) | 3e-4 | How fast the policy updates |
| Learning rate (critic) | 1e-3 | Critic updates faster for stable value estimates |
| Gamma | 0.95 | Discount factor for future rewards |
| GAE lambda | 0.95 | Advantage estimation bias-variance tradeoff |
| Clip epsilon | 0.2 | Maximum allowed policy ratio change per update |
| Entropy coefficient | 0.01 | Encourages exploration throughout training |
| Rollout steps | 128 | Steps collected before each gradient update |
| Batch size | 64 | Mini-batch size during PPO update epochs |
| N epochs | 4 | Number of update passes per rollout batch |

---

## Requirements

```
gradio>=6.0.0
torch
numpy
matplotlib
mpe2
pettingzoo
```

---

## Things to Try

**1. Watch untrained agents before training anything.**
Go to Watch the Swarm before any training. Agents move randomly and cluster together. After Quick Training (200 episodes) watch again. Even a short training run produces visible coordination improvement.

**2. Run Quick Training three times and compare curves.**
Each run starts with a different random initialisation. Reward curves will look different but should all trend upward. IPPO consistently improves despite stochasticity.

**3. Find when coordination emerges in the reward curve.**
Look at the Mission Report reward curve carefully. There is usually a slow improvement phase then a sharper jump. That transition is when emergent landmark claiming begins — typically around episodes 50 to 150.

**4. Evaluate and check per-agent reward balance.**
Run 3 test episodes in Mission Report. If all five agents earn similar rewards they are all contributing equally. If one earns much less, that agent may be consistently failing to hold its landmark.

**5. Train with a high learning rate vs the default.**
In Advanced Settings set learning rate to 1e-3 (default is 3e-4). Higher LR trains faster but less stably. Compare the reward curves at episode 200. This shows how sensitive IPPO is to this single hyperparameter.

---

## Further Reading

- Schulman et al., Proximal Policy Optimization Algorithms (2017) — the original PPO paper
- Lowe et al., Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (2017) — MADDPG paper which uses the same Simple Spread task
- de Witt et al., Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? (2020) — empirical evidence for when IPPO works
- Schulman et al., High-Dimensional Continuous Control Using Generalised Advantage Estimation (2016) — the GAE paper
