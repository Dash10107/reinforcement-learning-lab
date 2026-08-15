---
title: Marl Warehouse Sim
emoji: 📦
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# MARL Warehouse Coordinator — Multi-Agent Delivery with Independent PPO

A multi-agent reinforcement learning simulation where a team of robots learns to coordinate package deliveries inside a custom warehouse grid. Each robot is trained independently using PPO and must navigate corridors, pick up packages from loading zones, and deliver them to drop-off zones — all without bumping into each other. You can watch trained robots work, compare them against random and greedy baselines, and train new agents from scratch.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Dash10107/marl-warehouse-sim)

---

## What problem does this solve?

A warehouse with multiple autonomous robots is a coordination problem. Each robot needs to move through the same corridors, claim different tasks, and avoid collisions — all without any central controller. The challenge is that what one robot does affects what the others should do, which makes the problem harder than single-agent navigation.

Traditional approaches use hand-crafted rules: "if another robot is ahead, wait." Reinforcement learning lets the robots discover coordination strategies on their own by experiencing thousands of episodes and learning from rewards and penalties.

---

## The Algorithm: Independent PPO (IPPO)

In multi-agent settings, one natural approach is to treat each agent as if it is the only agent in the world. Each robot maintains its own policy network and trains independently. This is called Independent Learning, and when combined with PPO it is called IPPO.

PPO (Proximal Policy Optimization) is an on-policy policy gradient algorithm. The key idea is that updating a policy too aggressively can cause it to collapse — the new policy might be so different from the one that collected the training data that the updates are meaningless or harmful. PPO prevents this by clipping the policy update ratio:

```
ratio = new_policy(action | state) / old_policy(action | state)

actor_loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
```

The clip operation ensures the ratio never strays too far from 1.0. If the ratio is clipped, the gradient contribution from that sample is zeroed out. This keeps each update small and stable.

**The Critic** estimates how good each state is. During training it computes advantages using Generalised Advantage Estimation (GAE), which balances between using only immediate rewards (high variance) and using the full predicted return (high bias):

```
advantage_t = delta_t + gamma * lambda * advantage_(t+1)

delta_t = reward_t + gamma * V(next_state) - V(current_state)
```

The Actor and Critic share no parameters between robots. Each robot learns its own policy based only on its own experience.

**Why IPPO works here:** The warehouse task has shared rewards (the team benefits when any robot makes a delivery) but local observations (each robot only sees its own position and nearby robots). This makes the problem relatively stationary from each robot's perspective, which is the condition IPPO needs to converge.

---

## The Warehouse Environment

The warehouse is a custom 12 by 12 grid built specifically for this project.

**Grid elements:**
- Open floor — robots can move here freely
- Shelf blocks — solid obstacles forming the warehouse structure
- Pickup zones (P) — robots walk here to collect a package
- Delivery zones (D) — robots walk here to complete a delivery

**State per robot (13-dimensional):**

```
[own_row, own_col,           normalised position of this robot
 goal_row, goal_col,         position of current target (pickup or delivery)
 has_package,                whether this robot is carrying a package
 rel_robot_1_row, rel_1_col, relative position of nearest robot
 rel_robot_2_row, rel_2_col,
 rel_robot_3_row, rel_3_col,
 rel_robot_4_row, rel_4_col] up to 4 neighbours, padded with zeros
```

**Actions:** 5 discrete — stay, up, down, left, right.

**Reward structure:**

| Event | Reward |
|---|---|
| Package picked up | +0.5 |
| Package delivered | +2.0 |
| Collision with another robot | -0.3 |
| Each step taken | -0.01 |

The step penalty encourages efficiency. The collision penalty discourages robots from blocking each other. When a robot completes a delivery, it is immediately assigned a new pickup task.

**Collision handling:** If two robots try to move into the same cell at the same step, both are blocked and both pay the collision penalty. This is a hard physical constraint rather than something the robots need to learn to avoid entirely — they need to learn to anticipate it.

---

## Three Strategies Compared

**IPPO (trained):** Each robot uses its learned policy based on its 13-dimensional observation. After enough training, robots spread across the warehouse and claim different tasks without clustering.

**Greedy:** Each robot always moves one step closer to its current goal using Manhattan distance. It has no awareness of other robots, so robots frequently block each other in narrow corridors.

**Random:** Each robot picks a random direction each step. Included as the lower bound baseline.

---

## Project Structure

```
marl-warehouse-sim/
├── app.py                  main Gradio application
├── warehouse/
│   ├── env.py              WarehouseEnv — custom multi-agent grid environment
│   ├── layout.py           warehouse grid definition and preset layouts
│   └── renderer.py         matplotlib frame rendering and GIF export
├── agents/
│   ├── networks.py         Actor and Critic network architectures
│   └── ippo.py             IPPO trainer, rollout collection, model save/load
├── viz/
│   └── charts.py           training curves and comparison charts
└── requirements.txt
```

---

## Quick Setup

```bash
git clone https://github.com/yourusername/reinforcement-learning-lab
cd marl-warehouse-sim
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

**To try it out:** Go to the Simulation tab, select Random as the strategy, and click Run Simulation. You will see the robots moving randomly through the warehouse as an animated GIF. Then switch to IPPO (trained) and run again — the difference in coordination should be visible. Go to the Training tab to train agents from scratch and watch the reward and delivery curves improve over episodes.

---

## What each tab shows

**Simulation:** Choose a strategy (IPPO, Random, or Greedy), set the number of robots and animation speed, and run a 100-step episode. The animated GIF shows robot positions, package-carrying status (white square), task route lines, and a step-delivery-collision HUD.

**Train Agents:** Set number of robots, episodes, and learning rate, then click Start Training. The training dashboard shows team reward per episode, deliveries per episode, and collisions per episode, all updated with a Refresh button. Training runs in the background.

**Benchmark:** Run all three strategies on the same warehouse with the same seed. A bar chart compares deliveries, collisions, and total reward side by side.

**How It Works:** Explains IPPO, the observation space, reward structure, and how to interpret the results.

---

## Key hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| Learning rate | 3e-4 | How fast each robot updates its policy |
| Gamma | 0.95 | Discount factor for future rewards |
| GAE lambda | 0.95 | Bias-variance tradeoff in advantage estimation |
| Clip epsilon | 0.2 | Maximum allowed policy change per update |
| Entropy coefficient | 0.01 | Encourages continued exploration |
| Rollout steps | 100 | Steps collected per agent before each update |

---

## Requirements

```
gradio>=6.0.0
numpy
torch
matplotlib
```

---

## Things to Try

**1. Compare Random vs IPPO delivery counts.**
Run Simulation with Random, note deliveries. Switch to IPPO (trained) and run again. The difference quantifies exactly what training contributes.

**2. Increase robot count to 6 and run both strategies.**
With more robots the coordination problem gets harder. Does the IPPO agent handle 6 robots as well as 4? This tests whether the policy generalises beyond its training configuration.

**3. Train from scratch and watch the collision curve.**
In the Training tab start training with 4 robots. Refresh the Mission Report periodically and watch collisions-per-episode trend downward as robots learn to anticipate each other.

**4. Benchmark Greedy vs IPPO on deliveries.**
Greedy robots move toward their goal but ignore others and frequently block each other in narrow corridors. Count how many more deliveries IPPO completes in the same 100 steps.

**5. Run IPPO trained on 4 robots with only 2 robots.**
Train with 4, then switch to 2 in the Simulation tab. The observation space includes 4 neighbour slots — with only 2 robots some are zero-padded. Does the policy still work or degrade?

---

## Further Reading

- Schulman et al., Proximal Policy Optimization Algorithms (2017) — the original PPO paper
- de Witt et al., Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? (2020) — empirical case for IPPO
- Sutton and Barto, Reinforcement Learning: An Introduction — Chapter 13 for policy gradient foundations
