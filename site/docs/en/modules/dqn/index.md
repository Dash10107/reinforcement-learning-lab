---
description: "Deep Q-Network (DQN) explained — how neural networks replace Q-tables for large state spaces. Covers experience replay, target networks, Double DQN, and the formal TD loss function."
---

# DQN Explained — Deep Q-Network Tutorial
<br> *When the maze gets too big.*

A 5×5 maze has 25 states. Your Q-table has 25 rows.
A 20×20 maze has 400 states. Still fine.

Now imagine the state isn't a grid position. Imagine the state is what the agent *sees* — a 84×84 pixel image of the game screen. That's 7,056 pixel values, each ranging 0–255. The number of possible unique images is astronomically large — far more states than you could ever write rows for.

You cannot build a Q-table for that. You need something else.

You need something that doesn't *memorise* every state — it *generalises* across them.

---

## Function approximation: the key insight

The Q-table stores one Q-value per (state, action) pair. It treats every state as completely separate from every other state. Cell (3,4) in a maze has no relationship to cell (3,5) — as far as the table is concerned, they might as well be different universes.

But they're not. They're neighbours. A good agent should know that what worked at (3,4) probably works at (3,5) too, with small differences.

We want the agent to **generalise**. To say: "I've never been to this exact state before, but it looks a lot like state I have been to — so I'll apply what I learned there."

That's exactly what a neural network does.

Instead of a table, we use a network: it takes a state as input and outputs Q-values for each action. When it's seen similar states before, it can make a reasonable prediction for new ones. It learns patterns, not just individual cases.

```
State (pixels / features)
        ↓
  Neural Network
        ↓
Q-values for each action
[up: 0.3, down: -0.1, left: 0.7, right: 0.2]
```

This is called **function approximation** — we're approximating the Q function (the mapping from states and actions to values) with a neural network instead of a table.

---

## DQN: the architecture

DQN stands for Deep Q-Network. It was introduced by DeepMind in 2013 and trained entirely from raw pixels to play Atari games better than humans. It's the algorithm that convinced the research world that deep RL was real.

The network itself is simple:

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),   # one Q-value per action
        )

    def forward(self, state):
        return self.net(state)           # returns Q-values for all actions
```

Input: the current state. Output: one Q-value for each possible action. Pick the action with the highest Q-value. That's it.

But training this network is harder than training the table. Two problems show up immediately.

---

## Problem 1: correlated experiences

In a Q-table, you update one cell. The rest of the table is unaffected.

With a neural network, every update changes the weights — which means every update slightly changes the Q-values for *every* state. If the agent just took five consecutive steps in one corner of the environment, training on all five in sequence makes the network overfit to that corner. It "forgets" what it learned about everywhere else.

The fix is called **experience replay**.

Instead of training on what just happened, the agent stores its experiences in a buffer:

```python
from collections import deque
import random

replay_buffer = deque(maxlen=10_000)   # circular buffer, oldest discarded

def store(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def sample_batch(batch_size=64):
    return random.sample(replay_buffer, batch_size)
```

At each training step, the agent samples a *random batch* from this buffer — experiences from all across the episode history, not just the most recent ones. This breaks the correlation. The network trains on a diverse mix of situations every step.

---

## Problem 2: a moving target

In supervised learning, your targets don't change. You're training to predict the label "cat" — and "cat" stays "cat" no matter how many times you update the weights.

In DQN, the target is:

```
Target = reward + γ × max Q(next_state)
```

But `max Q(next_state)` is computed by the *same network you're updating*. Every time you update the network, the target shifts. You're chasing a target that keeps moving away from you. This causes training to oscillate wildly or diverge entirely.

The fix is a **target network**: a second copy of the network whose weights are frozen for a while.

```python
# Two networks: online (trains every step) and target (updates slowly)
online_net = DQN(state_dim, n_actions)
target_net = DQN(state_dim, n_actions)
target_net.load_state_dict(online_net.state_dict())  # start identical

# Training: compute target using the FROZEN target network
with torch.no_grad():
    next_q = target_net(next_states).max(dim=1).values
    targets = rewards + gamma * next_q * (1 - dones)

# Update the ONLINE network toward these targets
loss = F.mse_loss(online_net(states).gather(1, actions), targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Every N steps: slowly sync the target network to the online network
if step % target_update_freq == 0:
    target_net.load_state_dict(online_net.state_dict())
```

The target network lags behind the online network. This gives training a stable target to aim at. Without it, DQN almost never converges.

---

## The loss function, formally

The DQN training objective is to minimise the **Temporal Difference (TD) loss**:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

Where:
- $\theta$ = parameters of the **online network** (what we update)
- $\theta^-$ = parameters of the **target network** (frozen copy, updated periodically)
- $\mathcal{B}$ = the replay buffer (we sample from it, not from the live environment)
- $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ = the **target** (a stable estimate of the true Q-value)
- $Q_\theta(s, a)$ = the **prediction** (what the online network thinks the Q-value is)

In plain English: *find the parameters that minimise the squared difference between what the network predicts and what the target network says the true value should be.*

---

## Double DQN: fixing a systematic flaw

Standard DQN has a bias problem. When computing the target:

$$\text{Target} = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

The same target network both *selects* the best action ($\arg\max_{a'}$) and *evaluates* how good that action is ($Q_{\theta^-}$). If the target network overestimates any action's Q-value (which it tends to), it'll select that action, making the overestimation self-reinforcing.

**Double DQN** decouples selection from evaluation:

$$\text{Target} = r + \gamma \, Q_{\theta^-}\!\left(s',\; \arg\max_{a'} Q_\theta(s', a')\right)$$

In plain English: *use the online network to decide which action is best, but use the target network to estimate how good that action is.* The two networks check each other, reducing overestimation.

```python
# Standard DQN target
next_q = target_net(next_states).max(dim=1).values

# Double DQN target: online selects, target evaluates
best_actions = online_net(next_states).argmax(dim=1)
next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
```

Double DQN was introduced in 2015, is a three-line change, and consistently outperforms standard DQN. Most modern DQN implementations use it by default.

---

## What you'll notice in the demo

Open the [Green Logistics Optimizer ↗](https://huggingface.co/spaces/Dash10107/green-logistics-optimizer) — a DQN agent routing delivery trucks through a city grid.

**Three things to watch:**

1. **Early training.** The agent drives randomly, mostly stuck or looping. The replay buffer is filling up, but hasn't seen enough varied experience to learn anything useful yet.

2. **The loss curve.** It doesn't start low — it often spikes upward before coming down. This is the network learning to make internally consistent predictions (the target network stabilising the training).

3. **Late training.** Routes become smoother and more efficient. The agent starts avoiding high-traffic zones and finding shortcuts. Generalisation at work: it's applying what it learned about one neighbourhood to others.

---

## Try it yourself

**Experiment 1 — Remove experience replay.**
Set `batch_size = 1` and sample only the most recent experience. Watch training become erratic — the loss oscillates rather than declining. This is why the replay buffer matters.

**Experiment 2 — Freeze the target network forever.**
Set `target_update_freq` to something very large (e.g., 1,000,000). The target network never updates. The online network trains toward a fixed target. This sounds stable — but the fixed target is wrong (it's from an untrained network). Learning plateaus early.

**Experiment 3 — Change the network size.**
Try a single hidden layer of 32 units. Then try 512 units. Notice the tradeoff: bigger networks can represent more complex policies, but they're slower to train and can overfit. The right size depends on how complex the environment actually is.

---

## The bigger picture

DQN was a breakthrough — not because the algorithm was complicated, but because it worked. Combining a neural network, experience replay, and a target network was enough to play 49 Atari games from raw pixels, with no game-specific engineering.

But DQN still has a fundamental limitation: it only works when you can enumerate all possible actions. At each state, it outputs one Q-value per action. If there are 4 actions (up/down/left/right), that's 4 outputs. If there are a million possible actions, you can't enumerate them.

And if actions are *continuous* — not a choice from a menu, but a number like "rotate the joint 37.4 degrees" — DQN doesn't apply at all.

The next part of the course introduces a completely different philosophy: instead of learning *values*, learn the *policy* directly.
