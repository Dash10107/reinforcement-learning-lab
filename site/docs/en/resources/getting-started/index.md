---
description: "How to build your own reinforcement learning agent from scratch. 5-step guide: define state, action, reward, choose algorithm, implement training loop, and debug common failures."
---

# How to Build Your Own RL Agent

Reading about RL is one thing. Applying it to a new problem is another.

This guide walks you through the exact decisions you need to make — in order — to turn any problem into a working RL agent. It's the page to come back to when you have an idea and want to know where to start.

---

## Step 1: Is RL actually the right tool?

Before anything else, ask yourself these questions:

**Do you have an environment you can reset?**
RL learns by running episodes. If you can't reset your problem to a starting state and run it thousands of times, RL will be very slow to train.

**Do you have a notion of reward?**
You need some signal — even a weak or noisy one — that tells you when things are going well. If you can't define any reward at all, RL won't work.

**Is the right answer unknown in advance?**
If you already have labelled examples of correct behaviour, supervised learning is almost always faster and more data-efficient. RL is for problems where you don't know the right answer — only whether the outcome was good or bad.

**Can you afford many attempts?**
Training a basic RL agent takes thousands to millions of episodes. If each episode is expensive (a physical robot, a real-world financial trade, a medical intervention), RL may be impractical without a simulator.

If you answered yes to most of these, RL is appropriate. Continue.

---

## Step 2: Define your four components

This is the most important step. Get these wrong and no algorithm will save you.

### State: what does the agent observe?

Write down exactly what information the agent needs to make good decisions. Then remove everything that isn't necessary.

**Good states:**
- Robot: joint angles, velocities, distance to goal, foot contact sensors
- Trading: price history (last N candles), current position, available cash
- Game: raw pixels or a minimal feature vector (player position, enemy positions, health)

**Common mistakes:**
- Too much information: the agent has to learn to ignore irrelevant features. This is wasteful.
- Too little information: the state doesn't satisfy the Markov property (the right decision depends on things you're not giving the agent). Training will be slow or impossible.

Ask yourself: "Given only this state, could a smart human make a good decision?" If yes, the state is probably sufficient.

### Action: what can the agent do?

**Discrete or continuous?**
- Discrete (choose from a finite list): use DQN or PPO. Examples: turn left/right/straight, choose ad A/B/C/D, pick which item to produce.
- Continuous (output real numbers): use SAC or PPO with Gaussian policy. Examples: set motor torque, choose bid amount, set thermostat temperature.

**How many dimensions?**
Each continuous dimension adds complexity. A robot arm with 7 joints has a 7-dimensional continuous action space. This is manageable. A 100-dimensional action space is hard. Think about whether you can discretise some dimensions.

### Reward: what do you want to maximise?

This is where most RL projects fail. The reward signal is the only thing the agent optimises for — it will find the most direct path to maximising it, which is often not what you meant.

**Start simple.** The simplest reward that captures the goal is usually best.

```
Bad:  +1 for every second the pole stays upright; 
       -10 for falling; +0.1 for small control effort; +0.05 for being near centre
       → Too many terms. The agent will find ways to trade them off you didn't intend.

Good: +1 for every second the pole stays upright; 0 for falling.
       → One clear signal. Simple to interpret.
```

**Watch for reward hacking.** Imagine a agent that's trying to maximise your reward in the most adversarial way possible. Would it do something you didn't want? If yes, fix the reward.

**Common reward patterns:**
- Dense reward: reward at every step (easier to learn, but requires careful design)
- Sparse reward: reward only at success (harder to learn, but less risk of reward hacking)
- Shaped reward: add intermediate rewards that guide toward the goal (useful but adds design complexity)

### Episode: when does it end?

Define success and failure clearly. An episode ends when:
- The agent achieves the goal (success)
- The agent fails in a way that makes continuing pointless (fell over, crashed, lost the game)
- A time limit is reached (truncation)

Make sure your episodes are long enough for the agent to experience consequences of its actions, but not so long that the training is dominated by padding.

---

## Step 3: Pick the algorithm

Use this decision tree:

```
What kind of action space do you have?
│
├── Discrete (finite list of actions)
│   │
│   ├── Small state space (< a few thousand states)
│   │   └── Q-Learning (tabular) — fast, interpretable
│   │
│   └── Large or visual state space
│       └── DQN — neural Q-function approximation
│
└── Continuous (real-valued actions)
    │
    ├── Single agent
    │   ├── Off-policy OK (sample efficient) → SAC (recommended)
    │   └── Simpler/more stable → PPO with Gaussian head
    │
    └── Multiple agents
        └── IPPO — independent PPO for each agent
```

**When in doubt, start with PPO.** It's robust, well-understood, and works across a wide range of problems. SAC is more sample-efficient but requires more tuning. Q-Learning is only practical for small discrete state spaces.

**If you have very few real interactions** (physical robot, expensive simulation): add a world model (MBRL). Use real interactions to train the model, then plan using the model.

---

## Step 4: The training loop

Every RL training loop has the same basic structure. Here it is as a 40-line template you can adapt to any problem:

```python
import gymnasium as gym
import torch

# 1. Create environment and agent
env   = gym.make("YourEnv-v0")
agent = YourAgent(state_dim=env.observation_space.shape[0],
                  action_dim=env.action_space.n)

# 2. Training loop
for episode in range(10_000):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # 3. Agent observes state and chooses action
        action = agent.select_action(state)

        # 4. Environment responds
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 5. Agent stores the experience
        agent.store(state, action, reward, next_state, done)

        # 6. Agent learns from stored experience
        if agent.ready_to_train():
            loss = agent.update()

        state = next_state
        total_reward += reward

    # 7. Log progress
    if episode % 100 == 0:
        print(f"Episode {episode} | Reward: {total_reward:.1f}")

env.close()
```

Every RL algorithm — DQN, PPO, SAC — fits into this template. The differences are inside `select_action()`, `store()`, and `update()`.

---

## Step 5: When it doesn't work

Training doesn't improve? Go through this checklist in order.

**1. Reward is too sparse.**
The agent almost never gets a non-zero reward. It has nothing to learn from. Fix: add shaped rewards or use curriculum learning (start with an easier version of the task).

**2. State is wrong.**
Missing critical information → agent can't make good decisions. Extra irrelevant information → agent needs more data to learn to ignore it.
Fix: think carefully about what a human would need to know to make a good decision.

**3. Learning rate is too high.**
The loss oscillates wildly or explodes. Fix: reduce by 10× and try again.

**4. Episode length is wrong.**
Too short: the agent never experiences the consequences of its actions. Too long: reward is so diluted by time that the signal is weak. Fix: aim for episodes where the agent has enough time to both cause and observe outcomes.

**5. Reward function is being hacked.**
Training looks great, but the agent is doing something unexpected. Fix: watch the agent and ask "what is it actually doing to get this reward?" Then patch the reward or add constraints.

---

## Environments worth trying first

These environments have been used in RL research for decades. They're well-understood, fast to simulate, and cover a range of problem types.

**Gymnasium (formerly OpenAI Gym):**
- `CartPole-v1` — classic balancing task, discrete actions, great for testing any algorithm
- `MountainCar-v0` — sparse reward, good for testing exploration strategies
- `LunarLander-v2` — continuous control, realistic physics
- `Pendulum-v1` — continuous control, small state space, good for SAC/MBRL

**Multi-agent environments:**
- `PettingZoo` — a large collection of multi-agent environments (cooperative and competitive)
- `Gymnasium with multiple agents` — parallel Gymnasium instances with IPPO

**Custom environments:**
Use `gymnasium.Env` as your base class:

```python
import gymnasium as gym
import numpy as np

class MyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, seed=None):
        self.state = np.zeros(4)
        return self.state, {}

    def step(self, action):
        # Your environment logic here
        next_state = self.state  # update based on action
        reward = 0.0             # compute reward
        terminated = False       # True if success/failure
        truncated = False        # True if time limit reached
        return next_state, reward, terminated, truncated, {}
```

This is the minimum interface any Gymnasium environment needs. Everything else in this course — every algorithm, every training loop — will work with any environment that implements this interface.

[← Back to Resources](../)
