# Reinforcement Learning — Core Concepts

This document is the starting point if you are new to reinforcement learning. Read this before diving into any of the 12 projects. It explains what RL is, how it works, and how all the algorithms in this portfolio relate to each other.

---

## What is Reinforcement Learning?

Reinforcement learning is a way of teaching a computer to make decisions by letting it try things and learn from the results.

You do not show it correct answers (that is supervised learning). You do not just find patterns in data (that is unsupervised learning). Instead, you put it in a situation, let it act, and tell it whether things went well or badly. Over thousands of repetitions it figures out which actions tend to lead to good outcomes.

The classic analogy is training a dog. You do not explain to a dog in words that sitting earns a treat. You wait for it to sit, give it a treat, and it gradually learns that sitting is good. The dog is the agent, the treat is the reward, and the room it is in is the environment.

---

## The Four Core Concepts

Every single RL project, from the simplest maze solver to a multi-agent warehouse system, is built from the same four ideas.

**Agent** — the thing that makes decisions. In this portfolio: a robot, a drone, a battery controller, an ad system, a rocket.

**Environment** — everything the agent interacts with. The maze, the grid, the electricity market, the city. The environment responds to the agent's actions and produces the next situation.

**State** — a description of the current situation. What the agent can observe right now. The maze solver's state is just its current position. The rocket lander's state is position, velocity, angle, and whether its legs are touching down.

**Reward** — a number the environment gives after each action. Positive reward means the action was good. Negative reward (penalty) means it was bad. The agent's only goal is to collect as much reward as possible over time.

The loop is always the same:

```
Agent observes state
      ↓
Agent chooses action
      ↓
Environment produces next state + reward
      ↓
Agent learns from this experience
      ↓
Repeat
```

---

## Key Vocabulary

**Policy** — the agent's decision-making strategy. Given a state, which action does it choose? A policy can be a lookup table (tabular), a formula, or a neural network.

**Episode** — one complete run from start to finish. For a maze, one episode is one attempt to reach the exit. For a rocket, one episode is one landing attempt.

**Return** — the total reward collected across an episode. The agent wants to maximise the return.

**Discount factor (gamma)** — a number between 0 and 1 that makes the agent value immediate rewards more than distant ones. A reward received in 10 steps is worth gamma^10 of a reward received right now. This is why you see gamma = 0.99 (cares a lot about the future) or gamma = 0.9 (cares less) in the project configs.

**Exploration vs Exploitation** — the fundamental tension. The agent can exploit what it already knows works, or it can explore to find something even better. Explore too much and you waste time on bad actions. Exploit too early and you miss the best strategy. Every algorithm in this portfolio has a different answer to this tradeoff.

**Q-value** — Q(state, action) is the expected total reward if you take that action from that state and then follow the best possible policy afterward. Once you have good Q-values, making decisions is easy: always pick the action with the highest Q-value.

**Value function** — V(state) is the expected total reward from a state if you follow the best possible policy from there. Related to Q-values but does not depend on a specific action.

**Advantage** — A(state, action) = Q(state, action) - V(state). How much better is this specific action compared to what you would normally expect from this state? Positive advantage means this action is better than average. Negative means it is worse. This is the central idea behind A2C and PPO.

---

## The Algorithm Family Tree

All the algorithms in this portfolio fall into a few families. Understanding the tree makes it much easier to see why each project uses the algorithm it does.

```
Reinforcement Learning
│
├── Tabular Methods (small, discrete problems)
│   ├── Q-Learning ................. RL Maze Solver
│   ├── SARSA ...................... RL Maze Solver
│   └── Monte Carlo ................ RL Maze Solver
│
├── Value-Based Deep RL (discrete actions, neural network)
│   └── DQN ........................ Green Logistics, Smart Grid
│
├── Policy Gradient (direct policy optimisation)
│   ├── A2C ........................ AI Tutor
│   ├── PPO ........................ Swarm Architect, Warehouse, Huggy
│   └── SAC ........................ Rocket Lander (continuous actions)
│
├── Model-Based RL (learn the world first, then plan)
│   └── Ensemble Dynamics + MPC .... MBRL Pendulum
│
├── Multi-Armed Bandits (no state transitions)
│   ├── Epsilon-Greedy
│   ├── UCB1
│   ├── Thompson Sampling .......... MAB Banner Optimizer
│   ├── Gradient Bandit
│   └── EXP3
│
├── Multi-Agent RL (multiple agents interacting)
│   ├── IPPO ....................... Swarm Architect, Warehouse
│   └── (MADDPG, QMIX — not in this portfolio)
│
├── RLHF (learning from human preferences)
│   └── Bradley-Terry Reward Model . Digital Calligrapher
│
└── Statistical Inference (hidden state estimation)
    └── Gaussian HMM ............... Market Regime Detector
```

---

## Value-Based vs Policy Gradient

This is the most important distinction to understand early.

**Value-based methods** (Q-Learning, DQN) learn Q-values — a score for every (state, action) pair. Once learned, the policy is implicit: just always pick the highest Q-value action. These work very well when there are a small or moderate number of possible actions.

**Policy gradient methods** (A2C, PPO, SAC) learn the policy directly — a neural network that maps state to action probabilities or to a continuous action value. These work when actions are continuous (like engine throttle from 0 to 1) or when the number of possible actions is huge.

The rocket lander uses SAC (policy gradient) because the engine throttle is a continuous number. The maze solver uses Q-Learning (value-based) because there are only four possible moves.

---

## On-Policy vs Off-Policy

**On-policy** methods learn from experience collected by the current policy. SARSA and PPO are on-policy. Each time you update the policy, the old experience becomes stale and you must collect new data.

**Off-policy** methods learn from any experience, regardless of which policy collected it. Q-Learning, DQN, and SAC are off-policy. They store experience in a replay buffer and learn from it repeatedly. This makes them more sample efficient but more complex.

---

## Model-Free vs Model-Based

**Model-free** methods (everything except MBRL Pendulum) learn a policy or Q-function by interacting directly with the environment. They do not try to understand how the environment works internally.

**Model-based** methods first learn a model of the environment — a neural network that predicts what happens when you take an action. Then they use that model to plan future actions without needing to interact with the real environment. This can be dramatically more sample efficient but is harder to implement correctly because an imperfect model leads to bad plans.

---

## Multi-Armed Bandits

Bandits are a simplified version of RL where there are no state transitions. You just have a set of actions (the arms) and each one gives a random reward with some fixed probability. The only decision is which arm to pull next.

This sounds too simple to be useful but the exploration-exploitation problem is exactly the same as in full RL, and bandits appear everywhere: which ad to show, which drug to test in a clinical trial, which search result to rank first. Understanding bandits well makes full RL much easier to follow.

---

## How the Projects Connect

Reading the projects in this order will give you the clearest progression:

| Stage | Project | What it teaches |
|---|---|---|
| 1 — Basics | RL Maze Solver | The core RL loop, Q-Learning, tabular methods |
| 2 — Bandits | MAB Banner Optimizer | Exploration vs exploitation in isolation |
| 3 — Policy gradient | AI Tutor A2C | Actor-critic, advantage estimation |
| 4 — Continuous control | Rocket Lander SAC | Maximum entropy RL, continuous actions |
| 5 — Real-world DQN | Green Logistics | DQN applied to a spatial planning problem |
| 6 — Economics | Smart Grid | DQN vs optimal (DP) in a practical domain |
| 7 — Multi-agent | Swarm Architect | Independent learning, cooperative tasks |
| 8 — Multi-agent 2 | MARL Warehouse | Task-based coordination, richer rewards |
| 9 — Model-based | MBRL Pendulum | World models, planning, compounding errors |
| 10 — Statistics | Market Regime | State inference, hidden Markov models |
| 11 — Human feedback | Digital Calligrapher | RLHF, reward learning from comparisons |
| 12 — 3D control | Unity Huggy Demo | Physics simulation, joint torques, sparse reward |

---

## Quick Reference: Reward Design

The reward function is the most important design decision in any RL project. Here is how each project handles it and what tradeoffs that creates:

| Project | Reward | Tradeoff |
|---|---|---|
| Maze Solver | -1/step, -5 wall, +100 goal | Sparse goal reward; step penalty drives efficiency |
| Rocket Lander | Dense shaped (position, velocity, legs) | Rich signal but hard to tune |
| Green Logistics | -carbon/step, +20 delivery | Carbon cost teaches congestion avoidance |
| Smart Grid | Net revenue per hour | Economic signal, no manual shaping needed |
| AI Tutor | Current proficiency of studied subject | Directly measures learning progress |
| Warehouse | +2 delivery, -0.3 collision, -0.01/step | Multi-component reward drives multiple behaviours |
| Swarm Architect | -uncovered landmarks, -collisions | Team reward with local penalty component |
| MBRL Pendulum | -(theta² + 0.1ω² + 0.001τ²) | Dense analytic reward, differentiable |
| MAB Banner | CTR × revenue per impression | Direct business metric |
| Digital Calligrapher | Learned from human votes | No predefined formula — that is the point |
| Market Regime | No RL reward — statistical likelihood | HMM is not trained by reward |
| Unity Huggy | Distance to stick | Sparse, intentionally minimal |
