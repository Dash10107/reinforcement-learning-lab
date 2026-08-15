# Roadmap

What's been built, what's coming, and how you can contribute.

---

## What's complete

| Chapter | Status |
|---------|--------|
| MDP Framework | ✅ Done |
| What Is an Agent? | ✅ Done |
| The Problem With Guessing (Bandits) | ✅ Done |
| Smarter Exploration (UCB, Thompson) | ✅ Done |
| Q-Learning & SARSA | ✅ Done |
| Monte Carlo Methods | ✅ Done |
| Temporal Difference Learning | ✅ Done |
| Deep Q-Networks (DQN) | ✅ Done |
| Why Learn the Policy Directly? | ✅ Done |
| Actor-Critic (A2C) | ✅ Done |
| PPO | ✅ Done |
| SAC (continuous control) | ✅ Done |
| Multi-Agent RL (IPPO) | ✅ Done |
| Swarm Emergence | ✅ Done |
| Model-Based RL | ✅ Done |
| RLHF | ✅ Done |
| HMM Market Regimes (Bonus) | ✅ Done |
| Unity 3D Physics (Bonus) | ✅ Done |
| Core Concepts Glossary | ✅ Done |
| Build Your Own RL Agent | ✅ Done |
| Starter Templates (DQN, PPO) | ✅ Done |

---

## What's planned

**Algorithm chapters:**
- Dueling DQN — separating state value and action advantage
- Prioritised Experience Replay — not all experiences are equally useful
- Rainbow DQN — combining the six best DQN improvements
- DDPG — the predecessor to SAC for continuous control
- HER (Hindsight Experience Replay) — learning from failure
- Curiosity-driven exploration (ICM) — intrinsic motivation when reward is sparse

**Concept chapters:**
- Partial Observability (POMDPs) — when the agent can't see the full state
- Safe RL — constrained MDPs and staying within operating boundaries
- Offline RL — learning from a fixed dataset without environment interaction
- Distributional RL — learning the distribution of returns, not just the mean

**Resources:**
- SAC starter template
- Multi-agent environment setup guide
- "Reading an RL paper" guide — how to extract the algorithm from academic writing

---

## How to contribute

**Contribute a new chapter:**
Open an issue describing what topic you'd like to cover and how you'd explain it to a beginner. If the approach fits the course voice (empathy-first, math after intuition, no filler), we'll merge it.

**Fix something:**
Found an explanation that's unclear, a formula that's wrong, or a code snippet that doesn't run? Open a PR. These are the most valuable contributions.

**Add an experiment:**
Each chapter has three experiments. If you found a fourth — something that illustrates the concept in a surprising way — add it as a PR.

The only style rule for contributions: write for someone who is smart but doesn't know RL yet. Not for someone who already does.

[View on GitHub ↗](https://github.com/Dash10107/reinforcement-learning-lab)
