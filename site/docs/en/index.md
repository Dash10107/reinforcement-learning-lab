---
description: "Learn Reinforcement Learning from scratch — free, open-source course covering Q-Learning, DQN, PPO, SAC, Multi-Agent RL, and RLHF with real math and runnable code. Built for beginners and intermediates."
---

# Reinforcement Learning Lab

**The RL course that respects your intelligence and doesn't waste your time.**

Every concept starts with a situation you've already felt. Every formula gets one sentence of English before it and one after. Every chapter is self-contained. No prerequisites beyond Python.

<div class="hero-actions">
  <a href="./modules/mdp/" class="hero-btn-primary">Start Learning →</a>
  <a href="./resources/getting-started/" class="hero-btn-secondary">Build Your Own Agent</a>
</div>

---

## Why this course exists

Most RL tutorials fall into one of two traps.

The first: pure intuition. Toy examples, vague analogies, no math. You finish and feel like you understood something — but you can't read a paper, can't implement anything from scratch, can't explain why an algorithm works.

The second: pure theory. Bellman equations on slide 2, convergence proofs before you know what a policy is. You close the tab feeling stupid.

This course does neither.

We explain the *feeling* of a problem before introducing the formula that solves it. We show the math in full — but only after you already understand what it's measuring. We write code that's short enough to read but complete enough to actually run.

---

## What you'll be able to do

By the end of this course, you'll be able to:

- **Explain every major RL algorithm** — from Q-Learning to PPO to RLHF — in plain English *and* in the math
- **Implement DQN and PPO from scratch** using the starter templates, adapted to your own environment
- **Turn any problem into an RL environment** using the 5-step framework in the resources section
- **Read research papers** — you'll know the vocabulary, the standard formulations, the common tricks
- **Debug when training doesn't work** — the five failure modes and how to diagnose each one

---

## The curriculum

```
Before you begin
  The MDP Framework            The formal language behind all of RL

Part 1 — Foundations
  What Is an Agent?            The 5 building blocks: agent, env, state, action, reward
  The Explore-Exploit Problem  Why you can't just be greedy from the start
  Smarter Ways to Explore      UCB and Thompson Sampling — exploration guided by uncertainty

Part 2 — Memory
  Q-Learning & SARSA           Building a memory of what works where
  Monte Carlo Methods          Learning from the full episode, not each step
  Temporal Difference (TD)     The unified framework — TD(0), n-step, TD(λ)

Part 3 — Scale
  Deep Q-Networks (DQN)        When the world is too big for a table

Part 4 — Intention
  Policy Gradient Math         The theorem — why "do more of what worked" actually works
  Actor-Critic (A2C)           Two networks, one learning from the other
  PPO                          The algorithm behind game AI, robotics, and ChatGPT
  SAC                          Continuous control with maximum entropy exploration

Part 5 — Together
  Multi-Agent RL (IPPO)        What changes when multiple agents are learning at once
  Swarm Emergence              Complex group behaviour from simple individual rules

Part 6 — Imagination
  Model-Based RL               Learning a world model, planning in your head

Part 7 — Alignment
  RLHF                         Teaching AI what you want through preference comparisons
  DPO                          The modern alternative that skips the reward model entirely

Resources
  Core Concepts Glossary       Every term, plain English, one sentence each
  Build Your Own Agent         5-step guide from problem → working RL agent
  Starter Templates            DQN and PPO implementations, fully annotated
```

---

## How to use this course

**If you're a complete beginner:** start at [The MDP Framework](./modules/mdp/) and read in order. Each chapter assumes you've read the previous one, but nothing else.

**If you already know the basics:** jump to any chapter. Each one is as self-contained as possible. The sidebar has everything indexed.

**If you want to build something:** go straight to [Build Your Own Agent](./resources/getting-started/). It's the most practical page in the course — it walks you from "I have a problem" to "I have a working RL agent" in five steps.

**If you're stuck on a term:** the [Core Concepts Glossary](./resources/concepts/) has every technical term from the course, one plain-English sentence each.

---

## The writing rules

Everything in this course was written with four rules:

1. **Start with a situation, not a definition.** You feel the problem before you name it.
2. **One English sentence before every formula. One after.** Never show math without context.
3. **No filler.** If a sentence doesn't earn its place, it's cut.
4. **Projects are examples, not requirements.** This course stands alone — you don't need any specific app or repository to follow it.

[Begin with the MDP Framework →](./modules/mdp/)
