---
description: "What is reinforcement learning? Learn the five core components: agent, environment, state, action, and reward. Understand the explore-exploit tradeoff and how RL differs from supervised learning."
---

# What Is an Agent?

Imagine you wake up in a room you've never seen before.

No instructions. No map. Just a room with four doors. One of them leads outside. You don't know which one. You pick a door and walk through.

That's it. That's the core idea of reinforcement learning.

You're the agent. The room is the environment. Walking through a door is an action. Finding the exit is the reward. Everything else in RL — the equations, the algorithms, the neural networks — is just a more sophisticated version of this situation.

---

## The five things every RL problem has

Every single RL problem in existence — from teaching a robot to walk to ranking your social media feed — has these five components. They go by specific names, so let's name them now.

**Agent** — the thing that makes decisions. In our room: you.

**Environment** — everything the agent interacts with. In our room: the room, the doors, what's behind them.

**State** — a description of the current situation. In our room: which room you're in right now.

**Action** — something the agent can do. In our room: choosing a door.

**Reward** — feedback that tells the agent how well it's doing. In our room: +1 for finding the exit, 0 for everything else.

The agent observes the state, picks an action, receives a reward, and finds itself in a new state. Then it does it again. And again. This loop — observe, act, receive feedback, repeat — is called the **agent-environment loop**, and it's the heartbeat of every RL algorithm.

```
State ──→ Agent ──→ Action
  ↑                   ↓
  └─── Environment ───┘
           ↓
         Reward
```

---

## Why RL is different from other machine learning

In supervised learning, someone hands you a dataset. Every example in the dataset has a label — the right answer. Your job is to learn to predict that label. If you're wrong, you're told exactly how wrong you were, and you adjust.

RL doesn't give you labels. It gives you consequences.

Nobody tells the agent "you should have gone left there." The agent just knows that, three moves later, it hit a wall — or it found the exit. It has to figure out backwards which of its earlier decisions mattered.

This is called the **credit assignment problem**: when something good or bad happens, which of my past actions actually caused it?

It's the thing that makes RL hard. And it's the thing that makes it powerful — because it can learn to do things where we don't know the right answer in advance.

---

## Episodes

Most RL problems are broken into **episodes**: sequences with a clear start and end.

A maze has an episode. It starts when the agent enters the maze and ends when it either finds the exit or runs out of steps. The agent runs thousands of episodes, learning a little more each time.

Some problems don't have natural endings — think of a robot that just keeps moving through the world. Those are called **continuing tasks**. They're handled slightly differently (we'll see how when we get to the relevant algorithms), but the core loop is identical.

---

## What the agent is actually learning

The agent isn't just trying to win this episode. It's trying to build a **policy** — a map from situations to actions.

A **deterministic policy** maps each state directly to an action:

$$\pi(s) = a \quad \text{(always take this exact action in this state)}$$

A **stochastic policy** maps each state to a probability distribution over actions:

$$\pi(a \mid s) = P(\text{take action } a \text{ in state } s)$$

In plain English: a deterministic policy says "in this situation, always go left." A stochastic policy says "in this situation, go left 70% of the time, go right 30% of the time."

Why would you ever want a stochastic policy? Two reasons. First, in some games (like Rock Paper Scissors), being unpredictable is part of the optimal strategy — a purely deterministic policy can always be beaten by an opponent who figures it out. Second, during training, stochastic policies naturally explore: the randomness means the agent tries different actions in the same state, which is how it discovers what works.

At the start of training, the policy is random. The agent wanders around pressing doors with no strategy. Slowly, through thousands of episodes, it starts to notice patterns. The policy gets better.

A perfect policy would tell the agent exactly what to do in every possible situation. We never quite get there. But we get close enough to be useful.

---

## The reward signal is everything — and nothing

The reward tells the agent what to care about. Get this right and you get a brilliant agent. Get it wrong and you get an agent that finds impressive ways to fail.

A famous example: a researcher gave a boat-racing game agent a reward for collecting coins around a track. The agent discovered it could spin in circles collecting the same coins over and over without finishing the race. Technically it was maximising reward. Not what anyone wanted.

This is called **reward hacking**, and it's one of the reasons RL is both fascinating and dangerous. The agent will do exactly what you reward it to do. Nothing more, nothing less.

We'll see this again in [the RLHF chapter](../rlhf/) — the technique that tries to solve this by letting humans teach the reward function directly.

---

## Try it yourself

**Experiment 1 — Change the reward signal.**
Pick any simple environment (CartPole, a grid maze). Change the reward from +1 for success to +1 for every step survived. Does the agent behave differently? What does it optimise for now?

**Experiment 2 — Sparse vs dense reward.**
Give the agent reward only at the very end (sparse). Then add small rewards for getting closer to the goal each step (dense). Notice how the dense reward makes training faster but introduces risk: the agent might learn to circle near a large intermediate reward instead of finishing.

**Experiment 3 — Longer episodes.**
Double the episode length limit. Does the agent take longer to learn? Why? (Hint: the credit assignment problem gets harder when the gap between an action and its consequence grows.)

---

## Next

Now that you know what an agent is and what it's doing, the next chapter introduces the formal language that all RL algorithms share.
