---
description: "Learn the Markov Decision Process (MDP) — the formal framework behind all reinforcement learning. Covers states, actions, rewards, policies, value functions, and Bellman equations with plain English explanations."
---

# Markov Decision Processes (MDP) Explained — RL Framework
<br> *The math behind the decisions.*

Every reinforcement learning algorithm you'll ever encounter is solving the same underlying problem.

The problem has a name: the **Markov Decision Process** (MDP). It's not complicated — it's just a precise way to describe the situation an agent is in. Once you know it, you'll see it everywhere, and every new algorithm will feel like a familiar shape.

---

## Why we need a framework

Right now you have an intuition for what RL is: an agent, a world, some actions, some rewards. That's correct.

But intuition alone can't tell you whether two algorithms are solving the same problem differently, or fundamentally different problems. It can't tell you what guarantees an algorithm comes with. It can't help you read a research paper.

A framework gives us shared language. And the MDP framework is the language of RL.

---

## The five components

An MDP is defined by five things, written as a tuple: **(S, A, P, R, γ)**.

**S — the state space**
The set of all possible states the environment can be in. In a maze, this is the set of all grid cells. In an Atari game, it's the set of all possible screen images. In robotics, it's all possible combinations of joint angles and velocities.

**A — the action space**
The set of all possible actions the agent can take. Move up/down/left/right. Apply 12.7% thrust. Choose ad variant 3.

**P — the transition model**
The probability of moving from state *s* to state *s'* when you take action *a*.

$$P(s' | s, a) = \text{probability of landing in } s' \text{ given you were in } s \text{ and took } a$$

In most real problems, we don't know P. The agent has to figure out the consequences of its actions by trying them. This is why most RL algorithms are "model-free" — they don't assume they know the transition model.

**R — the reward function**
The immediate reward received after taking action *a* in state *s*.

$$R(s, a) = \text{expected immediate reward for taking action } a \text{ in state } s$$

**γ (gamma) — the discount factor**
A number between 0 and 1 that controls how much the agent cares about future rewards vs immediate rewards.

$$\gamma \in [0, 1]$$

At γ = 1, all future rewards matter equally. At γ = 0, only the immediate reward matters. In practice, γ = 0.99 is a common choice: future rewards matter a lot, but not quite as much as present ones.

Why discount at all? Two reasons. First, it's mathematically convenient — it guarantees the total reward sum converges (doesn't grow to infinity). Second, it's intuitively reasonable — a reward now is more certain than a reward in 50 steps.

---

## The Markov Property

The "Markov" in Markov Decision Process refers to a key assumption: **the future only depends on the present, not the past.**

$$P(s_{t+1} | s_0, a_0, s_1, a_1, ..., s_t, a_t) = P(s_{t+1} | s_t, a_t)$$

In plain English: to decide what will happen next, you only need to know where you are and what you're doing now. The full history of how you got here doesn't add any information.

Is this always true in reality? Not always. But it's a useful simplification. In practice, when the Markov property doesn't hold, we encode relevant history into the state itself — so that the state *does* summarise everything relevant from the past.

---

## The goal: find the optimal policy

A **policy** (π) is a rule for choosing actions. It maps states to actions, or to probability distributions over actions:

$$\pi(a | s) = \text{probability of taking action } a \text{ in state } s$$

The goal of RL is to find the **optimal policy** π* — the policy that maximises the expected total reward over time.

That total reward, starting from state *s*, is called the **return**:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

Where each reward in the future gets discounted by one additional factor of γ.

---

## Value functions: the two most important quantities in RL

Almost every RL algorithm is trying to estimate one of these two things.

**The State Value Function V^π(s):**
*How good is it to be in state s, if I'm following policy π?*

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\middle|\; s_t = s\right]$$

The expected total discounted reward, starting from state *s*, acting according to π from then on.

**The Action-Value Function Q^π(s,a):**
*How good is it to take action a in state s, and then follow policy π?*

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\middle|\; s_t = s, \; a_t = a\right]$$

The difference: V^π(s) averages over all actions the policy might take. Q^π(s,a) fixes the first action to *a*, then follows π afterward.

The relationship between them:

$$V^\pi(s) = \sum_a \pi(a|s) \cdot Q^\pi(s,a)$$

The value of a state is the weighted average of Q-values across all actions, weighted by how likely the policy is to take each one.

---

## The Bellman equations

The two value functions are connected to themselves through a beautiful recursive relationship, called the **Bellman equations**.

**Bellman Expectation Equation** (for any policy π):

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a) + \gamma V^\pi(s')\right]$$

In plain English: *the value of a state is the immediate reward I expect to get, plus the discounted value of where I end up.*

This equation says: "the value right now equals reward now plus value later." It's recursive — the value of a state depends on the value of the next state. This recursion is what the Bellman equation captures.

**Bellman Optimality Equation** (for the optimal policy π*):

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a) + \gamma V^*(s')\right]$$

In plain English: *the optimal value of a state is the reward from the best action, plus the discounted optimal value of where that action takes you.*

This is the equation every Q-Learning variant is trying to solve. The Q-table update rule from Chapter 3 is literally an iterative approximation of this equation.

---

## What you now have

You have the formal skeleton that every RL algorithm fits into:

- **Value-based methods** (Q-Learning, DQN): estimate Q*(s,a) directly, derive the policy by taking the argmax.
- **Policy-based methods** (REINFORCE, PPO): directly learn π(a|s) without going through Q-values.
- **Actor-Critic methods** (A2C, SAC): learn both π and V^π, use V to guide the update of π.
- **Model-based methods** (MBRL): learn P(s'|s,a) and use it for planning.

Every algorithm from here on is one of these four types. You'll recognise them because you now know what they're each trying to do.
