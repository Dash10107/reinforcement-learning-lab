# When the World Has Three Dimensions and Gravity

Everything we've built so far has lived in relatively clean spaces.

Grid worlds. Continuous pendulums. 2D rocket physics. These are real environments with real complexity — but they're still simplified versions of the physical world.

Now the world has three dimensions. The agent has a body with multiple joints, each controlled by separate motors. The body wants to fall over. The ground is uneven. Gravity is constant and unforgiving.

Welcome to 3D locomotion — and to what it feels like when everything comes together at once.

---

## Why walking is harder than it looks

You've been walking since you were one year old. It feels effortless. But your nervous system is running an extraordinarily complex control loop: monitoring hundreds of sensors in your muscles and joints, predicting ground conditions from visual input, making thousands of micro-adjustments per second to keep your centre of mass above your feet.

Teaching a simulated dog to walk from scratch requires learning all of that — not from evolution, but from a reward signal and a gradient update.

The action space is high-dimensional and continuous: multiple joints, each outputting a target angle, each step. The reward is sparse initially — you only get reward for moving forward, and early random policies mostly just fall over. The dynamics are complex enough that hand-crafted solutions don't work.

This is what makes it a perfect test of how far we've come.

---

## What the agent is controlling

The Unity ML-Agents environment provides a simulated dog ("Huggy") with:

- **Multiple limb segments** connected by joints
- **Physics simulation** with realistic gravity and friction
- **Proprioceptive observations** — joint angles, velocities, centre-of-mass position, orientation
- **A target** — a moving ball that the dog is rewarded for reaching

The agent outputs joint torques. At each step, it observes all the sensor readings and outputs a number for each joint. The physics engine applies those torques, updates the simulation, and returns the new state.

PPO trains the policy end-to-end: raw sensor readings → joint commands, with the reward signal guiding it toward smooth, efficient locomotion.

---

## Curriculum learning: starting simple

You can't start a dog-walking agent on a moving target in difficult terrain. The reward is too sparse — the agent almost never reaches the ball, so it gets almost no signal about what to do.

The solution is **curriculum learning**: start with an easier version of the task, let the agent get good at it, then gradually make it harder.

```
Stage 1: Ball is stationary, 1 metre away. Reward: reach it.
         → Agent learns to walk in a straight line.

Stage 2: Ball moves slowly. Reward: track it.
         → Agent learns to change direction.

Stage 3: Ball moves quickly, terrain is varied.
         → Agent refines gait for general locomotion.
```

Each stage uses the policy trained in the previous stage as its starting point. The agent doesn't re-learn from scratch — it builds on what it knows.

Curriculum learning is now standard in robotics. It's how Boston Dynamics' systems are trained. It's how AlphaGo was trained (starting against easy opponents, then stronger ones). The key insight: don't give the agent the full problem before it has the skills to make progress on it.

---

## The connection to everything before

This chapter is the last in the course because it requires almost everything that came before it:

- **PPO** (Chapter 7): the algorithm training the dog, with its conservative policy updates.
- **Continuous action spaces** (Chapter 8): joints don't pick from a list of angles, they output real numbers.
- **Curriculum learning**: a practical trick that makes sparse reward environments tractable.
- **The reward signal**: who designed it? A human, carefully, thinking about what smooth locomotion looks like. A different reward would produce a different gait — or no gait at all.

---

## What you'll notice in the demo

Open the [Huggy 3D Dog Walker ↗](https://huggingface.co/spaces/Dash10107/Unity-RL-Huggy-Demo) — a PPO agent trained to walk and fetch in 3D.

**Three things to watch:**

1. **Gait naturalness.** A well-trained Huggy develops a gait that looks almost organic — legs moving in coordinated pairs, body swaying naturally. Nobody programmed this. PPO found it as the most reward-efficient movement pattern.

2. **Recovery.** Push the agent (using the button in the demo). Watch it stumble, recalibrate, and continue. The policy has learned to recover from perturbations because training included random disturbances.

3. **Target tracking.** Move the ball while the agent is walking. It doesn't stop and replanning — it continuously adjusts its heading while maintaining its gait. This is the real-time adaptation that continuous control makes possible.

---

## Try it yourself

**Experiment 1 — No curriculum.**
Train from scratch on the full, moving-ball task with no curriculum. The agent takes 10× as many episodes to reach the same performance — and sometimes never reaches it. Sparse reward without curriculum is a very hard exploration problem.

**Experiment 2 — Noisy observations.**
Add Gaussian noise to the proprioceptive sensors. Watch the gait degrade gracefully — slightly less smooth, but still functional. Compare to a hand-coded locomotion controller under the same noise: it breaks immediately. Learned policies are naturally robust because training includes variation.

**Experiment 3 — Different morphologies.**
Unity ML-Agents includes several body configurations. Try a crawler (no legs, moves like a worm) and a spider (eight legs). The same PPO algorithm with the same reward function learns completely different locomotion strategies for each body. The algorithm adapts to the physics; you don't have to.

---

## What you've built

You started with the simplest possible problem: a row of slot machines and a question about which one to play.

From there, you built agents that remember which spots in a maze are worth visiting. Agents that navigate cities and manage power grids. Agents that control rocket thrusters and robot arms. Agents that coordinate in groups and develop emergent behaviour. Agents that learn from what humans prefer, not from reward functions we had to write by hand.

All of it from the same basic loop: observe, act, receive feedback, update.

RL is a very general idea. The specific algorithms are tools. What you've learned — the intuition for what each algorithm is doing and why — will make sense of every new algorithm you encounter, because they're all variations on the same themes.

The course ends here. But this is also where building starts.

[← Back to Welcome](../../)
