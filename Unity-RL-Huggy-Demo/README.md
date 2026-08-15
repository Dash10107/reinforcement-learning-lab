---
title: Unity RL Huggy Demo
emoji: 🐕
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Unity RL Huggy Demo — Deep Reinforcement Learning in a 3D Game Engine

<p align="center">
  <a href="https://dash10107.github.io/reinforcement-learning-lab/en/"><img src="https://img.shields.io/badge/Course_Chapter-Read-blue?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="Course Chapter"></a>
  <a href="https://huggingface.co/spaces/Dash10107/Unity-RL-Huggy-Demo"><img src="https://img.shields.io/badge/Live_Demo-Hugging_Face-yellow?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Demo"></a>
</p>

An interactive demonstration of a reinforcement learning agent trained entirely inside Unity using ML-Agents. The agent, Huggy, is a dog character that learns to fetch a stick thrown by the player. It has never been told how to run, turn, or time its approach — it discovered all of these behaviours through millions of trial-and-error episodes in a physics simulation.

**This project is part of the [Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab) — an interactive course and lab that bridges the gap between RL theory and practical implementation.**

---

## What this demonstrates

Most RL projects in this portfolio run in simple 2D grid worlds or low-dimensional physics simulations. Huggy is different. It runs inside a full 3D game engine with a ragdoll character, physics-based movement, and real-time rendering. The agent controls joint torques on a physically simulated dog body and must learn to coordinate all limbs to move purposefully toward a target.

This is an example of the kind of RL that is used in robotics, game AI, and animation — where the agent must learn to control a complex physical body rather than just choose directions on a grid.

---

## The Algorithm: Proximal Policy Optimization (PPO)

Huggy is trained using PPO, the same policy gradient algorithm used in the MARL Warehouse and Swarm Architect projects in this portfolio. PPO is Unity ML-Agents' default algorithm for continuous control because it is stable, general-purpose, and works well on both simple and complex tasks.

The training objective is to find a policy that maximises expected cumulative reward, subject to the constraint that each update does not change the policy too drastically:

```
actor_loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
```

For continuous control like limb movement, the actor outputs Gaussian distributions over joint torques rather than discrete action probabilities. At each step the agent samples from these distributions, applies the sampled torques to the physics simulation, and receives a reward signal.

**What makes this challenging:** The agent must coordinate multiple joints simultaneously. Moving one leg affects the torso, which affects balance, which affects where the other legs land. The reward is only given when Huggy gets close to the stick — there is no step-by-step guidance. The agent must discover locomotion, balance, and targeting entirely from this sparse signal over millions of physics steps.

---

## The Reward Structure

Huggy's reward during training comes from a single signal: getting close to the thrown stick earns positive reward, and touching the stick completes the episode with a large bonus. There is no explicit reward for walking correctly, maintaining balance, or approaching efficiently. The agent discovers all of these behaviours because they are necessary to consistently reach the target.

This is a key idea in modern RL: reward shaping (defining detailed intermediate rewards for sub-behaviours) often leads to agents that game the reward rather than learning the intended behaviour. Sparse, outcome-focused rewards — like Huggy's simple proximity signal — tend to produce more natural and robust behaviour, though they require more training time.

---

## Unity ML-Agents

Unity ML-Agents is an open-source toolkit that lets you train RL agents inside Unity scenes. It bridges the Unity game engine (C#, physics, rendering) with Python training frameworks (PyTorch, Stable-Baselines3). The training loop works like this:

1. The Unity scene runs the physics simulation and tracks the agent state
2. Observations (positions, velocities, sensor readings) are sent to the Python trainer
3. The Python trainer computes actions using the current policy
4. Actions (joint torques) are sent back to Unity and applied to the agent
5. The resulting reward is computed and sent back to Python
6. The Python trainer updates the policy using PPO

This back-and-forth happens thousands of times per second across multiple parallel Unity instances during training. The pre-trained model is then exported and can run inside Unity or in the browser via WebGL.

---

## How the Demo Works

The Hugging Face Space embeds a WebGL version of the Huggy environment that runs directly in your browser. No Unity installation is needed. The trained neural network policy runs locally in your browser using WebAssembly, controlling Huggy in real time.

You can click to throw the stick, and Huggy will run to fetch it. The physics and rendering all happen inside the browser tab.

---

## Project Structure

```
Unity-RL-Huggy-Demo/
├── app.py          Gradio app embedding the Unity WebGL demo
└── README.md
```

This project is intentionally minimal on the Python side — the interesting work happened during training inside Unity, and the demo just makes the result accessible through a web interface.

---

## Quick Setup

```bash
git clone https://github.com/Dash10107/reinforcement-learning-lab.git
cd Unity-RL-Huggy-Demo
pip install gradio
python app.py
```

Open `http://localhost:7860`. The demo iframe will load the interactive Unity WebGL environment.

---

## Training Huggy Yourself

If you want to train Huggy from scratch, you need Unity ML-Agents:

```bash
pip install mlagents
```

Then clone the ML-Agents repository, open the Huggy example scene in Unity, and run:

```bash
mlagents-learn config/ppo/Huggy.yaml --run-id=huggy_run_1
```

Training on a laptop CPU takes several hours. A GPU reduces this significantly. The Hugging Face Hub hosts pre-trained checkpoints that you can download and use directly.

---

## How this relates to the rest of the portfolio

Every other project in this repository uses simple 2D or low-dimensional environments. Huggy shows what RL looks like when you scale it to:

- A continuous, high-dimensional action space (joint torques rather than discrete directions)
- A physically simulated 3D body with complex dynamics
- A reward signal that is sparse and delayed
- Real-time inference running in a web browser

The core algorithm (PPO) is identical to what is used in the Swarm Architect and MARL Warehouse projects. The difference is entirely in the environment complexity. This is one of the most compelling aspects of modern RL: a single general-purpose algorithm can learn to play Atari games, coordinate robot swarms, manage energy grids, and make a simulated dog fetch a stick.

---

## Requirements

```
gradio>=6.0.0
```

---

## Things to Try

**1. Throw the stick in every direction.**
Click close, far, left, right, and behind. The policy was trained on random positions so it should handle all directions. Notice that direct frontal approaches are less stable due to physics.

**2. Redirect the stick mid-run.**
While Huggy is running toward one target, click somewhere else. The policy operates at every physics step so direction changes are near-instantaneous — there is no pre-planned trajectory to abandon.

**3. Compare the gait at different distances.**
At close range Huggy makes short, careful steps. At long range the gait opens up into a full run. The policy learned different movement patterns for different distances without being explicitly programmed to.

**4. Try to find positions where the policy fails.**
Throw the stick into corners or directly behind Huggy. Some positions are harder than others. A policy trained with uniform random targets may have blind spots for extreme angles — this is exactly the test distribution coverage problem in RL.

**5. Read the observation table in How It Was Trained and think about what is missing.**
Huggy observes joint angles, velocities, and stick position but not terrain shape, not other objects, not history. What additional observations might help? Would adding the stick velocity help? This analysis is how you design RL observation spaces in practice.

---

## Further Reading

- Unity ML-Agents documentation: https://github.com/Unity-Technologies/ml-agents
- Juliani et al., Unity: A General Platform for Intelligent Agents (2018) — the ML-Agents paper
- The Huggy project on Hugging Face: https://huggingface.co/huggy
- Deep RL Course on Hugging Face — Unit 1 covers Huggy from scratch: https://huggingface.co/learn/deep-rl-course
