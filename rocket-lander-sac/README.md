---
title: Rocket Lander Sac
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Rocket Lander SAC — Soft Actor-Critic Continuous Control

A reinforcement learning system that lands a rocket using Soft Actor-Critic, one of the best algorithms for continuous control tasks. The trained agent controls two engine throttles in real time to bring the lander down safely. You can simulate multiple landing attempts, change environmental conditions like gravity and wind, inspect detailed telemetry for each episode, and fine-tune the agent in the browser.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Dash10107/rocket-lander-sac)

---

## What problem does this solve?

Landing a rocket is a continuous control problem. The agent must output precise throttle values — not just "fire" or "don't fire," but exactly how much thrust to apply to each engine at every moment. Small errors compound: too much thrust at the wrong time sends the lander spinning; too little and it crashes.

This is fundamentally different from discrete-action problems like maze solving or game playing. The action space is infinite, which rules out Q-tables and makes naive policy gradient methods unstable. SAC was designed specifically for problems like this.

---

## The Algorithm: Soft Actor-Critic (SAC)

SAC is an off-policy, maximum-entropy deep RL algorithm. The "maximum entropy" part is the distinguishing idea: instead of just maximising expected reward, SAC also maximises the entropy (randomness) of its policy. This leads to more exploration, more robust behaviour, and often faster learning.

The training objective is:

```
J(pi) = sum over time of E[reward(s_t, a_t) + alpha * H(pi(· | s_t))]
```

where H is the entropy of the policy and alpha is the temperature — a tunable parameter that controls how much exploration to encourage. SAC automatically adjusts alpha during training to hit a target entropy level.

**Four networks run simultaneously during training:**

The **Actor** outputs a Gaussian distribution over actions: a mean and a log standard deviation for each engine throttle. Actions are sampled from this distribution during training (for exploration) and the mean is used at test time (for deterministic execution).

**Two Critics** (Q-networks) each estimate the expected return from a (state, action) pair. SAC uses two critics and takes the minimum of their predictions to avoid overestimation — a technique borrowed from TD3. The Bellman target with entropy is:

```
y = reward + gamma * (min(Q1, Q2)(next_state, next_action) - alpha * log_pi(next_action | next_state))
```

**Two Target Critics** are slow-moving copies of the critics, updated via exponential moving average:

```
target_params = tau * online_params + (1 - tau) * target_params
```

This prevents the training targets from changing too fast, which would cause oscillation.

**Experience Replay:** SAC is off-policy, meaning it learns from a buffer of past transitions rather than only from the most recent experience. This makes it highly sample efficient — one real interaction can contribute to many gradient updates.

The combination of off-policy learning, dual critics, and entropy regularisation makes SAC one of the most sample-efficient and stable algorithms available for continuous control.

---

## The LunarLander Environment

The environment is LunarLander-v3 with continuous actions from OpenAI Gymnasium.

**State (8-dimensional):**

| Component | Meaning |
|---|---|
| x position | Horizontal position (0 = pad centre) |
| y position | Altitude above ground |
| x velocity | Horizontal speed |
| y velocity | Vertical speed (negative = falling) |
| angle | Tilt from vertical |
| angular velocity | Rotation speed |
| left leg contact | 1 if left leg touches ground |
| right leg contact | 1 if right leg touches ground |

**Action (2-dimensional continuous):**

- Main engine throttle: -1 to +1 (negative disables, positive fires with variable intensity)
- Lateral thruster: -1 to +1 (negative fires left, positive fires right)

**Reward:** The environment provides a dense reward signal:
- Positive reward for moving toward the landing pad
- Negative reward for each leg that is not on the ground
- Bonus of +100 when both legs touch down safely
- Penalty of -100 for crashing
- Small fuel cost for each engine firing

A total episode reward above 200 is considered a successful landing. Above 150 is a passing grade.

---

## Environmental Conditions

The app lets you change the physics beyond the defaults.

**Gravity:** Reducing gravity (less negative than -10) makes landing easier because the lander falls more slowly. Increasing it makes the descent steeper and harder to control.

**Wind disturbance:** Enabling wind adds a lateral force that varies each step. The agent must learn to counteract it with the lateral thruster. Higher wind power and turbulence values stress-test the policy.

These options let you explore how robust the trained policy is and what happens when you push it outside its training distribution.

---

## Project Structure

```
rocket-lander-sac/
├── app.py                  main Gradio application
├── core/
│   ├── mission.py          StepData, EpisodeResult, MissionResult, run_mission()
│   └── trainer.py          SAC training pipeline with live callbacks
├── viz/
│   ├── charts.py           mission dashboard, comparison charts, training curves
│   └── replay.py           animated GIF export with HUD overlay
├── sac_rocket_lander.zip   pre-trained SAC model
└── requirements.txt
```

---

## Quick Setup

```bash
git clone https://github.com/yourusername/rl-portfolio
cd rocket-lander-sac
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

**Suggested starting point:** Go to the Mission Control tab, leave gravity at -10 and wind off, set landing attempts to 3, enable the animated replay, and click Initiate Launch Sequence. You will see an animated GIF of the lander descending, a four-panel mission dashboard (reward bars, flight trajectory, cumulative reward, engine throttle), and a stats table. Then enable wind and run again to see how the policy handles disturbance.

---

## What each tab shows

**Mission Control:** Configure gravity, wind, number of attempts, and vehicle. Click Run to execute all attempts. Results appear as a four-panel dashboard showing per-episode reward bars, 2D flight trajectories, cumulative reward over steps, and engine throttle profiles. A GIF replay with HUD overlay (step counter, reward, throttle bars burned in) appears for single episodes or as a side-by-side comparison for multiple episodes.

**Training Lab:** Fine-tune or retrain the SAC agent from scratch. Choose total timesteps, learning rate, and batch size. A live training dashboard shows episode reward history, actor and critic loss curves, and the entropy coefficient as it adjusts during training. The fine-tuned model saves automatically and is used in future missions.

**Algorithm Guide:** Full explanation of SAC, the LunarLander environment state and action spaces, the training hyperparameters, and how to read each chart.

---

## Key hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| Learning rate | 3e-4 | Shared by actor and both critics |
| Buffer size | 1,000,000 | Replay buffer capacity |
| Batch size | 256 | Transitions sampled per gradient update |
| Tau | 0.005 | Soft target network update rate |
| Gamma | 0.99 | Discount factor — values long-term stability |
| Target entropy | -2.0 | Desired policy entropy for auto-tuning alpha |

---

## Reading the Mission Dashboard

**Reward bars:** Green bars above 200 indicate perfect landings. Bars above 150 are successful. Below zero means crash.

**Flight trajectory:** The (x, y) path from launch to landing. A successful flight converges toward the centre (x=0) and descends smoothly. A crash shows an erratic path or steep final descent.

**Engine throttle chart:** Shows when the main engine fired and when lateral thrusters corrected drift. A well-trained agent uses brief, precise bursts rather than continuous full throttle.

---

## Requirements

```
gradio>=6.0.0
stable-baselines3[extra]
gymnasium[box2d]
shimmy
matplotlib
```

---

## Things to Try

**1. Run 5 landing attempts and read the variance.**
Even a trained deterministic policy produces different outcomes across episodes. The success rate card quantifies this. Try to identify what differs between successful and failed attempts in the trajectory charts.

**2. Enable wind and see how the policy degrades.**
Run 3 attempts with no wind. Enable wind at power 10 and run again. At power 18 the pre-trained policy may fail consistently — it was not trained under those conditions. This is distribution shift.

**3. Change gravity and find the breaking point.**
Default gravity is -10 (Earth). Set it to -5 (Moon-like) and landing becomes easier. Set it to -18 and rapid descent gives less time to correct. Find the gravity where success rate drops to roughly 50%.

**4. Read the throttle chart for a success vs a crash.**
Successful landings usually show a burst of main engine activity at altitude followed by brief lateral corrections near the pad. Crashes often show erratic throttle or sustained full thrust too late. The pattern difference is the learned control strategy.

**5. Fine-tune in the Training Lab and compare.**
Run 10,000 timesteps of fine-tuning. Then run 5 mission attempts. Fine-tuning a pre-trained model is much faster than training from scratch — you should see meaningful improvement in a short run.

---

## Further Reading

- Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (2018) — the original SAC paper
- Haarnoja et al., Soft Actor-Critic Algorithms and Applications (2019) — the follow-up with automatic entropy tuning
- Stable-Baselines3 SAC documentation: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
