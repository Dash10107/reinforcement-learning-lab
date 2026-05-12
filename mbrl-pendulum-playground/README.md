---
title: Mbrl Pendulum Playground
emoji: 🌀
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# MBRL Pendulum Playground — Model-Based Reinforcement Learning

A complete Model-Based Reinforcement Learning demonstration using the classic Pendulum control task. Instead of learning a policy directly from environment interactions, this system first learns a model of how the pendulum moves, then uses that model to plan actions without needing to interact with the real environment at every step. You can train the dynamics model, watch it imagine future trajectories, run a planning controller, and visualise where the model is uncertain.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Dash10107/mbrl-pendulum-playground)

---

## What problem does this solve?

Most RL algorithms learn through trial and error directly in the environment. This works, but it is expensive: you need millions of real interactions before the policy becomes good. In the physical world — robots, chemical plants, medical devices — those interactions take real time and can cause real damage.

Model-Based RL takes a different route. You first spend a small number of real interactions collecting data, then train a neural network to predict what happens when you take each action. Once you have this learned world model, you can simulate millions of experiences in your head — fast and safely — and use those to plan.

The tradeoff is that your model is never perfect. Planning inside an imperfect model can lead you astray, especially over long horizons. This is one of the central challenges in MBRL, and this project lets you explore it directly through the imagination rollout visualization.

---

## The Algorithm: MBRL with Random Shooting MPC

This project implements two distinct components: learning the world model, and planning with it.

### Learning the Dynamics Model

The dynamics model is a neural network that predicts the next state of the pendulum given the current state and action:

```
f(state, action) → next_state
```

Training data is collected by running a random policy in the real environment — just taking random torques and recording what happens. This gives us a dataset of (state, action, next_state) tuples. The model is then trained to minimise the mean squared error between its predictions and the real outcomes.

**Ensemble for uncertainty:** Rather than a single model, we train five independent networks on different random subsets of the data (bootstrap sampling). At inference time, all five models make predictions. The mean of those predictions is used as the best estimate. The standard deviation across predictions is used as an uncertainty estimate — if the five models disagree strongly, the model is uncertain about that region of state space.

This is called an ensemble approach, and it gives us a principled way to know when not to trust the model.

### Planning with Model Predictive Control

Once the model is trained, we use it for planning via Random Shooting MPC (Model Predictive Control):

1. At the current state, sample 512 random action sequences, each of length H (planning horizon)
2. Roll each sequence forward through the learned model, computing the predicted total reward
3. Select the sequence with the highest predicted reward
4. Execute only the first action from that sequence in the real environment
5. Observe the real next state and repeat

The reason for executing only the first action and replanning is that the model's predictions become less accurate the further into the future you look. By replanning at every step, you correct for model errors before they accumulate.

The reward function used during planning is the analytical Pendulum-v1 reward:

```
reward = -(theta^2 + 0.1 * angular_velocity^2 + 0.001 * torque^2)
```

where theta is the angle from vertical. Perfect balance scores 0. Any deviation scores negative.

---

## The Pendulum Environment

The Pendulum-v1 task from Gymnasium is a classic continuous control problem.

**State (3-dimensional):**

| Component | Range | Meaning |
|---|---|---|
| cos(theta) | -1 to 1 | Cosine of the pole angle |
| sin(theta) | -1 to 1 | Sine of the pole angle |
| angular velocity | -8 to 8 | Speed and direction of rotation |

The angle is encoded as cos and sin rather than directly to avoid a discontinuity: when theta passes through 180 degrees, the value wraps from π to -π, creating a sudden jump that is hard for neural networks to learn. The cos/sin encoding is smooth everywhere.

**Action:** A single continuous torque value between -2 and 2 Newton-metres.

**Goal:** Swing the pendulum up from hanging at the bottom and balance it upright at theta = 0.

---

## The Imagination Rollout

This is the most educational feature of the project. Starting from a given state, both the real environment and the learned model are stepped forward for N steps using the same sequence of actions. The resulting trajectories are plotted side by side.

What you will see:
- For short horizons (5-10 steps), the model closely tracks reality
- For longer horizons (30-50 steps), the model trajectory drifts away from the real trajectory
- The ensemble uncertainty (shown as a shaded band) widens as the horizon grows

This directly demonstrates why MBRL uses short planning horizons. A model that is accurate for 10 steps may produce meaningless predictions at 50 steps due to compounding errors.

---

## Project Structure

```
mbrl-pendulum-playground/
├── app.py                    main Gradio application
├── model/
│   ├── dynamics.py           DynamicsModel, EnsembleDynamics, reward function
│   └── train.py              data collection and ensemble training pipeline
├── planning/
│   └── mpc.py                Random Shooting MPC and episode rollout
├── viz/
│   └── plots.py              imagination charts, training curves, uncertainty heatmap
├── dynamics_model.pth        pre-trained single-model checkpoint
└── requirements.txt
```

---

## Quick Setup

```bash
git clone https://github.com/yourusername/rl-portfolio
cd mbrl-pendulum-playground
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

**Suggested starting point:** Go to the Explorer tab. Set cos(theta) to -1 and sin(theta) to 0 — this places the pendulum hanging straight down. Apply a torque of 0.5 and click Predict. You will see the bar chart comparing what the real environment predicts versus what the learned model predicts. Then go to the Imagination tab, select "Hanging down" as the starting state, set the horizon to 25 steps, and click Imagine. Watch where the model's prediction diverges from reality.

To see the full MBRL loop in action, go to the MPC Control tab and click Run MPC Episode. This uses the learned model to plan 200 steps of the real pendulum.

---

## What each tab shows

**Explorer:** Single-step state explorer. Set pendulum state and action, see real vs predicted next state as a bar chart. If the ensemble is loaded, uncertainty bars (95% confidence interval) appear alongside predictions.

**Train Model:** Background training pipeline. Set number of transitions to collect and training epochs, click Start. A live train/validation loss curve updates as training progresses. Lower validation loss means more accurate predictions.

**Imagination:** Multi-step rollout comparison. Choose a starting state preset, set the horizon length, and click Imagine. Four panels show: the three state dimensions plotted over time (real in solid purple, imagined in dashed cyan), and a cumulative prediction error chart showing how MSE grows with horizon.

**MPC Control:** Full episode using the planning algorithm. Configure planning horizon and number of candidate sequences, click Run. You get an animated GIF of the pendulum swinging and balancing, plus an analysis chart showing reward per step, torque sequence, and how cos(theta) evolves toward 1.0 (upright).

**Uncertainty Map:** Ensemble disagreement across the (theta, angular velocity) state space with zero torque applied. Bright regions show where the model has seen little training data and its predictions are unreliable.

**How It Works:** Full explanation of MBRL, the ensemble, MPC random shooting, compounding errors, and how this relates to model-free RL.

---

## Key design choices

**Why five models in the ensemble?** Fewer models give noisier uncertainty estimates. More than five gives diminishing returns while slowing down training and inference. Five is a practical sweet spot.

**Why random shooting rather than gradient-based planning?** Gradient-based planning (differentiating through the model to find the optimal action sequence) can get trapped in local optima and requires differentiable dynamics. Random shooting is simpler, parallelisable, and works well when the action space is low-dimensional.

**Why a short planning horizon?** Model errors compound multiplicatively. A model with 1% error per step has roughly 10% error after 10 steps and over 60% error after 100 steps. The MPC horizon of 15-20 steps is chosen to stay in the reliable region.

---

## Requirements

```
gradio>=6.0.0
torch
numpy
gymnasium[classic_control]
matplotlib
```

---

## Things to Try

**1. Compare imagination accuracy at different horizons.**
Run the Imagination rollout at 5, 25, and 50 steps from the Hanging Down start state. Watch the cumulative MSE panel. Error at step 50 should be dramatically higher than at step 5 — compounding model error made visible.

**2. Train with very little data vs plenty.**
Try 500 transitions and check validation loss. Then try 3000 transitions. The well-trained model should track the real trajectory much more closely in the Imagination tab.

**3. Run MPC with a short vs long planning horizon.**
Run with horizon 5 and 64 candidates. Then horizon 20 and 512 candidates. Short-horizon control is more reactive. Long-horizon control plans further ahead and should produce smoother swings.

**4. Check the uncertainty map after training.**
In the Uncertainty tab click Compute. Bright regions show where the five models disagree most — typically the extreme corners of state space where training data was sparse. This is exactly where the MPC should plan most cautiously.

**5. Compare pre-trained vs freshly trained ensemble.**
The pre-trained checkpoint loads by default. Train a fresh ensemble then run MPC again. A better model should lead to more confident, less erratic pendulum control.

---

## Further Reading

- Deisenroth and Rasmussen, PILCO: A Model-Based and Data-Efficient Approach to Policy Search (2011) — early influential MBRL paper
- Chua et al., Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (2018) — PETS algorithm, closely related to this project
- Nagabandi et al., Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning (2018) — combining model-based and model-free approaches
