---
title: AI Tutor A2C
emoji: 👁
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# AI Tutor Pro — Personalized Learning with Actor-Critic RL

An interactive reinforcement learning system that acts as an intelligent tutor. You set a student's current proficiency across five subjects, and a trained A2C agent recommends which subject to focus on next in order to reach mastery as efficiently as possible. You can watch it simulate a full learning path, inspect why it made each decision, and even retrain the agent from scratch.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Dash10107/AI-Tutor-A2C)

---

## What problem does this solve?

Imagine you have limited study time and five subjects to master. Which one should you pick up right now? The naive answer is always study the weakest subject. But that ignores forgetting — if you ignore a subject for too long, you lose ground on it. A tutor that has seen thousands of study sessions learns a smarter strategy: one that accounts for decay, sequencing, and the non-linear way humans learn.

This project trains an RL agent to discover that strategy on its own, without being told any rules.

---

## The Algorithm: Actor-Critic (A2C)

A2C stands for Advantage Actor-Critic. It belongs to a family of algorithms called policy gradient methods, which train a neural network to directly output what action to take rather than estimating a value table like Q-learning does.

A2C maintains two networks simultaneously.

The Actor looks at the current state (proficiency scores) and outputs a probability distribution over actions — which subject to study. Over time it learns to put higher probability on actions that lead to better outcomes.

The Critic looks at the same state and estimates how good that situation is overall, a single number called the value. It acts as a baseline that helps the actor understand whether an outcome was better or worse than average.

The advantage in the name is the key insight. Instead of telling the actor "this action was good," we tell it "this action was better than what we normally expect from this situation." That relative signal is much more informative and leads to faster, more stable learning.

The update at each step works roughly like this:

```
advantage    = (reward received + discounted future value) - value predicted by critic
actor_loss   = -log_prob(action_taken) * advantage
critic_loss  = (reward + gamma * next_value - predicted_value) squared
```

The actor improves by increasing the probability of actions with positive advantage. The critic improves by reducing its prediction error. They train together on the same experience, which is why A2C is computationally efficient.

---

## The Environment

The tutoring environment is a custom Gymnasium environment that simulates how students learn and forget.

**State space:** Five proficiency scores, each between 0 and 1. A score of 0 means no knowledge and 1 means full mastery. At the start of each episode scores are randomly initialised between 5% and 35% to represent a beginner student.

**Action space:** Five discrete choices, one for each subject: Mathematics, Physics, Literature, History, and Computer Science.

**What happens each step:**

When the agent picks a subject to study, that subject gains between 12% and 28% proficiency. All other subjects lose between 0.5% and 2.5% due to forgetting. The subject being studied does not decay.

**Reward:** The reward at each step equals the current proficiency of the subject that was just studied. This incentivises the agent to focus on subjects where it can achieve high proficiency, while accounting for the overall trajectory.

**Episode end:** The episode finishes when all five subjects reach 98% or higher, or after 200 steps as a hard cap.

This setup forces the agent to discover a nuanced scheduling strategy. Simply always picking the lowest-scoring subject does not work well because it ignores decay and the compounding effect of neglecting other subjects.

---

## Project Structure

```
AI-Tutor-A2C/
├── app.py                  main Gradio application
├── core/
│   ├── environment.py      custom Gymnasium environment
│   └── agent.py            A2C loading, inference, and training
├── viz/
│   └── charts.py           matplotlib charts for trajectory, radar, and training
├── tutor_model.zip         pre-trained A2C policy
└── requirements.txt
```

---

## Quick Setup

Clone and install dependencies:

```bash
git clone https://github.com/yourusername/reinforcement-learning-lab
cd AI-Tutor-A2C
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

**To try it out:** Set the five subject sliders to represent a student's current knowledge, click Analyse State to see what the policy recommends, then click Simulate Path to watch the agent run a full 20-step study plan. The Training Lab tab lets you retrain the agent from scratch.

---

## What each tab shows

**Dashboard:** Five sliders let you set proficiency levels. After clicking Analyse, three stat cards update — average proficiency, the policy's confidence, and the recommended subject. A live radar chart updates as you drag sliders. A bar chart shows the probability the policy assigns to each subject.

**Analytics:** After running a simulation you get a trajectory chart showing how each subject's proficiency evolves, with a colour-coded action bar below it. A second chart shows cumulative reward and how attention was distributed across subjects.

**Training Lab:** Choose how many timesteps to train for, hit Start, and watch the reward curve update live. The newly trained model loads automatically when finished.

**How A2C Works:** Explains the algorithm, the environment dynamics, and how to read each chart.

---

## Key hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| Learning rate | 7e-4 | Controls how fast the networks update each step |
| Gamma | 0.99 | How much the agent values future rewards vs immediate ones |
| n_steps | 5 | Number of steps collected before each gradient update |
| Entropy coefficient | 0.01 | Prevents the policy from becoming too certain too early |
| Max episode steps | 200 | Hard cap so episodes always terminate |

---

## Requirements

```
gymnasium
numpy
stable-baselines3
gradio>=6.0.0
matplotlib
```

---

## Further reading

- Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016) — the paper that introduced A3C, the precursor to A2C
- Stable-Baselines3 A2C docs: https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
- Sutton and Barto, "Reinforcement Learning: An Introduction" — Chapter 13 covers policy gradient methods in depth

## Things to Try

**1. Force a weak subject and watch the policy respond.**
Set Mathematics to 5% and all others to 60%. Click Analyse. The policy should strongly recommend Mathematics. Now raise Mathematics to 55% and click again — watch the recommended subject shift.

**2. Create an unbalanced student and simulate.**
Set one subject to 90% and all others to 15%. Run a 20-step simulation. Does the agent focus the weak subjects evenly or find a particular ordering?

**3. Compare different training budgets.**
In the Training Lab, train for 5,000 steps and note the final rolling reward. Then train for 50,000 steps. How much does the extra compute improve the policy confidence scores?

**4. Watch the entropy signal.**
After training, set all five subjects to exactly 50%. The confidence card shows how certain the policy is. A well-trained policy should still show some preference. A poorly trained one will be near 20% for all subjects.

**5. Set all subjects to 95% and analyse.**
With mastery nearly achieved the task is almost done. See how the policy reacts — does it spread attention evenly or still pick one subject aggressively?