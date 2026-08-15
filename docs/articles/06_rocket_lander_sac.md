---
title: "Precision Engineering: Landing a Rocket with Soft Actor-Critic (SAC)"
published: true
description: "How do you teach an AI to control continuous rocket thrusters? We dive into the leap from Discrete to Continuous Control using the Soft Actor-Critic algorithm."
tags: "machinelearning, reinforcementlearning, ai, robotics"
series: "Reinforcement-Learning"
cover_image: "https://raw.githubusercontent.com/Dash10107/reinforcement-learning-lab/main/assets/rocket_sac_cover.png"
---

![Rocket SAC Cover](https://raw.githubusercontent.com/Dash10107/reinforcement-learning-lab/main/assets/rocket_sac_cover.png)

In the previous projects in this portfolio, our AI agents interacted with the world using a "D-pad." The warehouse robots could move *Up, Down, Left, or Right*. The smart grid battery could *Charge* or *Discharge*. These are called **Discrete Actions**. 

But the physical world rarely works like a D-pad. It works like a steering wheel.

If you are trying to land a multi-million dollar rocket back onto a landing pad, you cannot just tell the main engine to "Turn On" or "Turn Off." You have to output an exact, continuous value. You might need exactly `42.7%` thrust on the main engine, while simultaneously firing the left lateral thruster at `14.2%`. If you output too much thrust, the rocket shoots back into the atmosphere. If you output too little, it violently crashes into the concrete.

This is the frontier of **Continuous Control**.

To solve this, I built the **[Rocket Lander Simulator](https://huggingface.co/spaces/Dash10107/rocket-lander-sac)**. In this project, we transition away from the algorithms we've used so far and introduce one of the most powerful continuous control algorithms in modern robotics: **Soft Actor-Critic (SAC)**.

---

## 1. The Curse of the Continuous

Why can't we just use the Deep Q-Networks (DQN) we used to solve our Smart Grid and Logistics projects? 

DQN is a discrete algorithm. It calculates the expected future reward for a fixed list of actions, and then picks the highest one. If you wanted to use DQN to land a rocket, you would have to chop the engine throttle into discrete buckets: 10%, 20%, 30%, etc. 

But what if the perfect landing requires exactly `15.5%` thrust? If you create a bucket for every possible decimal percentage across multiple engines, the number of possible combinations explodes into the millions. The neural network would freeze, unable to compute the math in real time. 

We need an algorithm that doesn't choose from a list, but rather *generates* a highly precise continuous number.

### The Physics of the Action Space
To achieve this, the neural network in our simulation outputs an array of two continuous numbers, both bounded between `-1.0` and `1.0`:
1. **Main Engine:** Negative values mean the engine is off. Positive values map smoothly to thrust intensity (e.g., `0.5` = 50% thrust).
2. **Lateral Thrusters:** Negative values fire the left thruster (pushing the rocket right). Positive values fire the right thruster.

Because these outputs are mathematically continuous, the AI can make micro-adjustments smaller than a fraction of a percent to perfectly balance the rocket.

---

## 2. Generating Precision: The Gaussian Actor

In the Warehouse project, we introduced the **Actor-Critic** architecture. SAC uses this same dual-brain setup, but the Actor behaves entirely differently.

Instead of outputting probabilities for discrete buttons, the SAC Actor outputs the mathematical parameters of a **Gaussian Distribution** (a Bell Curve). Specifically, for every engine, it outputs two variables:
1. **The Mean (μ):** What the AI thinks the exact perfect throttle percentage is.
2. **The Standard Deviation (σ):** How confident the AI is in that guess (the spread of the curve).

Here is exactly what that looks like in PyTorch. The neural network outputs the mean and standard deviation, and we sample an action from that curve:

```python
import torch
from torch.distributions import Normal

# 1. The Neural Network outputs the parameters of the Bell Curve
mean = actor_network(state)
log_std = torch.clamp(actor_log_std, min=-20, max=2)
std = log_std.exp()

# 2. We build the mathematical Bell Curve
distribution = Normal(mean, std)

# 3. We sample a random throttle percentage from the curve
action = distribution.rsample() 

# 4. Squeeze it between -1.0 and 1.0 for the physics engine
action = torch.tanh(action)
```

When the AI is completely untrained, its Standard Deviation is massive. The code above will wildly fire the engines at random percentages. As the Critic network slowly coaches the Actor, the Standard Deviation mathematically shrinks. The Bell Curve becomes a sharp, narrow spike, and the AI outputs highly precise, deterministic throttle commands.

But landing a rocket is dangerous. We have to prevent the AI from becoming recklessly overconfident.

---

## 3. Curing Neural Optimism (The Twin Critics)

Neural Networks suffer from a well-documented psychological flaw: **Optimism Bias**. 

If an untrained AI accidentally fires its engines at 100% and miraculously avoids crashing due to a lucky gust of wind, the Critic network might immediately assume that 100% thrust is a genius move. It vastly overestimates the value of that action. In a physical simulation, optimism leads to catastrophic, vehicle-destroying crashes.

To cure this, SAC introduced a brilliant engineering trick: **Twin Critics**.

Instead of having one Critic coach the Actor, SAC uses *two* completely independent Critic networks (`Q1` and `Q2`). When the Actor asks, "How many points will I get if I fire the engine at 42%?", both Critics calculate an answer. 

The algorithm mathematically forces the Actor to take the **minimum** of the two predictions:

```python
# Critic 1 predicts the future reward
q1_value = critic_1(state, action)

# Critic 2 independently predicts the future reward
q2_value = critic_2(state, action)

# The AI assumes the absolute worst-case scenario
target_q_value = torch.min(q1_value, q2_value)
```

By forcing the AI to assume the absolute worst-case scenario, SAC completely eliminates optimism bias. The AI only attempts a dangerous maneuver if *both* pessimistic coaches agree that it is completely safe.

---

## 4. Maximum Entropy RL: Rewarding Chaos

The most defining feature of Soft Actor-Critic is the word "Soft". This refers to **Maximum Entropy Reinforcement Learning**.

In standard RL, the goal is simple: Maximize the Reward. 
But SAC changes the fundamental equation of AI by adding a new term to the objective function:

**Objective = Maximize (Reward + α * Entropy)**

Entropy is the mathematical measurement of chaos and randomness. Why on earth would we actively reward a rocket for flying chaotically? 

Imagine an AI that finds exactly one perfect, elegant path to the landing pad. It memorizes that path perfectly. But what happens during a real flight if a massive gust of wind blows the rocket 10 feet to the left? Because the AI only memorized one path, it has no idea what to do, panics, and crashes.

By mathematically rewarding Entropy (scaled by a temperature parameter `α` or `alpha`), we actively force the AI to *not* memorize a single path. 

```python
# Calculate how 'predictable' the AI's action was
log_prob = distribution.log_prob(action)

# If the action was highly unpredictable (high entropy),
# the log_prob is very low/negative. We subtract it to give a bonus!
actor_loss = (alpha * log_prob) - min_q_value 
```

We force it to discover 1,000 different, slightly messy ways to land the rocket. By forcing the AI to explore the chaos, it builds an incredibly robust, generalized intuition. When that massive gust of wind hits it in the real world, the AI doesn't panic—it has already explored that exact chaotic state during its training and knows exactly how to dynamically recover.

---

## 5. Liquid Memory and Sample Efficiency

If you've ever trained a neural network, you know they are incredibly data-hungry. To solve this, SAC is designed to be an **Off-Policy** algorithm with a massive **Replay Buffer**. 

Instead of throwing away data after every flight, SAC records every single millisecond of telemetry (State, Action, Reward, Next State) into a memory bank that holds 1,000,000 steps. While the rocket is flying, the AI is constantly "daydreaming" about old flights, randomly sampling batches of past mistakes to squeeze every possible drop of mathematical insight out of them. This makes SAC incredibly **Sample Efficient**.

### Polyak Averaging (The Smooth Update)
Because the AI is learning so aggressively from its past, we run into a stability problem. If the Target Critic network updates its weights too abruptly, the AI loses its mind and unlearns how to fly. 

To fix this, SAC uses a trick called **Polyak Averaging**. Instead of replacing the Target network's brain entirely, it updates using an exponential moving average (`tau = 0.005`). 
Every step, the new brain is composed of `99.5%` of the old weights, and `0.5%` of the newly learned weights. This creates an ultra-smooth, liquid learning curve that prevents the math from oscillating wildly.

---

## 🧪 Try It Yourself

To truly appreciate the power of SAC, you have to watch its continuous precision and test its limits. Open the **[Rocket Lander Simulator](https://huggingface.co/spaces/Dash10107/rocket-lander-sac)** and run these engineering tests:

1. **Read the Telemetry:** Go to the Mission Control tab and run a baseline flight. Look at the `Engine Throttle` chart at the bottom. You won't see blocky, ON/OFF steps. You will see incredibly smooth, continuous curves as the AI perfectly modulates the main engine to counter gravity, followed by lightning-fast micro-bursts from the lateral thrusters to correct its angle.
2. **The Wind Turbulence Test:** This is where the Entropy training shines. Turn the `Wind Power` up to 15, and the `Turbulence` up to 1.5. Run the mission again. Watch the replay GIF. You will see the rocket get violently pushed off-course by invisible forces, but because of its Maximum Entropy training, it dynamically corrects its continuous thrusters in real-time to fight the wind and hit the pad.
3. **The Sim-to-Real Gap (Distribution Shift):** Change the gravity from `-10.0` (Earth) to `-5.0` (The Moon). If the pre-trained AI crashes, you are witnessing **Distribution Shift**. Because the AI was only trained on Earth, its Bell Curves are perfectly tuned for Earth physics. When the physical rules change, the AI fails. This is exactly why deploying robotics in the real world is so hard—simulators never perfectly match reality. 
4. **Fine-Tune the Brain:** If the AI failed the Moon test, go to the Training Lab. Run 10,000 timesteps of fine-tuning, and then run the Moon mission again. Watch how quickly the Replay Buffer adapts the weights to the new gravity.

---

### Wrapping Up

Moving from discrete grids to continuous physical simulations is the holy grail of robotics. It requires us to abandon simple lookup tables and embrace algorithms like Soft Actor-Critic—managing infinite action spaces with Gaussian distributions, curing optimism with Twin Critics, building physical robustness by actively rewarding chaos, and stabilizing learning with liquid memory updates.

This is the sixth of 12 interactive RL projects I am building to bridge the gap between academic math and real-world intuition. If this deep dive into continuous control helped clarify how real robots think, I would be incredibly grateful if you checked out the source code and dropped a star on the full repository:

⭐ **Reinforcement Learning Lab on GitHub**{% embed https://github.com/Dash10107/reinforcement-learning-lab %}

Let me know in the comments: *If you had to write the reward function for a self-driving car, what penalty would you assign to a bumpy brake versus a slow arrival?*
