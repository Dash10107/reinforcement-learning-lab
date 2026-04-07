---
title: Smart Grid Energy Optimizer
emoji: ⚡
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Smart Grid Energy Optimizer — DQN Battery Storage Management

A reinforcement learning platform for managing a battery energy storage system connected to an electricity grid. The agent learns when to charge the battery (buy cheap electricity), when to discharge it (sell at peak prices), and how to factor in solar generation and building load — all without knowing future prices in advance. Three strategies are compared side by side: a DQN agent, an optimal dynamic programming solver, and a simple rule-based heuristic.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Dash10107/smart-grid-energy-optimizer)

---

## What problem does this solve?

Electricity prices change every hour. At 3am they might be 6 cents per kilowatt-hour. At 6pm during peak demand they might be 60 cents. A battery system can buy electricity when it is cheap, store it, and sell or use it when prices are high. This is called energy arbitrage, and done well it can significantly reduce electricity costs and carbon emissions.

The challenge is that you do not know the future. You do not know whether prices will rise or fall in the next two hours. A rule-based system can use simple thresholds — charge when price is below average, discharge when price is above average — but it misses the nuanced timing that a learned policy can find.

This project trains a DQN agent to make those decisions, then compares it against the theoretical best possible strategy (dynamic programming with perfect foresight) to measure how close the learning agent gets to optimal.

---

## The Algorithm: Deep Q-Network (DQN)

DQN learns a Q-function: given the current state of the grid (time of day, battery charge level, current price, solar output, building load), it estimates the expected future revenue for each possible charging action.

```
Q(state, action) = immediate_reward + gamma * max Q(next_state, all_actions)
```

The agent always chooses the action with the highest Q-value. Training uses experience replay and a target network for stability — the same techniques from the original Atari DQN paper.

**Why DQN rather than continuous-action algorithms like SAC?** The action space here is discretised into seven levels from -3kW (full discharge) to +3kW (full charge) in 1kW steps. This makes discrete DQN a natural fit and keeps the problem simple enough to train quickly in the browser.

**What the DQN learns that rule-based cannot:** The rule-based strategy knows today's price but not tomorrow's. The DQN has seen thousands of 24-hour episodes and has learned patterns — for instance, that prices almost always spike in the late afternoon, so it should be fully charged by 2pm. It also learns to account for solar generation: if the sun is producing free power right now, there is no need to draw from the grid for charging.

---

## The Three Strategies

**DQN Agent:** A neural network with three hidden layers (128, 128, 64 neurons) trained using Stable-Baselines3. It receives an 8-dimensional observation and selects one of seven charge/discharge actions. It has no access to future prices — it must rely on patterns learned from training.

**Dynamic Programming (DP) Optimal:** Uses backward induction over a discretised state space (50 SOC levels × 24 hours) to compute the provably optimal policy given perfect knowledge of all future prices, solar generation, and load. This is the theoretical upper bound. No real-world system can match it because it requires knowing the future. It exists to show how close the DQN gets.

**Rule-Based Heuristic:** Charges when the current price is below the median by more than 25%, discharges when it is above the median by more than 25%, and holds otherwise. This mimics what a simple automated controller would do. It is the baseline that any intelligent system should beat.

**No-Storage Baseline:** Simply buys all electricity from the grid at whatever the current price is, with no storage. Comparing against this shows the total economic value of having a battery at all.

---

## The Environment

The environment models a 24-hour operating cycle with one decision per hour.

**Observation (8-dimensional):**

| Feature | Encoding | Meaning |
|---|---|---|
| sin(hour × pi/12) | cyclic | Smooth encoding of time of day |
| cos(hour × pi/12) | cyclic | Paired with sin for full hour information |
| SOC (normalised) | -1 to 1 | Current battery charge level |
| Price (normalised) | -1 to 1 | Is today's price high or low relative to the day? |
| Price trend | -1 to 1 | Is the price rising or falling from last hour? |
| Solar (normalised) | -1 to 1 | Current solar generation output |
| Load (normalised) | -1 to 1 | Current building demand |
| Time to peak | -1 to 1 | How many hours until the typical daily price peak |

**Action (7 discrete levels):** -3, -2, -1, 0, +1, +2, +3 kilowatts. Negative means discharging, positive means charging, zero means holding.

**Reward:** Net revenue from grid transactions, solar exports, and demand offset in each hour. The battery has physical constraints: it cannot charge beyond its capacity, cannot discharge below empty, and loses some energy to heat during each charge-discharge cycle (efficiency of 92%).

**Stochastic training:** During training, prices are perturbed with 12% Gaussian noise and solar generation is multiplied by a random cloud factor. This forces the agent to learn a robust policy rather than memorising a single price schedule.

---

## Scenario Presets

| Scenario | Price Profile | Solar | Description |
|---|---|---|---|
| Summer Peak | High evening spike to 70c/kWh | 5kW system | Hardest arbitrage challenge |
| Winter Mild | Moderate midday peak | 2kW system | Reduced solar, less extreme prices |
| Grid Crisis | Extreme prices up to 250c/kWh | 4kW system | Tests policy under stress |
| Residential | Summer profile | 3kW rooftop | Home energy management context |
| No Solar | Summer profile | None | Pure price arbitrage baseline |

---

## Project Structure

```
smart-grid-energy-optimizer/
├── app.py                    main Gradio application
├── env/
│   ├── grid_env.py           SmartGridEnv — Gymnasium-compatible BESS environment
│   └── scenarios.py          price profiles, solar generation, load profiles
├── agents/
│   ├── dqn_agent.py          DQN training and inference with live callbacks
│   ├── dp_solver.py          Dynamic programming optimal solver
│   └── naive.py              rule-based and no-storage baselines
└── requirements.txt
```

---

## Quick Setup

```bash
git clone https://github.com/yourusername/rl-portfolio
cd smart-grid-energy-optimizer
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

**Suggested starting point:** Go to the Dispatch tab, select the Summer Peak scenario, leave strategy on DP Optimal, and click Run Dispatch. The 24-hour dashboard will show the price curve, battery SOC trajectory, energy flows (solar/grid/battery), and hourly profit and loss. Then switch to Benchmark to run all four strategies and compare them on the same day.

To train your own DQN, go to the Training Lab tab. Start with 20,000 timesteps for a quick run. The reward curve and rolling mean update as training progresses.

---

## What each tab shows

**Dispatch:** Run any single strategy on any scenario. The main chart has three panels: electricity price and battery SOC over 24 hours, stacked energy flows (solar generation, grid import, battery charging and discharging, building load), and hourly profit/loss bars. Four KPI cards show daily revenue, solar self-sufficiency, battery cycles used, and peak grid import.

**Benchmark:** Run all four strategies on the same day with the same conditions. Charts show SOC trajectories and cumulative revenue for each strategy, grid import profiles, and a revenue comparison bar chart. The gap between DQN and DP Optimal shows how much room remains for improvement.

**Training Lab:** Train the DQN from scratch or continue fine-tuning. Configurable timesteps, learning rate, and batch size. A live training dashboard shows reward history and rolling mean. The model saves automatically and loads for the next Dispatch run.

**Technical Guide:** Full explanation of the reward function formula, state encoding, DQN hyperparameters, how to read each chart, and the difference between DP optimal and RL.

---

## Key hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| Gamma | 0.97 | High discount — values future peak revenue |
| Exploration fraction | 0.30 | 30% of training is random exploration |
| Learning starts | 500 | Fills replay buffer before first update |
| Network architecture | 128, 128, 64 | Three hidden layers |
| Battery efficiency | 0.92 | 92% round-trip efficiency loss |

---

## Requirements

```
gradio>=6.0.0
numpy
matplotlib
stable-baselines3[extra]
gymnasium
```

---

## Things to Try

**1. Run all four strategies on Summer Peak and compare revenue.**
The gap between DP Optimal and DQN is how much the agent has left to learn. The gap between Rule-Based and No Storage shows the minimum value of having any battery at all.

**2. Watch the battery charge timing on Summer Peak.**
Electricity prices spike sharply around hours 15-17. In the Dispatch chart, a good strategy should have the battery near full by hour 14, charged during the cheap overnight hours, and discharged heavily during the spike.

**3. Compare EV vs Diesel on the Grid Crisis scenario.**
With extreme price spikes the relative revenue numbers differ but the agent's charge and discharge timing should be similar. The strategy is driven by price patterns, not vehicle type.

**4. Train for 10,000 steps then benchmark. Then 30,000 steps and benchmark again.**
The gap between DQN and DP Optimal should narrow with more training. This makes RL as an iterative improvement process tangible rather than abstract.

**5. Try No Solar and compare with Summer Peak.**
Without solar the agent cannot use free midday power — it must buy everything from the grid. How does this change the optimal charge timing? Does the DQN adapt, or does it behave as if solar were still available?

---

## Further Reading

- Mnih et al., Human-level control through deep reinforcement learning (2015) — the DQN paper
- Sutton and Barto, Reinforcement Learning: An Introduction — Chapter 6 for the Bellman equation and temporal difference learning
- Bellman, Dynamic Programming (1957) — the mathematical foundation for the DP optimal solver
