---
title: Mab Banner Optimizer
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# MAB Banner Optimizer — Ad Campaign Intelligence with Multi-Armed Bandits

<p align="center">
  <a href="https://dash10107.github.io/reinforcement-learning-lab/en/"><img src="https://img.shields.io/badge/Course_Chapter-Read-blue?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="Course Chapter"></a>
  <a href="https://huggingface.co/spaces/Dash10107/mab-banner-optimizer"><img src="https://img.shields.io/badge/Live_Demo-Hugging_Face-yellow?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Demo"></a>
</p>

An interactive platform for understanding how advertising systems learn which banner to show. Six different bandit algorithms compete head-to-head on the same simulated ad campaign. You can watch their reward and regret curves diverge in real time, step through individual decisions in learner mode, and build custom campaign scenarios to test different conditions.

**This project is part of the [Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab) — an interactive course and lab that bridges the gap between RL theory and practical implementation.**

---

## What problem does this solve?

A website has four banner variants for a product. Each has a different design. The company does not know in advance which design gets the most clicks, and finding out takes time and lost revenue. They need a strategy that explores enough to learn which banner is best, while also showing the best-performing banner as much as possible to maximise revenue.

This is the exploration-exploitation trade-off, one of the fundamental problems in machine learning and decision theory. Multi-Armed Bandit algorithms solve it without requiring a full model of the environment, just feedback from each action taken.

The name comes from a metaphor: imagine a row of slot machines (one-armed bandits) each with a different payout probability. You do not know the probabilities. You want to maximise total winnings. How do you balance trying new machines against sticking with ones that have paid well so far?

---

## The Six Algorithms

**Epsilon-Greedy**

The simplest approach. With probability epsilon (default 10%), pick a random banner. Otherwise, pick whichever banner has the highest estimated click rate so far. This guarantees some exploration at every step, but the exploration never stops even after you have learned which banner is best. That inefficiency is why more sophisticated algorithms exist.

**Decaying Epsilon-Greedy**

The same as Epsilon-Greedy, but epsilon shrinks over time as 1 divided by the square root of the step count. Early on, when you know little, epsilon is large and you explore widely. Later, when you have accumulated evidence, epsilon is small and you mostly exploit.

**UCB1 — Upper Confidence Bound**

UCB1 never picks randomly. Instead it adds an uncertainty bonus to each banner estimate:

```
score(arm) = estimated_value + c * sqrt(log(total_pulls) / pulls_for_this_arm)
```

The bonus is large when an arm has been pulled few times and shrinks as it gets more pulls. The agent always picks the arm with the highest score. This is called optimism in the face of uncertainty. UCB1 has provably optimal regret bounds of order log(T).

**Thompson Sampling**

Thompson Sampling takes a Bayesian approach. It maintains a Beta distribution over the true click rate of each banner, parameterised by successes (alpha) and failures (beta). At each step it samples one value from each banner posterior and picks the highest. Over time posteriors narrow around the true rates. Good banners get sampled often; uncertain banners get sampled occasionally. This naturally balances exploration and exploitation without any tuning and performs very well in practice.

**Gradient Bandit**

Instead of estimating click rates, the Gradient Bandit learns a preference score for each arm. Action probabilities are computed via softmax over these preferences. After each pull the chosen arm gets a preference boost if reward was above the running baseline, and a penalty if below. This is stochastic gradient ascent on expected reward.

**EXP3 — Adversarial Bandit**

EXP3 uses importance-weighted updates designed for adversarial environments where rewards can be chosen by an opponent to fool you. In a standard stochastic setting it is not the most efficient, but it provides the strongest theoretical guarantee and is useful when the environment is non-stationary or unpredictable.

---

## Key Concepts

**Regret** is the main metric. It measures how much reward you lost by not always picking the best arm. Cumulative regret adds this up over all steps. A good algorithm has regret that grows slowly (ideally logarithmically) rather than linearly.

**CTR** means click-through rate: the probability that a user clicks on a banner when shown it. In the simulation this is a fixed probability assigned to each arm, hidden from the algorithms.

**Expected value** of an arm is CTR multiplied by revenue per conversion. Even a low-CTR arm can be worth choosing if it converts to high-value purchases.

**Non-stationarity** means the environment changes over time. In the campaign simulation, true CTRs drift slightly each step with Gaussian noise, simulating seasonality, ad fatigue, and competitor activity.

---

## The Environment

Each scenario defines a set of banner arms. Each arm has a hidden true CTR and a revenue per conversion. When an algorithm pulls arm i:

```
converted = Bernoulli(true_ctr[i])
reward    = revenue[i] if converted else 0
```

After each pull the true CTR drifts slightly, preventing CTRs from hitting zero or one and keeping the environment realistic.

---

## Project Structure

```
mab-banner-optimizer/
├── app.py                  main Gradio application
├── bandits/
│   ├── agents.py           all six algorithm implementations
│   ├── environment.py      CampaignEnvironment and BannerArm dataclass
│   └── simulator.py        run_comparison and run_averaged utilities
├── viz/
│   └── charts.py           comparison dashboard, belief charts, analytics
└── requirements.txt
```

---

## Quick Setup

```bash
git clone https://github.com/Dash10107/reinforcement-learning-lab.git
cd mab-banner-optimizer
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

**Suggested starting point:** Go to the Algorithm Face-Off tab, select Thompson Sampling, UCB1, and Epsilon-Greedy, pick E-Commerce Sale as the scenario, and click Run Face-Off. Then go to the Learner Mode tab, pick Thompson Sampling, click Initialise, and step through impressions one by one watching the Beta distributions narrow.

---

## What each tab shows

**Algorithm Face-Off:** All selected algorithms run on identical environments. Four panels show cumulative revenue, cumulative regret, arm pull distribution per algorithm, and a final summary table with the winner marked.

**Campaign Analytics:** Single algorithm deep dive with per-banner statistics, rolling reward chart, and arm selection timeline showing how focus shifted across banners over the campaign.

**Learner Mode:** Step through decisions one impression at a time. For Thompson Sampling, Beta PDF curves update after each vote showing how the posterior narrows. For other algorithms, Q-value bars show current estimates.

**Scenario Lab:** Preview the expected value of each banner before running a campaign. Build custom scenarios by entering CTRs and revenues as JSON lists.

**How It Works:** Explanation of each algorithm, the exploration-exploitation dilemma, and how to interpret each chart.

---

## Requirements

```
gradio>=6.0.0
numpy
matplotlib
scipy
```

---

## Things to Try

**1. Race Epsilon-Greedy against Thompson Sampling.**
Use E-Commerce with 5000 impressions. The cumulative regret chart will show Thompson Sampling pulling ahead progressively as it narrows posteriors faster.

**2. Set epsilon to 0.0 and observe linear regret.**
In Advanced Settings set epsilon to 0. Now the agent never explores — it commits to whichever banner looked best in the first impressions. Regret grows linearly. This is the cost of pure exploitation.

**3. Step through the SaaS scenario in Learner Mode.**
The three banners have very different CTR-revenue combinations. Step through 20 impressions manually and watch the Beta distributions narrow. The Buy Now banner takes many more impressions because conversions are rare.

**4. Enable drift and compare UCB1 vs Thompson Sampling.**
Set drift to 0.008. With non-stationary CTRs Thompson Sampling adapts better because its posteriors naturally follow the shifting true rates.

**5. Build a scenario with one dominant banner.**
Set CTRs to [0.01, 0.01, 0.01, 0.30] with equal revenue. Count how many impressions each algorithm wastes on the three bad banners before committing — that waste is exactly the regret.

---

## Further Reading

- Lattimore and Szepesvari, Bandit Algorithms (2020) — free online textbook covering UCB, Thompson Sampling, and EXP3 with full proofs
- Russo et al., A Tutorial on Thompson Sampling (2018) — accessible introduction to Bayesian bandits
- Auer et al., Finite-time Analysis of the Multiarmed Bandit Problem (2002) — the UCB1 paper with the log(T) regret bound
