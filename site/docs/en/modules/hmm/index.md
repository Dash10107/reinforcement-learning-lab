# When the World Has Moods

Markets don't move randomly.

They trend upward for months, then crash suddenly, then recover slowly, then go sideways for a year. Each of these periods has a distinct character. Traders talk about "bull markets" and "bear markets" as if the market has a personality — a mood it's in.

They're not wrong.

What if we could detect which mood the market is in — not by guessing, but by systematically inferring it from the data we can observe?

That's exactly what a **Hidden Markov Model (HMM)** does.

---

## Hidden states, observable signals

The key idea of an HMM is the distinction between two kinds of things:

**Hidden states** — things you can't observe directly. The "true regime" of the market: is it in a trending, volatile, or quiet phase?

**Observable signals** — things you can measure. Daily returns, trading volume, volatility.

The model says: there are a small number of hidden states. At each time step, the system is in one of those states. The state you're in determines what signals you're likely to observe. And the state changes over time, according to fixed transition probabilities.

```
Hidden:   Bull → Bull → Bull → Crash → Recovery → Recovery
              ↓       ↓      ↓      ↓         ↓          ↓
Observable: +2%  +1.5%  +3%  -8%     +1%      +2%
```

You can't see the hidden state directly. But you can infer it from the observable signals — and that inference is what the HMM provides.

---

## The three questions an HMM answers

**1. Evaluation:** Given a sequence of observations, how likely is this sequence under the model? This lets you compare different models: "which regime model best explains what we saw last month?"

**2. Decoding (Viterbi):** Given a sequence of observations, what's the most likely sequence of hidden states? This is regime detection — "we're probably in the crash regime right now."

**3. Learning (Baum-Welch):** Given a sequence of observations, what are the model parameters — the transition probabilities, the emission distributions — that best explain the data?

```python
from hmmlearn import hmm
import numpy as np

# Fit a 3-regime HMM to market returns
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
model.fit(returns.reshape(-1, 1))

# Decode: infer the most likely regime at each time step
hidden_states = model.predict(returns.reshape(-1, 1))
```

The output `hidden_states` is a sequence of integers (0, 1, 2) — one per time step — telling you which regime the model thinks the market was in at each point.

---

## Why HMM is in this course

HMM is not technically a reinforcement learning algorithm. There's no agent, no action, no reward signal.

But it belongs here for two reasons.

First, it's an important technique in the RL ecosystem. Many real-world RL problems have non-stationary environments — environments that switch between "modes." A trading agent, an energy grid controller, a robot in changing terrain. HMMs can detect which mode the environment is in, giving the RL agent better context for its decisions.

Second, it teaches a deeply RL-adjacent idea: **learning from sequences of observations**. The Baum-Welch algorithm (which learns HMM parameters) is a form of expectation-maximisation — you infer hidden states, use them to update parameters, infer again, repeat. It's iterative, it's data-driven, and it converges to a solution. Sound familiar?

---

## What you'll notice in the demo

Open the [Market Regime Detector ↗](https://huggingface.co/spaces/Dash10107/market-regime-detector-hmm) — fitting an HMM to historical market data and visualising regime detection.

**Three things to watch:**

1. **Regime stability.** The HMM doesn't flip between regimes on every day — transitions have probabilities. A high self-transition probability means the model expects regimes to persist. Watch how the model captures sustained bull runs and sustained crashes.

2. **The three regimes.** After fitting, look at the Gaussian parameters for each hidden state. One will have positive mean (bull), one negative (bear), one near-zero with high variance (volatile/sideways). The model discovers these without labels.

3. **Regime transitions around known events.** Run the model on data spanning the 2020 COVID crash. Watch the regime detection shift sharply in March 2020, then back again by August. The HMM picks up the crash and recovery without any knowledge of what happened.

---

## Try it yourself

**Experiment 1 — Two regimes vs three.**
Fit a 2-state HMM and a 3-state HMM to the same data. Compare how well they explain the data (log-likelihood). In most market data, 3 states fit significantly better than 2 — there's a meaningful "sideways" regime that's distinct from bull and bear.

**Experiment 2 — Short vs long training window.**
Fit the model on 1 year of data, then 5 years. The transition probabilities change — a 5-year window includes more market cycles and produces more stable regime estimates.

**Experiment 3 — Use regime as RL context.**
Use the HMM's real-time regime estimate as an input feature for a DQN trading agent. Does the agent perform better with regime context than without? In most experiments: yes. Knowing which mood the market is in is a useful prior for any action-selection algorithm.
