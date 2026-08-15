---
title: Market Regime Detector Hmm
emoji: 📈
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Market Regime Detector — HMM Financial Analytics

A financial analytics platform that uses Gaussian Hidden Markov Models to identify which market regime a stock is currently in: Bull, Neutral, Bear, or Crisis. You enter a ticker symbol and a date range, and the system downloads the data, fits a statistical model, colours the price chart by regime, shows how the model transitions between states, and backtests a simple regime-switching trading strategy.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Dash10107/market-regime-detector-hmm)

---

## What problem does this solve?

Financial markets do not behave the same way all the time. Sometimes prices drift upward steadily. Sometimes they oscillate without clear direction. Sometimes they fall sharply with high volatility. A strategy that works well in one regime often fails in another.

The goal here is to detect which regime the market is currently in, without being told in advance how many regimes there are or what they look like. The Hidden Markov Model does this by finding recurring statistical patterns in price data and grouping them into hidden states.

---

## The Algorithm: Gaussian Hidden Markov Model

A Hidden Markov Model assumes that at any point in time the world is in one of a small number of hidden states. You cannot observe the state directly, only noisy measurements of it. Here, the hidden states are market regimes and the measurements are financial features computed from daily prices.

The model has three components.

**Transition probabilities:** A matrix that says how likely the market is to move from one regime to another. If the probability of staying in the Bear regime is 0.92, it means once the market enters a bear phase it tends to stay there for a while.

**Emission distributions:** For each hidden state, a Gaussian distribution over the observed features. The Bull regime might be centred on positive returns with low volatility. The Bear regime might be centred on negative returns with high volatility.

**Initial probabilities:** How likely the series is to start in each state.

**Training via the Baum-Welch algorithm:** This is an Expectation-Maximisation procedure. The E-step estimates the probability of being in each state at each time step given the current parameters. The M-step updates the parameters to maximise the likelihood of the observed data given those state probabilities. These two steps alternate until the log-likelihood converges.

**Decoding via Viterbi:** Once the model is trained, the Viterbi algorithm finds the most likely sequence of hidden states given the observations. This gives one regime label per trading day.

**Regime labelling:** The model discovers states but does not name them. We label them by sorting on mean daily return: highest return gets the Bull label, lowest gets the Bear label, and so on. If four states are fitted, the most volatile low-return state becomes Crisis.

---

## Feature Engineering

The model is fitted on four features computed from daily close prices.

**Log return:** The natural log of today's price divided by yesterday's price. This is approximately the percentage change and is better behaved statistically than raw price differences.

**Annualised volatility:** The 20-day rolling standard deviation of log returns, scaled to a yearly figure. This captures whether the market is calm or turbulent.

**5-day momentum:** The log of today's price divided by the price five days ago. This captures short-term price direction.

**RSI (14-day):** The Relative Strength Index, a momentum indicator. We normalise it from its usual 0 to 100 range to -1 to +1 for consistency with the other features.

All four features are standardised to mean zero and unit variance before fitting. This prevents features with larger magnitudes from dominating the model.

---

## BIC Model Selection

How many regimes should the model have? Too few and the model cannot distinguish meaningfully different market conditions. Too many and each regime has too few observations and the results become noisy.

The Bayesian Information Criterion penalises model complexity:

```
BIC = -2 * log_likelihood + k * log(n)
```

where k is the number of free parameters and n is the number of observations. Lower BIC is better. The app can automatically fit models with 2, 3, 4, and 5 states and select the one with the lowest BIC.

---

## The Regime-Switching Trading Strategy

Once regimes are detected, the app backtests a simple allocation strategy:

```
Bull regime     → 100% invested in the stock
Neutral regime  → 50% invested
Bear regime     → 0% (hold cash)
Crisis regime   → 0% (hold cash)
```

Strategy signals are lagged by one day to avoid look-ahead bias: you can only act on the regime label from yesterday's close. Transaction costs of 10 basis points are charged whenever the allocation changes.

This is compared against a buy-and-hold baseline over the same period. The app computes total return, annualised return, Sharpe ratio, maximum drawdown, and Calmar ratio for both.

---

## Project Structure

```
market-regime-detector-hmm/
├── app.py                  main Gradio application
├── core/
│   ├── data.py             data download, feature engineering, regime statistics
│   ├── hmm_model.py        HMM fitting, BIC selection, diagnostics
│   └── strategy.py         backtest engine and performance metrics
├── viz/
│   └── charts.py           regime price chart, analytics dashboard, backtest chart
└── requirements.txt
```

---

## Quick Setup

```bash
git clone https://github.com/yourusername/reinforcement-learning-lab
cd market-regime-detector-hmm
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

**Suggested starting point:** In the Regime Detection tab, enter SPY as the ticker, set the date range from 2020-01-01 to today, leave regimes at 3, and click Detect Regimes. The price chart will appear with coloured background bands. Then go to the Backtest tab with the same settings to see whether the regime-switching strategy would have beaten buy-and-hold over that period.

---

## What each tab shows

**Regime Detection:** Price chart with coloured background bands for each regime period, a log return bar chart coloured by regime below it, and a regime timeline strip at the bottom. A current regime probability gauge shows how confident the model is about the most recent day. A summary table shows per-regime statistics including annualised return, Sharpe ratio, and percentage of time spent in each state.

**Analytics:** Four panels showing return distribution curves per regime, a transition probability heatmap showing how likely each regime is to follow each other, a time allocation bar chart, and a BIC model selection curve.

**Backtest:** Equity curves for the regime strategy and buy-and-hold, drawdown comparison, allocation timeline showing when the strategy was invested vs holding cash, and a full metrics table covering return, volatility, Sharpe, drawdown, and Calmar ratio.

**Multi-Asset:** Enter up to six tickers comma-separated and see their regime timelines stacked vertically. This shows whether different assets are in the same regime at the same time, which matters for portfolio diversification.

**How It Works:** Full explanation of HMM, Baum-Welch, Viterbi, the feature engineering choices, BIC selection, and the backtest methodology.

---

## A note on HMM and reinforcement learning

The HMM is a statistical model, not an RL model in the strict sense. It belongs in this portfolio because it solves the state estimation problem that underlies many RL applications: inferring hidden world state from observable signals. In an RL agent operating in a financial environment you would typically feed regime labels as part of the state representation to help the policy condition on the current market phase. The HMM acts as the perception layer before the decision-making layer.

---

## Requirements

```
gradio>=6.0.0
yfinance
hmmlearn
scikit-learn
pandas
numpy
matplotlib
```

---

## Things to Try

**1. Find the COVID crash.**
Run SPY from 2020-01-01 to 2023-01-01. March 2020 should appear as a Crisis or Bear regime with very clear boundaries. Check the transition matrix for that period.

**2. Compare 2 regimes vs 4 on the same data.**
The 4-regime model carves out distinct Neutral and Crisis periods that 2 regimes lumps together. Use the BIC Analytics panel to see which the model statistically prefers.

**3. Compare a volatile stock with a stable one.**
Run TSLA and JNJ separately. TSLA will show rapid regime switching with short durations. JNJ will show longer, more persistent regimes. The transition matrices look very different.

**4. Run the backtest on a period you know well.**
If you remember the 2021 bull run or the 2022 rate hike bear market, run regime detection on that period and see if the model correctly identifies what you experienced.

**5. Use Multi-Asset with SPY, GLD, and TLT.**
During Bear regimes in equities you should see gold and bonds in Neutral or Bull regimes. That is the classic flight-to-safety relationship appearing in the regime labels.

---

## Further Reading

- Rabiner, A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition (1989) — the canonical HMM reference, highly readable despite the different domain
- Hamilton, A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle (1989) — applying HMMs to economic regimes
- hmmlearn documentation: https://hmmlearn.readthedocs.io/en/latest/
