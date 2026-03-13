"""
Market Regime Detector — HMM Analytics Platform
End-to-end financial market regime detection using Gaussian Hidden Markov Models.
Identifies Bull / Neutral / Bear / Crisis regimes with full analytics,
backtesting, and multi-asset comparison.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from datetime import date
import numpy as np
import pandas as pd
import gradio as gr

from core.data import (
    fetch_ohlcv, build_features, label_regimes,
    regime_statistics, transition_matrix,
)
from core.hmm_model import (
    fit_hmm, select_n_regimes, current_regime_probs, model_diagnostics,
)
from core.strategy import backtest_regime_strategy, strategy_metrics
from viz.charts import (
    regime_price_chart, regime_analytics, backtest_chart,
    multi_asset_chart, regime_gauge, empty_fig,
)

# ── Shared pipeline ───────────────────────────────────────────────────────────

def _run_pipeline(
    ticker: str,
    start: str,
    end: str,
    n_regimes: int,
    auto_select: bool = False,
) -> tuple[pd.DataFrame, dict, list, dict]:
    """
    Download → features → HMM → labelling.
    Returns (feat_with_labels, diagnostics, bic_scores, label_map)
    """
    raw  = fetch_ohlcv(ticker, start, end)
    feat = build_features(raw)

    if len(feat) < n_regimes * 15:
        raise ValueError(
            f"Only {len(feat)} data points — need at least {n_regimes*15} "
            f"for {n_regimes} regimes. Widen the date range."
        )

    bic_scores = []
    if auto_select:
        best_n, bic_scores = select_n_regimes(feat)
        n_regimes = best_n

    model, scaler, regimes = fit_hmm(feat, n_regimes)
    feat["regime"] = regimes
    feat, label_map = label_regimes(feat)

    diag = model_diagnostics(model, scaler, feat, n_regimes)

    # Attach current regime probs to feat for later use
    feat._hmm_model  = model
    feat._hmm_scaler = scaler
    feat._label_map  = label_map

    return feat, diag, bic_scores, label_map


def _parse_dates(start_val, end_val) -> tuple[str, str]:
    def _fmt(v) -> str:
        if isinstance(v, str):
            return v[:10]
        if isinstance(v, (int, float)):
            unit = "s" if abs(v) < 1e10 else "ms"
            return pd.to_datetime(v, unit=unit).strftime("%Y-%m-%d")
        return str(v)[:10]

    s = _fmt(start_val)
    e = _fmt(end_val)
    if e > date.today().isoformat():
        e = date.today().isoformat()
    return s, e


# ── Tab callbacks ─────────────────────────────────────────────────────────────

def cb_regime_detection(
    ticker, start_val, end_val, n_regimes, auto_select,
    progress: gr.Progress = gr.Progress(),
):
    try:
        ticker = ticker.strip().upper()
        start, end = _parse_dates(start_val, end_val)
        progress(0.15, desc="Downloading market data…")

        feat, diag, bic_scores, label_map = _run_pipeline(
            ticker, start, end, int(n_regimes), bool(auto_select)
        )
        progress(0.60, desc="Rendering regime chart…")

        price_fig = regime_price_chart(feat, ticker)

        progress(0.80, desc="Computing current regime…")
        model  = feat._hmm_model
        scaler = feat._hmm_scaler
        probs  = current_regime_probs(model, scaler, feat)
        gauge  = regime_gauge(probs, label_map)

        # Current regime
        current = feat["regime_label"].iloc[-1]
        current_col = feat["regime_color"].iloc[-1]

        # Build stats table markdown
        stats_df = regime_statistics(feat)
        rows = "\n".join(
            f"| {r['Regime']} | {r['Ann. Return']} | {r['Ann. Volatility']} "
            f"| {r['Sharpe Ratio']} | {r['Max Drawdown']} | {r['% of Time']} | {r['Avg Duration']} |"
            for _, r in stats_df.iterrows()
        )
        status_md = f"""
### Analysis Complete — {ticker}  `{start}` → `{end}`

**Current Regime:** {current}  ·  **Data Points:** {len(feat)}  ·  **Regimes:** {int(n_regimes) if not auto_select else diag['n_states']}

| Regime | Ann. Return | Ann. Vol | Sharpe | Max DD | % Time | Avg Duration |
|---|---|---|---|---|---|---|
{rows}

> **Model:** Log-likelihood `{diag['log_likelihood']}` · AIC `{diag['aic']}` · BIC `{diag['bic']}` · Converged: `{diag['converged']}`
"""
        progress(1.0)
        return price_fig, gauge, status_md

    except Exception as e:
        return empty_fig(f"Error: {e}"), empty_fig(""), f"❌ **Error:** {e}"


def cb_analytics(
    ticker, start_val, end_val, n_regimes, auto_select,
    progress: gr.Progress = gr.Progress(),
):
    try:
        ticker = ticker.strip().upper()
        start, end = _parse_dates(start_val, end_val)
        progress(0.1, desc="Fitting model…")

        feat, diag, bic_scores, label_map = _run_pipeline(
            ticker, start, end, int(n_regimes), bool(auto_select)
        )
        progress(0.6, desc="Computing analytics…")

        stats_df = regime_statistics(feat)
        n = int(diag["n_states"])
        trans_df = transition_matrix(feat, n, label_map)
        fig = regime_analytics(feat, stats_df, trans_df, diag, bic_scores)

        progress(1.0)
        return fig, f"Analytics computed for **{ticker}** — {len(feat)} observations, {n} regimes."

    except Exception as e:
        return empty_fig(f"Error: {e}"), f"❌ **Error:** {e}"


def cb_backtest(
    ticker, start_val, end_val, n_regimes, auto_select,
    progress: gr.Progress = gr.Progress(),
):
    try:
        ticker = ticker.strip().upper()
        start, end = _parse_dates(start_val, end_val)
        progress(0.1, desc="Fitting model…")

        feat, diag, _, label_map = _run_pipeline(
            ticker, start, end, int(n_regimes), bool(auto_select)
        )
        progress(0.5, desc="Running backtest…")

        bt      = backtest_regime_strategy(feat)
        metrics = strategy_metrics(bt)
        fig     = backtest_chart(bt, metrics, ticker)

        strat_m = metrics["Strategy"]
        bnh_m   = metrics["Buy & Hold"]
        summary = f"""
### Backtest Results — {ticker}

**Regime Strategy:** Invest 100% (🐂 Bull), 50% (⚖️ Neutral), 0% (🐻 Bear / 💥 Crisis)

| Metric | Regime Strategy | Buy & Hold |
|---|---|---|
| **Total Return** | `{strat_m['Total Return']}` | `{bnh_m['Total Return']}` |
| **Ann. Return** | `{strat_m['Ann. Return']}` | `{bnh_m['Ann. Return']}` |
| **Ann. Volatility** | `{strat_m['Ann. Volatility']}` | `{bnh_m['Ann. Volatility']}` |
| **Sharpe Ratio** | `{strat_m['Sharpe Ratio']}` | `{bnh_m['Sharpe Ratio']}` |
| **Max Drawdown** | `{strat_m['Max Drawdown']}` | `{bnh_m['Max Drawdown']}` |
| **Calmar Ratio** | `{strat_m['Calmar Ratio']}` | `{bnh_m['Calmar Ratio']}` |

> **Note:** All signals lagged by 1 day (no look-ahead bias).
> Transaction cost: 10bps per allocation change.
"""
        progress(1.0)
        return fig, summary

    except Exception as e:
        return empty_fig(f"Error: {e}"), f"❌ **Error:** {e}"


def cb_multi_asset(
    tickers_str, start_val, end_val, n_regimes,
    progress: gr.Progress = gr.Progress(),
):
    try:
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()][:6]
        start, end = _parse_dates(start_val, end_val)
        results = {}
        for i, t in enumerate(tickers):
            progress((i + 0.5) / len(tickers), desc=f"Analysing {t}…")
            try:
                feat, _, _, _ = _run_pipeline(t, start, end, int(n_regimes))
                results[t] = feat
            except Exception as e:
                pass  # skip failed tickers silently

        if not results:
            return empty_fig("No valid tickers found."), "❌ All tickers failed."

        fig = multi_asset_chart(results)
        ok_tickers = ", ".join(results.keys())
        progress(1.0)
        return fig, f"Regime sync chart for: **{ok_tickers}**"

    except Exception as e:
        return empty_fig(f"Error: {e}"), f"❌ **Error:** {e}"


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0d14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

.mrd-header {
    border-bottom: 1px solid #1e2a3d;
    padding: 1.4rem 1.5rem 0.9rem;
    background: linear-gradient(180deg, #0a0d14 0%, #0f1520 100%);
}
.mrd-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.2rem, 3vw, 1.9rem);
    font-weight: 700; color: #00d4aa; margin: 0; letter-spacing: -0.01em;
}
.mrd-sub { color: #4a6080; font-size: 0.85rem; margin-top: 0.3rem; }
.mrd-badges { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.7rem; }
.mrd-badge {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    letter-spacing:0.08em; padding:3px 10px; border-radius:3px;
    text-transform:uppercase;
}
.b-teal   { background:#061a16; color:#00d4aa; border:1px solid #0a4a3c; }
.b-amber  { background:#1e1206; color:#faad14; border:1px solid #5c3a06; }
.b-red    { background:#1a0808; color:#ef4444; border:1px solid #5c1a1a; }
.b-purple { background:#130a1e; color:#7c3aed; border:1px solid #3a1a6e; }

.tab-nav { border-bottom:1px solid #1e2a3d !important; background:transparent !important; }
.tab-nav button {
    font-family:'JetBrains Mono',monospace !important; font-size:0.72rem !important;
    letter-spacing:0.05em !important; color:#4a6080 !important;
    background:transparent !important; border:none !important;
    padding:0.7rem 1.1rem !important; text-transform:uppercase !important;
}
.tab-nav button.selected { color:#00d4aa !important; border-bottom:2px solid #00d4aa !important; }

button.primary {
    font-family:'JetBrains Mono',monospace !important; font-weight:600 !important;
    background:linear-gradient(135deg,#063028,#084a40) !important;
    color:#00d4aa !important; border:1px solid #00d4aa !important;
    border-radius:5px !important; transition:all 0.2s !important;
}
button.primary:hover { box-shadow:0 0 14px rgba(0,212,170,0.3) !important; }
button.secondary {
    font-family:'JetBrains Mono',monospace !important;
    background:#0f1520 !important; color:#faad14 !important;
    border:1px solid #faad14 !important; border-radius:5px !important;
}

label span, .gradio-container label {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.7rem !important; color:#4a6080 !important;
    text-transform:uppercase !important; letter-spacing:0.06em !important;
}
input[type=range] { -webkit-appearance:none; height:3px;
                    background:#1e2a3d; border-radius:2px; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:14px; height:14px;
    border-radius:50%; background:#00d4aa; cursor:pointer;
    border:2px solid #0a0d14;
}
textarea, .gradio-container textarea {
    font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
    background:#060a10 !important; color:#00d4aa !important;
    border:1px solid #1e2a3d !important; border-radius:4px !important;
}
.gradio-container h2, .gradio-container h3 {
    color:#00d4aa !important; font-family:'JetBrains Mono',monospace !important;
}
.gradio-container p  { color:#64748b !important; }
table { width:100%; border-collapse:collapse; }
th { background:#0f1520; color:#00d4aa; font-family:'JetBrains Mono',monospace;
     font-size:0.7rem; text-align:left; padding:7px 12px;
     border-bottom:1px solid #1e2a3d; text-transform:uppercase; }
td { padding:7px 12px; border-bottom:1px solid #0a0d14;
     color:#e2e8f0; font-size:0.83rem; }
code { font-family:'JetBrains Mono',monospace; background:#0f1520;
       color:#faad14; padding:1px 5px; border-radius:3px; }
blockquote { border-left:3px solid #00d4aa; padding:0.6rem 1rem;
             background:#0f1520; border-radius:0 4px 4px 0; margin:0.5rem 0; }
strong { color:#e2e8f0 !important; }
footer { display:none !important; }
.gradio-container .block { background:transparent !important; border:none !important; }
"""

# ── Shared input panel ────────────────────────────────────────────────────────

def _input_panel(default_ticker="SPY", key_suffix=""):
    ticker = gr.Textbox(value=default_ticker, label="Ticker Symbol",
                        info="e.g. SPY, AAPL, ^GSPC, MSFT, BTC-USD")
    start  = gr.Textbox(value="2020-01-01", label="Start Date  (YYYY-MM-DD)")
    end    = gr.Textbox(value=str(date.today()), label="End Date  (YYYY-MM-DD)")
    n_reg  = gr.Slider(2, 5, value=3, step=1, label="Number of Regimes")
    auto   = gr.Checkbox(label="Auto-select (BIC)", value=False,
                         info="Automatically find optimal number of regimes")
    return ticker, start, end, n_reg, auto


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Market Regime Detector — HMM") as demo:

    gr.HTML("""
    <div class="mrd-header">
        <div class="mrd-title">📈 MARKET REGIME DETECTOR</div>
        <div class="mrd-sub">
            Gaussian Hidden Markov Model · Market State Identification ·
            Regime Analytics · Strategy Backtesting · Multi-Asset Synchronisation
        </div>
        <div class="mrd-badges">
            <span class="mrd-badge b-teal">● Gaussian HMM</span>
            <span class="mrd-badge b-amber">● 4-Feature Engineering</span>
            <span class="mrd-badge b-red">● Strategy Backtest</span>
            <span class="mrd-badge b-purple">● BIC Model Selection</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Regime Detection
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📈 REGIME DETECTION"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    PRICE CHART WITH REGIME BANDS
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Coloured background bands show market regime at each point in time.
                    Log returns coloured by regime. Regime timeline bar at bottom.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    t1, s1, e1, n1, a1 = _input_panel("SPY")
                    btn1 = gr.Button("🔍 DETECT REGIMES", variant="primary")

                    gr.HTML("""
                    <div style="background:#0f1520;border:1px solid #1e2a3d;border-radius:6px;
                                padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                    color:#4a6080;text-transform:uppercase;margin-bottom:0.5rem;">
                            REGIME LEGEND
                        </div>
                        <div style="font-size:0.82rem;line-height:1.9;">
                            <div><span style="color:#00d4aa">■</span> 🐂 Bull — positive return, low vol</div>
                            <div><span style="color:#faad14">■</span> ⚖️ Neutral — mixed signals</div>
                            <div><span style="color:#ef4444">■</span> 🐻 Bear — negative return</div>
                            <div><span style="color:#7c3aed">■</span> 💥 Crisis — extreme volatility</div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=2):
                    d1_status = gr.Markdown("*Configure parameters and click Detect Regimes.*")
                    d1_gauge  = gr.Plot(label="Current Regime Probability")

            d1_chart = gr.Plot(label="Regime Price Chart")

            btn1.click(cb_regime_detection, [t1, s1, e1, n1, a1],
                       [d1_chart, d1_gauge, d1_status])

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Regime Analytics
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📊 ANALYTICS"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    DEEP-DIVE REGIME ANALYTICS
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Return distributions · Transition matrix ·
                    Time allocation · BIC model selection curve
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    t2, s2, e2, n2, a2 = _input_panel("SPY")
                    btn2 = gr.Button("📊 RUN ANALYTICS", variant="primary")
                with gr.Column(scale=2):
                    d2_status = gr.Markdown("*Run analytics to see charts.*")

            d2_chart = gr.Plot(label="Analytics Dashboard")
            btn2.click(cb_analytics, [t2, s2, e2, n2, a2], [d2_chart, d2_status])

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Strategy Backtest
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("⚖️ BACKTEST"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    REGIME-SWITCHING STRATEGY vs BUY & HOLD
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Allocates 100% in Bull, 50% in Neutral, 0% in Bear/Crisis.
                    All signals lagged 1 day · 10bps transaction cost.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    t3, s3, e3, n3, a3 = _input_panel("SPY", "3")
                    btn3 = gr.Button("⚖️ RUN BACKTEST", variant="primary")
                with gr.Column(scale=2):
                    d3_summary = gr.Markdown("*Run backtest to see performance metrics.*")

            d3_chart = gr.Plot(label="Strategy Performance")
            btn3.click(cb_backtest, [t3, s3, e3, n3, a3], [d3_chart, d3_summary])

        # ══════════════════════════════════════════════════════════════════
        # Tab 4 — Multi-Asset
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🌐 MULTI-ASSET"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    CROSS-ASSET REGIME SYNCHRONISATION
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Compare regime timelines across multiple assets.
                    Are equities and bonds in the same regime? Does crypto lead equities?
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    ma_tickers = gr.Textbox(
                        value="SPY, QQQ, GLD, TLT",
                        label="Tickers (comma-separated, max 6)",
                    )
                    ma_start = gr.Textbox(value="2020-01-01", label="Start Date")
                    ma_end   = gr.Textbox(value=str(date.today()), label="End Date")
                    ma_n     = gr.Slider(2, 4, value=3, step=1, label="Regimes per Asset")
                    btn4     = gr.Button("🌐 COMPARE ASSETS", variant="primary")

                    gr.HTML("""
                    <div style="background:#0f1520;border:1px solid #1e2a3d;
                                border-radius:6px;padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                    color:#4a6080;text-transform:uppercase;margin-bottom:0.4rem;">
                            SUGGESTED COMBINATIONS
                        </div>
                        <div style="font-size:0.78rem;color:#64748b;line-height:1.9;">
                            <div><code>SPY, QQQ, GLD, TLT</code> — US macro</div>
                            <div><code>SPY, EEM, DX-Y.NYB</code> — Global risk</div>
                            <div><code>AAPL, MSFT, AMZN, GOOGL</code> — Big tech</div>
                            <div><code>BTC-USD, ETH-USD, SPY</code> — Crypto vs equity</div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=2):
                    d4_status = gr.Markdown("*Enter tickers and click Compare.*")

            d4_chart = gr.Plot(label="Cross-Asset Regime Timeline")
            btn4.click(cb_multi_asset, [ma_tickers, ma_start, ma_end, ma_n],
                       [d4_chart, d4_status])

        # ══════════════════════════════════════════════════════════════════
        # Tab 5 — How It Works
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📚 HOW IT WORKS"):
            gr.Markdown("""
## Hidden Markov Models for Market Regimes

Financial markets don't behave uniformly. They cycle through distinct **regimes** —
periods of sustained trending, calm, or turbulence. HMMs capture this by modelling
the market as a system with a small number of hidden states, where each state has
a characteristic statistical signature.

---

## The Model

**Gaussian HMM** assumes:
1. At each time step, the market is in one of N hidden states (regimes)
2. The observed features are drawn from a Gaussian distribution specific to that state
3. Regime transitions follow a Markov chain (next regime depends only on current)

```
State sequence:   S₁ → S₂ → ... → Sₜ     (hidden, to be inferred)
Observations:     O₁   O₂   ...   Oₜ     (features we compute)
Each Oₜ | Sₜ ~ N(μ_s, Σ_s)               (Gaussian per state)
```

**Fitting (EM / Baum-Welch algorithm):**
Iteratively refines state means, covariances, and transition probabilities
to maximise the likelihood of the observed feature sequence.

**Inference (Viterbi algorithm):**
Finds the most likely sequence of hidden states given the observations.

---

## Feature Engineering (4 Features)

| Feature | Formula | What it captures |
|---|---|---|
| `log_ret` | `log(Pₜ / Pₜ₋₁)` | Daily return sign and magnitude |
| `volatility` | `20d rolling std × √252` | Annualised price fluctuation |
| `momentum_5d` | `log(Pₜ / Pₜ₋₅)` | Short-term price direction |
| `rsi` | `RSI(14)` normalised [-1,1] | Overbought / oversold condition |

All features are standardised (zero mean, unit variance) before fitting.

---

## Regime Labelling

Regimes are sorted by **mean daily return**:
- Highest return → 🐂 **Bull** (positive drift, lower volatility)
- Middle → ⚖️ **Neutral** (mixed, range-bound)
- Lowest return → 🐻 **Bear** (negative drift)
- If 4 states: highest volatility among lower-return states → 💥 **Crisis**

---

## BIC Model Selection

The **Bayesian Information Criterion** penalises model complexity:
```
BIC = -2 × log_likelihood + k × log(n)
```
Lower BIC = better trade-off between fit and complexity.
The auto-select option fits models for n=2,3,4,5 and picks the lowest BIC.

---

## Backtest Strategy Logic

```
Regime    → Allocation
🐂 Bull   → 100% long
⚖️ Neutral → 50% long
🐻 Bear   → 0% (cash)
💥 Crisis → 0% (cash)
```

**No look-ahead bias:** regime labels at day T are only used for trading at day T+1.
**Transaction cost:** 10 basis points per allocation change.

---

## Transition Matrix

Shows the probability of staying in or switching regimes:
- High diagonal = regimes are persistent (sticky)
- High off-diagonal = frequent regime changes

In most equities, Bull regimes are the most persistent,
and Bear/Crisis regimes have high self-transition probability once entered.

---

## Limitations

- HMM assumes stationarity of observations given the regime (not always true)
- Regime labels are unsupervised — "Bull" and "Bear" are our interpretation
- Past regime statistics don't guarantee future regime behaviour
- The model has no knowledge of fundamentals, news, or macro events
""")

    gr.HTML("""
    <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                color:#1e2a3d;padding:1.5rem 0 0.5rem;border-top:1px solid #0f1520;
                letter-spacing:0.1em;text-transform:uppercase;">
        Gaussian HMM · hmmlearn · yfinance · Regime Analytics · Gradio
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
