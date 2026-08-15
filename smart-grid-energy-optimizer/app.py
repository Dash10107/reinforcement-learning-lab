"""
Smart Grid Energy Optimizer
Professional RL-based Battery Energy Storage System (BESS) management platform.

Compares three strategies on the same grid scenario:
  • DQN (Deep Q-Network) — learned policy, no price foresight
  • DP Optimal           — dynamic programming, perfect foresight (upper bound)
  • Rule-Based           — price-threshold heuristic (industry baseline)
"""

from __future__ import annotations

import os

import gradio as gr
import numpy as np
from agents.dp_solver import rollout_dp, solve_dp
from agents.dqn_agent import TrainingState, rollout_dqn, start_training
from agents.naive import rollout_naive, rollout_no_battery
from env.grid_env import SmartGridEnv
from env.scenarios import (
    BASE_PRICES,
    LOAD_PROFILES,
    SCENARIO_PRESETS,
    SOLAR_PROFILE,
    add_price_noise,
    add_solar_noise,
    get_scenario,
)
from viz.dashboard import (
    _empty_fig,
    comparison_chart,
    dispatch_dashboard,
    scenario_price_chart,
    training_chart,
)

DQN_PATH = "dqn_smartgrid.zip"
_train_state = TrainingState()
_last_scenario: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_env(scenario: dict, stochastic: bool = False) -> SmartGridEnv:
    return SmartGridEnv(
        battery_capacity_kwh=scenario.get("battery_cap", 10.0),
        max_charge_rate_kw=scenario.get("max_rate", 3.0),
        efficiency=scenario.get("efficiency", 0.92),
        solar_capacity_kw=scenario.get("solar_cap", 5.0),
        building_load=scenario.get("load_profile", "commercial"),
        price_profile=scenario.get("price_profile", "summer_peak"),
        stochastic=stochastic,
        seed=42,
    )


def _run_all_strategies(scenario: dict) -> dict:
    """Run DP, naive, no-battery on the same deterministic scenario."""
    results = {}

    # DP Optimal
    env_dp = _make_env(scenario)
    env_dp._prices = scenario["prices"]
    env_dp._solar = scenario["solar"]
    env_dp._load = scenario["load"]
    policy, _, soc_grid = solve_dp(env_dp)
    rollout_dp(env_dp, policy, soc_grid)
    results["DP Optimal"] = {
        "log": {k: list(v) for k, v in env_dp.log.items()},
        "reward": env_dp.total_reward,
    }

    # Rule-Based
    env_nb = _make_env(scenario)
    env_nb._prices = scenario["prices"]
    env_nb._solar = scenario["solar"]
    env_nb._load = scenario["load"]
    rollout_naive(env_nb)
    results["Rule-Based"] = {
        "log": {k: list(v) for k, v in env_nb.log.items()},
        "reward": env_nb.total_reward,
    }

    # No Storage
    env_ns = _make_env(scenario)
    env_ns._prices = scenario["prices"]
    env_ns._solar = scenario["solar"]
    env_ns._load = scenario["load"]
    rollout_no_battery(env_ns)
    results["No Storage"] = {
        "log": {k: list(v) for k, v in env_ns.log.items()},
        "reward": env_ns.total_reward,
    }

    # DQN (if available)
    if os.path.exists(DQN_PATH) or os.path.exists(DQN_PATH + ".zip"):
        env_dqn = _make_env(scenario, stochastic=False)
        env_dqn._prices = scenario["prices"]
        env_dqn._solar = scenario["solar"]
        env_dqn._load = scenario["load"]
        r = rollout_dqn(env_dqn, DQN_PATH)
        if r is not None:
            results["DQN Agent"] = {
                "log": {k: list(v) for k, v in env_dqn.log.items()},
                "reward": r,
            }

    return results


# ── Callbacks ─────────────────────────────────────────────────────────────────


def cb_load_scenario(scenario_name: str):
    global _last_scenario
    _last_scenario = get_scenario(scenario_name)
    fig = scenario_price_chart(
        _last_scenario["prices"],
        _last_scenario["solar"],
        _last_scenario["load"],
        scenario_name,
    )
    return fig, f"Scenario **{scenario_name}** loaded."


def cb_run_dispatch(
    scenario_name: str,
    strategy: str,
    progress: gr.Progress = gr.Progress(),
):
    global _last_scenario
    progress(0.1, desc="Loading scenario…")
    _last_scenario = get_scenario(scenario_name)
    sc = _last_scenario

    progress(0.3, desc=f"Running {strategy}…")
    env = _make_env(sc)
    env._prices = sc["prices"]
    env._solar = sc["solar"]
    env._load = sc["load"]

    if strategy == "DP Optimal":
        policy, _, soc_grid = solve_dp(env)
        rollout_dp(env, policy, soc_grid)
    elif strategy == "DQN Agent":
        r = rollout_dqn(env, DQN_PATH)
        if r is None:
            return _empty_fig(
                "Train the DQN agent first in the Training tab."
            ), "*No DQN model found — train first.*"
    elif strategy == "Rule-Based":
        rollout_naive(env)
    else:
        rollout_no_battery(env)

    progress(0.8, desc="Rendering dashboard…")
    fig = dispatch_dashboard(
        {k: list(v) for k, v in env.log.items()},
        scenario_name=scenario_name,
        strategy=strategy,
    )
    total_r = env.total_reward
    solar_total = sum(env.log["solar"])
    load_total = sum(env.log["load"])
    self_suff = min(100, solar_total / max(load_total, 1e-9) * 100)

    status = f"""
**Strategy:** {strategy} · **Scenario:** {scenario_name}

| KPI | Value |
|---|---|
| **Daily Net Revenue** | `${total_r:+.4f}` |
| **Solar Self-Sufficiency** | `{self_suff:.1f}%` |
| **Total Solar Generated** | `{solar_total:.2f} kWh` |
| **Total Building Load** | `{load_total:.2f} kWh` |
"""
    progress(1.0)
    return fig, status


def cb_benchmark(scenario_name: str, progress: gr.Progress = gr.Progress()):
    global _last_scenario
    progress(0.1, desc="Preparing scenario…")
    _last_scenario = get_scenario(scenario_name)
    progress(0.3, desc="Running all strategies…")
    results = _run_all_strategies(_last_scenario)
    progress(0.85, desc="Building comparison chart…")
    fig = comparison_chart(results)

    rows = "\n".join(
        f"| {name} | `${data['reward']:+.4f}` | "
        f"`{'↑ upper bound' if name == 'DP Optimal' else ('✅ trained RL' if name == 'DQN Agent' else '')}` |"
        for name, data in results.items()
    )
    summary = f"""
### Benchmark: {scenario_name}

| Strategy | Daily Revenue | Notes |
|---|---|---|
{rows}

> **DP Optimal** has perfect price foresight — it's the theoretical upper bound.
> **DQN Agent** learns without foresight — closer to DP = better learned policy.
> **Gap (DQN vs DP)** measures how much RL has left to improve.
"""
    progress(1.0)
    return fig, summary


def cb_start_training(scenario_name: str, timesteps: int, lr: float, batch: int):
    global _train_state
    if _train_state.running:
        return "⚠️ Training already running. Stop it first."
    _train_state = TrainingState()
    sc_cfg = list(SCENARIO_PRESETS.values())[
        list(SCENARIO_PRESETS.keys()).index(scenario_name)
    ]
    env_kwargs = {
        "battery_capacity_kwh": 10.0,
        "max_charge_rate_kw": 3.0,
        "efficiency": 0.92,
        "solar_capacity_kw": sc_cfg["solar_cap"],
        "building_load": sc_cfg["load"],
        "price_profile": sc_cfg["price"],
    }
    start_training(
        env_kwargs=env_kwargs,
        total_timesteps=int(timesteps),
        learning_rate=float(lr),
        batch_size=int(batch),
        state=_train_state,
        save_path=DQN_PATH,
    )
    return f"✅ Training started — {int(timesteps):,} timesteps on **{scenario_name}**."


def cb_stop_training():
    _train_state.running = False
    return "⏹ Stop requested."


def cb_refresh_training():
    fig = training_chart(_train_state)
    return fig, _train_state.status


def cb_custom_scenario(
    price_profile: str,
    solar_cap: float,
    load_profile: str,
    battery_cap: float,
    max_rate: float,
    efficiency: float,
    add_noise: bool,
):
    global _last_scenario
    prices = BASE_PRICES[price_profile].copy()
    solar = SOLAR_PROFILE * solar_cap
    load = LOAD_PROFILES[load_profile].copy()

    if add_noise:
        rng = np.random.default_rng(99)
        prices = add_price_noise(prices, rng=rng)
        solar = add_solar_noise(solar, rng=rng)

    _last_scenario = {
        "prices": prices,
        "solar": solar,
        "load": load,
        "solar_cap": solar_cap,
        "battery_cap": battery_cap,
        "max_rate": max_rate,
        "efficiency": efficiency,
        "load_profile": load_profile,
        "price_profile": price_profile,
    }

    fig = scenario_price_chart(prices, solar, load, "Custom Scenario")
    return fig, "Custom scenario ready. Run Dispatch or Benchmark from the other tabs."


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0f1e !important;
    color: #e8eaf0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* Header */
.sg-header {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1526 50%, #0a1428 100%);
    border-bottom: 1px solid #1a2744;
    padding: 1.5rem 1.5rem 1rem;
    margin-bottom: 0;
}
.sg-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.3rem, 3vw, 1.9rem);
    font-weight: 700;
    color: #00d4aa;
    letter-spacing: -0.01em;
    margin: 0;
}
.sg-subtitle { color: #4a6080; font-size: 0.85rem; margin-top: 0.3rem; }
.sg-badges {
    display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.8rem;
}
.sg-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.08em;
    padding: 3px 10px; border-radius: 3px; text-transform: uppercase;
}
.badge-green  { background:#0a2018; color:#52c41a; border:1px solid #1a5c2e; }
.badge-blue   { background:#0a1a30; color:#1890ff; border:1px solid #1a3a6e; }
.badge-teal   { background:#081e1c; color:#00d4aa; border:1px solid #0a4a44; }
.badge-orange { background:#201208; color:#fa8c16; border:1px solid #5c2a0a; }

/* Tabs */
.tab-nav { border-bottom: 1px solid #1a2744 !important; background: transparent !important; }
.tab-nav button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important; letter-spacing: 0.05em !important;
    color: #4a6080 !important; background: transparent !important;
    border: none !important; padding: 0.7rem 1.2rem !important;
    text-transform: uppercase !important;
}
.tab-nav button.selected {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
}

/* Buttons */
button.primary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    background: linear-gradient(135deg, #004d40, #00695c) !important;
    color: #00d4aa !important; border: 1px solid #00d4aa !important;
    border-radius: 4px !important; letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #00695c, #00897b) !important;
    box-shadow: 0 0 12px rgba(0,212,170,0.3) !important;
}
button.secondary {
    font-family: 'JetBrains Mono', monospace !important;
    background: #0d1526 !important; color: #1890ff !important;
    border: 1px solid #1890ff !important; border-radius: 4px !important;
}
button.stop {
    background: #1a0808 !important; color: #ff4d4f !important;
    border: 1px solid #ff4d4f !important; border-radius: 4px !important;
}

/* Labels */
label span, .gradio-container label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important; color: #4a6080 !important;
    text-transform: uppercase !important; letter-spacing: 0.05em !important;
}

/* Slider */
input[type=range] { -webkit-appearance:none; height:3px;
                    background:#1a2744; border-radius:2px; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:14px; height:14px;
    border-radius:50%; background:#00d4aa; cursor:pointer;
    border:2px solid #0a0f1e;
}

/* Textarea */
textarea, .gradio-container textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important; background: #060c18 !important;
    color: #00d4aa !important; border: 1px solid #1a2744 !important;
    border-radius: 4px !important;
}

/* Markdown */
.gradio-container h2 { color: #00d4aa !important; font-family:'JetBrains Mono',monospace !important; }
.gradio-container h3 { color: #1890ff !important; }
.gradio-container p  { color: #8899b0 !important; }
table { width:100%; border-collapse:collapse; }
th { background:#0d1526; color:#00d4aa;
     font-family:'JetBrains Mono',monospace; font-size:0.72rem;
     text-align:left; padding:7px 12px; border-bottom:1px solid #1a2744;
     text-transform:uppercase; letter-spacing:0.05em; }
td { padding:7px 12px; border-bottom:1px solid #060c18;
     color:#e8eaf0; font-size:0.85rem; }
blockquote { border-left:3px solid #00d4aa; padding-left:1rem;
             color:#4a6080 !important; margin:0.5rem 0; background:#0d1526;
             padding:0.6rem 1rem; border-radius:0 4px 4px 0; }
code { font-family:'JetBrains Mono',monospace; background:#0d1526;
       color:#faad14; padding:1px 4px; border-radius:3px; }

footer { display:none !important; }
.gradio-container .block { background:transparent !important; border:none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

STRATEGY_CHOICES = ["DP Optimal", "DQN Agent", "Rule-Based", "No Storage"]
PRICE_PROFILES = list(BASE_PRICES.keys())
LOAD_PROFILES_LIST = list(LOAD_PROFILES.keys())

with gr.Blocks(title="Smart Grid Energy Optimizer") as demo:
    gr.HTML("""
    <div class="sg-header">
        <div class="sg-title">⚡ SMART GRID ENERGY OPTIMIZER</div>
        <div class="sg-subtitle">
            Battery Energy Storage System · Reinforcement Learning + Dynamic Programming
        </div>
        <div class="sg-badges">
            <span class="sg-badge badge-teal">● DQN Agent</span>
            <span class="sg-badge badge-green">● DP Optimal Solver</span>
            <span class="sg-badge badge-blue">● Real-Time Dispatch</span>
            <span class="sg-badge badge-orange">● Scenario Lab</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Dispatch Dashboard
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("⚡ DISPATCH"):
            gr.HTML("""
            <div style="padding:0.8rem 0 0.5rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    24-HOUR BATTERY DISPATCH SIMULATION
                </div>
                <div style="color:#8899b0;font-size:0.85rem;margin-top:0.3rem;">
                    Run any strategy on a chosen grid scenario and see the full energy dispatch,
                    price curve, and economics breakdown.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    disp_scenario = gr.Dropdown(
                        list(SCENARIO_PRESETS.keys()),
                        value="☀️ Summer Peak",
                        label="Grid Scenario",
                    )
                    disp_strategy = gr.Dropdown(
                        STRATEGY_CHOICES,
                        value="DP Optimal",
                        label="Control Strategy",
                        info="DQN requires a trained model (see Training tab)",
                    )
                    btn_dispatch = gr.Button("▶ RUN DISPATCH", variant="primary")

                    gr.HTML("""
                    <div style="background:#0d1526;border:1px solid #1a2744;border-radius:6px;
                                padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                    color:#4a6080;text-transform:uppercase;margin-bottom:0.5rem;">
                            STRATEGY GUIDE
                        </div>
                        <div style="font-size:0.82rem;color:#8899b0;line-height:1.7;">
                            <div><span style="color:#52c41a">■</span> <strong style="color:#e8eaf0">DP Optimal</strong> — knows all future prices (theoretical max)</div>
                            <div><span style="color:#1890ff">■</span> <strong style="color:#e8eaf0">DQN Agent</strong> — neural net, no price foresight</div>
                            <div><span style="color:#faad14">■</span> <strong style="color:#e8eaf0">Rule-Based</strong> — charge low, discharge high (industry heuristic)</div>
                            <div><span style="color:#ff4d4f">■</span> <strong style="color:#e8eaf0">No Storage</strong> — baseline (grid + solar only)</div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=2):
                    disp_status = gr.Markdown(
                        "*Select scenario and strategy, then click Run Dispatch.*"
                    )

            disp_fig = gr.Plot(label="Dispatch Dashboard")

            btn_dispatch.click(
                cb_run_dispatch,
                inputs=[disp_scenario, disp_strategy],
                outputs=[disp_fig, disp_status],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Benchmark
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🏆 BENCHMARK"):
            gr.HTML("""
            <div style="padding:0.8rem 0 0.5rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    STRATEGY COMPARISON · SAME SCENARIO
                </div>
                <div style="color:#8899b0;font-size:0.85rem;margin-top:0.3rem;">
                    All strategies run on identical conditions.
                    Gap between DQN and DP Optimal quantifies how much RL has learned.
                </div>
            </div>
            """)

            with gr.Row():
                bench_scenario = gr.Dropdown(
                    list(SCENARIO_PRESETS.keys()),
                    value="☀️ Summer Peak",
                    label="Scenario",
                )
                btn_bench = gr.Button("🏆 RUN BENCHMARK", variant="primary")

            bench_fig = gr.Plot(label="Benchmark Comparison")
            bench_summary = gr.Markdown(
                "*Click Run Benchmark to compare all strategies.*"
            )

            btn_bench.click(
                cb_benchmark,
                inputs=[bench_scenario],
                outputs=[bench_fig, bench_summary],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Train DQN
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🧠 TRAIN DQN"):
            gr.HTML("""
            <div style="padding:0.8rem 0 0.5rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    DEEP Q-NETWORK TRAINING
                </div>
                <div style="color:#8899b0;font-size:0.85rem;margin-top:0.3rem;">
                    Train a DQN agent from scratch. The agent sees: time-of-day, battery SOC,
                    current price, price trend, solar output, and building load.
                    It has NO access to future prices — unlike the DP solver.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    train_scenario = gr.Dropdown(
                        list(SCENARIO_PRESETS.keys()),
                        value="☀️ Summer Peak",
                        label="Training Scenario",
                    )
                    train_steps = gr.Slider(
                        5_000,
                        200_000,
                        value=30_000,
                        step=5_000,
                        label="Training Timesteps",
                        info="~30k = fast (~30s) · ~100k = better policy · ~200k = near-optimal",
                    )
                    train_lr = gr.Slider(
                        1e-5,
                        1e-3,
                        value=5e-4,
                        step=1e-5,
                        label="Learning Rate",
                    )
                    train_batch = gr.Slider(
                        32,
                        256,
                        value=64,
                        step=32,
                        label="Batch Size",
                    )
                    with gr.Row():
                        btn_train = gr.Button("▶ START TRAINING", variant="primary")
                        btn_stop = gr.Button("⏹ STOP", variant="stop")
                    btn_refresh = gr.Button("🔄 REFRESH METRICS", variant="secondary")
                    train_msg = gr.Textbox(label="Status", lines=2, interactive=False)

                with gr.Column(scale=2):
                    train_status_md = gr.Markdown(
                        "*Start training to see live metrics.*"
                    )
                    train_fig = gr.Plot(label="Training Dashboard")

            btn_train.click(
                cb_start_training,
                inputs=[train_scenario, train_steps, train_lr, train_batch],
                outputs=[train_msg],
            )
            btn_stop.click(cb_stop_training, outputs=[train_msg])
            btn_refresh.click(cb_refresh_training, outputs=[train_fig, train_status_md])

            gr.HTML("""
            <div style="background:#0d1526;border:1px solid #1a2744;border-radius:8px;
                        padding:1.2rem;margin-top:1rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                            color:#4a6080;text-transform:uppercase;margin-bottom:0.8rem;">
                    DQN ARCHITECTURE
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
                    <div>
                        <div style="color:#00d4aa;font-family:'JetBrains Mono',monospace;font-size:0.8rem;font-weight:600;">Input (8)</div>
                        <div style="color:#4a6080;font-size:0.78rem;margin-top:0.2rem;">sin/cos hour · SOC · price · trend · solar · load · t2peak</div>
                    </div>
                    <div>
                        <div style="color:#00d4aa;font-family:'JetBrains Mono',monospace;font-size:0.8rem;font-weight:600;">Hidden (128→128→64)</div>
                        <div style="color:#4a6080;font-size:0.78rem;margin-top:0.2rem;">ReLU activations · 3 fully-connected layers</div>
                    </div>
                    <div>
                        <div style="color:#00d4aa;font-family:'JetBrains Mono',monospace;font-size:0.8rem;font-weight:600;">Output (7)</div>
                        <div style="color:#4a6080;font-size:0.78rem;margin-top:0.2rem;">Q-values for actions -3 to +3 kW</div>
                    </div>
                    <div>
                        <div style="color:#00d4aa;font-family:'JetBrains Mono',monospace;font-size:0.8rem;font-weight:600;">Key features</div>
                        <div style="color:#4a6080;font-size:0.78rem;margin-top:0.2rem;">Experience replay · Target network · ε-greedy exploration</div>
                    </div>
                </div>
            </div>
            """)

        # ══════════════════════════════════════════════════════════════════
        # Tab 4 — Scenario Lab
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🔬 SCENARIO LAB"):
            gr.HTML("""
            <div style="padding:0.8rem 0 0.5rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    CUSTOM SCENARIO BUILDER
                </div>
                <div style="color:#8899b0;font-size:0.85rem;margin-top:0.3rem;">
                    Build a custom grid scenario and preview before running dispatch or benchmark.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    lab_price = gr.Dropdown(
                        PRICE_PROFILES, value="summer_peak", label="Price Profile"
                    )
                    lab_load = gr.Dropdown(
                        LOAD_PROFILES_LIST, value="commercial", label="Building Type"
                    )
                    lab_solar = gr.Slider(
                        0, 10, value=5.0, step=0.5, label="Solar Capacity (kW)"
                    )
                    lab_batcap = gr.Slider(
                        2, 50, value=10.0, step=1, label="Battery Capacity (kWh)"
                    )
                    lab_rate = gr.Slider(
                        1, 6, value=3.0, step=0.5, label="Max Charge Rate (kW)"
                    )
                    lab_eff = gr.Slider(
                        0.7, 0.99, value=0.92, step=0.01, label="Round-Trip Efficiency"
                    )
                    lab_noise = gr.Checkbox(
                        label="Add price & solar uncertainty", value=False
                    )
                    btn_lab = gr.Button("🔬 PREVIEW SCENARIO", variant="primary")
                    lab_status = gr.Markdown("")

                with gr.Column(scale=2):
                    lab_fig = gr.Plot(label="Scenario Preview")

            btn_lab.click(
                cb_custom_scenario,
                inputs=[
                    lab_price,
                    lab_solar,
                    lab_load,
                    lab_batcap,
                    lab_rate,
                    lab_eff,
                    lab_noise,
                ],
                outputs=[lab_fig, lab_status],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 5 — Technical Guide
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📋 TECHNICAL GUIDE"):
            gr.Markdown("""
## Smart Grid Energy Storage — System Overview

A **Battery Energy Storage System (BESS)** connected to:
- The **electricity grid** (buy/sell at market prices)
- **Solar PV panels** (free generation, intermittent)
- A **building load** (demand to be met)

The optimisation objective: minimise net electricity cost (or maximise revenue)
by intelligently deciding when to charge, discharge, or hold the battery.

---

## Why Reinforcement Learning?

Dynamic Programming (DP) solves the problem **perfectly** — but requires
knowing all future prices. In reality, prices are uncertain.

**RL learns a policy that generalises across price scenarios** without needing
future knowledge. This is the practical, deployable solution.

| Method | Price Knowledge | Real-World Applicable |
|---|---|---|
| DP Optimal | Perfect foresight ✓ | No — cheats |
| Rule-Based | Current price only | Partial |
| **DQN Agent** | **Current state only** | **Yes** |

---

## State Space (8 dimensions)

| Feature | Encoding | Why |
|---|---|---|
| `sin(hour × π/12)` | Cyclic | Encodes time-of-day smoothly |
| `cos(hour × π/12)` | Cyclic | Pair with sin for full hour info |
| `SOC (normalised)` | [−1, 1] | Current battery level |
| `Price (normalised)` | [−1, 1] | Is price currently high or low? |
| `Price trend` | [−1, 1] | Is price rising or falling? |
| `Solar (normalised)` | [−1, 1] | Current generation available |
| `Load (normalised)` | [−1, 1] | Current demand to serve |
| `Time to peak` | [−1, 1] | Urgency to charge before peak |

---

## Action Space (7 discrete actions)

`−3, −2, −1, 0, +1, +2, +3 kW`

Negative = discharging (selling/supplying load), Positive = charging (buying/storing).
Constrained by battery capacity and state of charge.

---

## Reward Function

```
Revenue from discharge offsetting load  = kW_discharged × efficiency × price
Revenue from grid export                = kW_exported × price
Cost of grid import for charging        = kW_from_grid × price
Cost of residual load from grid         = remaining_load × price
Degradation penalty                     = |kW_actual| × 0.001
```

Net reward = Revenue − Costs (in dollars per hour step)

---

## DQN Key Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `gamma` | 0.97 | High discount — values future peak revenue |
| `exploration_fraction` | 0.30 | 30% of training is random exploration |
| `learning_starts` | 500 | Fills replay buffer before first update |
| `train_freq` | 24 | Updates every episode (24 steps) |
| `target_update_interval` | 240 | Target network updated every 10 episodes |
| `net_arch` | [128, 128, 64] | 3-layer MLP, ~35k parameters |

---

## Interpreting the Charts

- **SOC curve**: Should charge during low-price hours, hold through peak, discharge at peak
- **DP vs DQN gap**: Smaller gap = DQN has learned closer to optimal policy
- **Rolling reward**: Should trend upward during training. Plateau = convergence
- **Hourly P&L**: Green bars = profitable hours, red = costly hours
""")

    gr.HTML("""
    <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#1a2744;padding:1.5rem 0 0.5rem;border-top:1px solid #0d1526;
                margin-top:1rem;letter-spacing:0.1em;text-transform:uppercase;">
        BESS Optimizer · DQN + DP + Rule-Based · Gymnasium · Stable-Baselines3
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
