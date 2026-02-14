"""
MAB Banner Optimizer — Ad Campaign Intelligence Platform
End-to-end Multi-Armed Bandit showcase:
  · 6 algorithms compared head-to-head on the same environment
  · Interactive learner mode — step through decisions with live belief updates
  · Campaign analytics deep-dive per algorithm
  · 4 real-world scenario presets + custom builder
"""

from __future__ import annotations
import json
import gradio as gr

from bandits.environment import CampaignEnvironment, BannerArm, SCENARIOS
from bandits.agents import make_agent, AGENT_REGISTRY
from bandits.simulator import run_comparison, run_single
from viz.charts import (
    comparison_dashboard, belief_chart, campaign_analytics,
    scenario_preview, learner_step_chart, empty_fig,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_arms(names_str: str, ctrs_str: str, revs_str: str) -> list[BannerArm]:
    names = [n.strip() for n in names_str.split(",") if n.strip()]
    ctrs  = json.loads(ctrs_str)
    revs  = json.loads(revs_str)
    if len(names) != len(ctrs) or len(names) != len(revs):
        raise ValueError(f"Length mismatch: {len(names)} names, {len(ctrs)} CTRs, {len(revs)} revenues")
    return [BannerArm(n, float(c), float(r)) for n, c, r in zip(names, ctrs, revs)]


def _arms_from_scenario(scenario_name: str) -> list[BannerArm]:
    return SCENARIOS[scenario_name]["arms"]


def _agent_params(algo: str, eps: float, c: float, alpha: float, gamma: float) -> dict:
    return {
        "Epsilon-Greedy":          {"epsilon": eps},
        "Decaying Epsilon-Greedy": {"decay": 1.0},
        "UCB1":             {"c": c},
        "Gradient Bandit":  {"alpha": alpha},
        "EXP3 (Adversarial)":{"gamma": gamma},
        "Thompson Sampling":{},
    }.get(algo, {})


# ── Tab 1: Algorithm Face-Off ─────────────────────────────────────────────────

def cb_faceoff(
    scenario_name, custom_names, custom_ctrs, custom_revs,
    selected_algos, n_steps, drift, seed,
    eps, c_val, alpha_val, gamma_val,
    progress: gr.Progress = gr.Progress(),
):
    try:
        if scenario_name == "🔬 Custom":
            arms = _parse_arms(custom_names, custom_ctrs, custom_revs)
        else:
            arms = _arms_from_scenario(scenario_name)

        algos = selected_algos if selected_algos else ["Thompson Sampling"]
        progress(0.1, desc="Setting up environment…")

        agent_params = {a: _agent_params(a, float(eps), float(c_val),
                                          float(alpha_val), float(gamma_val))
                        for a in algos}

        agents = run_comparison(
            arms, algos, int(n_steps),
            drift_std=float(drift), seed=int(seed),
            agent_params=agent_params,
        )
        progress(0.8, desc="Rendering comparison dashboard…")

        env_probe = CampaignEnvironment(arms)
        fig = comparison_dashboard(agents, [a.name for a in arms], env_probe.optimal_ev)

        best_algo = max(agents, key=lambda k: agents[k].total_reward)
        best_rev  = agents[best_algo].total_reward

        summary = f"""
### Face-Off Complete — {len(algos)} algorithms · {int(n_steps):,} impressions

| Algorithm | Revenue | Regret | Best Arm % |
|---|---|---|---|
""" + "\n".join(
            f"| {'👑 ' if k==best_algo else ''}{k} | "
            f"`${v.total_reward:,.2f}` | `${v.cumulative_regret:,.2f}` | "
            f"`{v.arm_pull_pcts.max():.1f}%` |"
            for k, v in agents.items()
        ) + f"""

> **Winner:** {best_algo} with ${best_rev:,.2f} total revenue
> **Optimal benchmark:** ${env_probe.optimal_ev * int(n_steps):,.2f}
> **Efficiency:** {best_rev / (env_probe.optimal_ev * int(n_steps)) * 100:.1f}%
"""
        progress(1.0)
        return fig, summary

    except Exception as e:
        return empty_fig(f"Error: {e}"), f"❌ **Error:** {e}"


# ── Tab 2: Campaign Deep-Dive ─────────────────────────────────────────────────

def cb_deep_dive(
    scenario_name, custom_names, custom_ctrs, custom_revs,
    algo, n_steps, drift, seed,
    eps, c_val, alpha_val, gamma_val,
    progress: gr.Progress = gr.Progress(),
):
    try:
        if scenario_name == "🔬 Custom":
            arms = _parse_arms(custom_names, custom_ctrs, custom_revs)
        else:
            arms = _arms_from_scenario(scenario_name)

        progress(0.1, desc=f"Running {algo}…")
        params = _agent_params(algo, float(eps), float(c_val), float(alpha_val), float(gamma_val))
        env    = CampaignEnvironment(arms, drift_std=float(drift), seed=int(seed))
        agent  = make_agent(algo, len(arms), params)
        run_single(env, agent, int(n_steps), seed=int(seed))

        progress(0.6, desc="Rendering analytics…")
        arm_names = [a.name for a in arms]
        true_ctrs = [a.true_ctr for a in arms]
        revenues  = [a.revenue for a in arms]

        analytics_fig = campaign_analytics(agent, arm_names, true_ctrs, revenues)
        belief_fig    = belief_chart(agent, arm_names, true_ctrs, int(n_steps))

        progress(1.0)
        return analytics_fig, belief_fig, f"Deep-dive complete for **{algo}** on **{scenario_name}**"

    except Exception as e:
        return empty_fig(f"Error: {e}"), empty_fig(""), f"❌ {e}"


# ── Tab 3: Learner Mode ───────────────────────────────────────────────────────

def cb_learner_init(scenario_name, custom_names, custom_ctrs, custom_revs, algo,
                    eps, c_val, alpha_val, gamma_val):
    try:
        if scenario_name == "🔬 Custom":
            arms = _parse_arms(custom_names, custom_ctrs, custom_revs)
        else:
            arms = _arms_from_scenario(scenario_name)

        params = _agent_params(algo, float(eps), float(c_val), float(alpha_val), float(gamma_val))
        env    = CampaignEnvironment(arms, drift_std=0.002, seed=42)
        agent  = make_agent(algo, len(arms), params)
        env.reset(seed=42)

        arm_names = [a.name for a in arms]
        true_ctrs = [a.true_ctr for a in arms]
        revenues  = [a.revenue for a in arms]

        fig = scenario_preview(arm_names, true_ctrs, revenues)
        status = f"✅ **{algo}** initialised on **{scenario_name}**. Click **Next Step** to begin."

        return (env, agent, arm_names, true_ctrs, revenues, 0, -1, 0.0), \
               fig, status, gr.update(interactive=True)

    except Exception as e:
        return None, empty_fig(f"Error: {e}"), f"❌ {e}", gr.update(interactive=False)


def cb_learner_step(state):
    if state is None:
        return state, empty_fig("Init first."), "⚠️ Click Init first."

    env, agent, arm_names, true_ctrs, revenues, step, *_ = state

    arm    = agent.choose()
    reward, converted = env.pull(arm)
    agent.update(arm, reward, converted, env.optimal_ev)

    fig    = learner_step_chart(agent, arm_names, true_ctrs, revenues, arm, reward)
    status = (
        f"**Step {agent.t}** · Chose: **{arm_names[arm]}** · "
        f"Reward: **${reward:.2f}** · Converted: {'✅' if converted else '❌'} · "
        f"Total Revenue: **${agent.total_reward:.2f}**"
    )
    new_state = (env, agent, arm_names, true_ctrs, revenues, step + 1, arm, reward)
    return new_state, fig, status


# ── Tab 4: Scenario Builder ───────────────────────────────────────────────────

def cb_preview_scenario(scenario_name, custom_names, custom_ctrs, custom_revs):
    try:
        if scenario_name == "🔬 Custom":
            arms = _parse_arms(custom_names, custom_ctrs, custom_revs)
        else:
            arms = _arms_from_scenario(scenario_name)
        arm_names = [a.name for a in arms]
        true_ctrs = [a.true_ctr for a in arms]
        revenues  = [a.revenue for a in arms]
        desc = SCENARIOS.get(scenario_name, {}).get("description", "Custom scenario")
        fig  = scenario_preview(arm_names, true_ctrs, revenues)
        info = f"**{scenario_name}** — {desc}\n\n**{len(arms)} banners** · Best EV: ${max(c*r for c,r in zip(true_ctrs,revenues)):.4f}/impression"
        return fig, info
    except Exception as e:
        return empty_fig(f"Error: {e}"), f"❌ {e}"


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

.mab-header {
    border-bottom: 1px solid #1e2a3d;
    padding: 1.4rem 1.5rem 0.9rem;
    background: linear-gradient(180deg, #0a0d14 0%, #0f1520 100%);
}
.mab-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.2rem, 3vw, 1.9rem);
    font-weight: 700; color: #00d4aa; margin: 0;
}
.mab-sub { color: #4a6080; font-size: 0.85rem; margin-top: 0.3rem; }
.mab-badges { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.7rem; }
.mab-badge {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    letter-spacing:0.08em; padding:3px 10px; border-radius:3px; text-transform:uppercase;
}
.b-teal   { background:#061a16; color:#00d4aa; border:1px solid #0a4a3c; }
.b-blue   { background:#0a1a30; color:#1890ff; border:1px solid #1a3a6e; }
.b-amber  { background:#1e1206; color:#faad14; border:1px solid #5c3a06; }
.b-purple { background:#130a1e; color:#a855f7; border:1px solid #3a1a6e; }
.b-red    { background:#1a0808; color:#ef4444; border:1px solid #5c1a1a; }

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
    background:#0f1520 !important; color:#1890ff !important;
    border:1px solid #1890ff !important; border-radius:5px !important;
}
button.stop {
    background:#0f1520 !important; color:#a855f7 !important;
    border:1px solid #a855f7 !important; border-radius:5px !important;
}

label span, .gradio-container label {
    font-family:'JetBrains Mono',monospace !important; font-size:0.7rem !important;
    color:#4a6080 !important; text-transform:uppercase !important;
    letter-spacing:0.06em !important;
}
input[type=range] { -webkit-appearance:none; height:3px;
                    background:#1e2a3d; border-radius:2px; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:14px; height:14px;
    border-radius:50%; background:#00d4aa; cursor:pointer;
    border:2px solid #0a0d14;
}
textarea, .gradio-container textarea, input[type=text] {
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

# ── Shared scenario sidebar ───────────────────────────────────────────────────

ALGO_NAMES = list(AGENT_REGISTRY.keys())
SCENARIO_NAMES = list(SCENARIOS.keys())

DEFAULT_NAMES = "Red Urgency, Blue Minimal, Green Social, Video Teaser"
DEFAULT_CTRS  = "[0.04, 0.09, 0.06, 0.12]"
DEFAULT_REVS  = "[45, 45, 45, 45]"


def _scenario_inputs():
    scenario = gr.Dropdown(SCENARIO_NAMES, value="🛒 E-Commerce Sale",
                           label="Scenario Preset")
    with gr.Accordion("✏️ Custom Scenario", open=False):
        names = gr.Textbox(value=DEFAULT_NAMES, label="Banner Names (comma-sep)")
        ctrs  = gr.Textbox(value=DEFAULT_CTRS,  label="True CTRs (JSON list)")
        revs  = gr.Textbox(value=DEFAULT_REVS,  label="Revenues $ (JSON list)")
    return scenario, names, ctrs, revs


def _algo_params():
    with gr.Accordion("⚙️ Algorithm Parameters", open=False):
        eps   = gr.Slider(0.01, 0.5,  value=0.1,  step=0.01, label="ε (Epsilon-Greedy)")
        c_val = gr.Slider(0.5,  5.0,  value=2.0,  step=0.5,  label="c (UCB1 confidence)")
        alpha = gr.Slider(0.01, 1.0,  value=0.1,  step=0.01, label="α (Gradient Bandit)")
        gamma = gr.Slider(0.01, 0.5,  value=0.1,  step=0.01, label="γ (EXP3)")
    return eps, c_val, alpha, gamma


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="MAB Banner Optimizer") as demo:

    gr.HTML("""
    <div class="mab-header">
        <div class="mab-title">🎯 MAB BANNER OPTIMIZER</div>
        <div class="mab-sub">
            Multi-Armed Bandit · Ad Campaign Intelligence ·
            6 Algorithms · Real-Time Exploration vs Exploitation
        </div>
        <div class="mab-badges">
            <span class="mab-badge b-teal">● Thompson Sampling</span>
            <span class="mab-badge b-blue">● UCB1</span>
            <span class="mab-badge b-amber">● ε-Greedy</span>
            <span class="mab-badge b-purple">● Gradient Bandit</span>
            <span class="mab-badge b-red">● EXP3 Adversarial</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Algorithm Face-Off
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🏁 ALGORITHM FACE-OFF"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    HEAD-TO-HEAD COMPARISON · IDENTICAL ENVIRONMENT
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    All algorithms see the same banner impressions in the same order.
                    Watch which strategy accumulates the most revenue and lowest regret.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    fo_scenario, fo_names, fo_ctrs, fo_revs = _scenario_inputs()

                    fo_algos = gr.CheckboxGroup(
                        ALGO_NAMES,
                        value=["Thompson Sampling", "UCB1", "Epsilon-Greedy"],
                        label="Select Algorithms to Compare",
                    )
                    fo_steps = gr.Slider(500, 20000, value=5000, step=500,
                                         label="Impressions (steps)")
                    fo_drift = gr.Slider(0, 0.01, value=0.002, step=0.001,
                                         label="Market Drift (CTR noise/step)")
                    fo_seed  = gr.Slider(0, 100, value=42, step=1, label="Random Seed")
                    fo_eps, fo_c, fo_alpha, fo_gamma = _algo_params()
                    btn_fo   = gr.Button("🏁 RUN FACE-OFF", variant="primary")

                with gr.Column(scale=2):
                    fo_summary = gr.Markdown("*Select algorithms and click Run Face-Off.*")

            fo_chart = gr.Plot(label="Algorithm Comparison Dashboard")
            btn_fo.click(
                cb_faceoff,
                [fo_scenario, fo_names, fo_ctrs, fo_revs,
                 fo_algos, fo_steps, fo_drift, fo_seed,
                 fo_eps, fo_c, fo_alpha, fo_gamma],
                [fo_chart, fo_summary],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Campaign Deep-Dive
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📊 CAMPAIGN ANALYTICS"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    SINGLE ALGORITHM DEEP DIVE
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Per-banner performance breakdown, arm selection timeline,
                    rolling reward, and belief state after the full campaign.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    dd_scenario, dd_names, dd_ctrs, dd_revs = _scenario_inputs()
                    dd_algo  = gr.Dropdown(ALGO_NAMES, value="Thompson Sampling",
                                            label="Algorithm")
                    dd_steps = gr.Slider(500, 20000, value=5000, step=500,
                                          label="Impressions")
                    dd_drift = gr.Slider(0, 0.01, value=0.002, step=0.001,
                                          label="Market Drift")
                    dd_seed  = gr.Slider(0, 100, value=42, step=1, label="Seed")
                    dd_eps, dd_c, dd_alpha, dd_gamma = _algo_params()
                    btn_dd   = gr.Button("📊 RUN ANALYTICS", variant="primary")

                with gr.Column(scale=2):
                    dd_status = gr.Markdown("*Run analytics to see deep-dive charts.*")
                    dd_belief = gr.Plot(label="Final Belief State")

            dd_chart = gr.Plot(label="Campaign Analytics Dashboard")
            btn_dd.click(
                cb_deep_dive,
                [dd_scenario, dd_names, dd_ctrs, dd_revs,
                 dd_algo, dd_steps, dd_drift, dd_seed,
                 dd_eps, dd_c, dd_alpha, dd_gamma],
                [dd_chart, dd_belief, dd_status],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Interactive Learner
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🎓 LEARNER MODE"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    STEP-BY-STEP INTERACTIVE EXPLORATION
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Watch the algorithm make decisions one impression at a time.
                    See beliefs update in real time. Click Next Step as many times as you want.
                </div>
            </div>
            """)

            lm_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    lm_scenario, lm_names, lm_ctrs, lm_revs = _scenario_inputs()
                    lm_algo = gr.Dropdown(ALGO_NAMES, value="Thompson Sampling",
                                           label="Algorithm")
                    lm_eps, lm_c, lm_alpha, lm_gamma = _algo_params()

                    with gr.Row():
                        btn_lm_init = gr.Button("⚡ INITIALISE", variant="primary")
                        btn_lm_step = gr.Button("➡️ NEXT STEP", variant="secondary",
                                                interactive=False)

                    lm_status = gr.Markdown("*Click Initialise to begin.*")

                with gr.Column(scale=2):
                    lm_chart = gr.Plot(label="Live Belief / Q-Value State")

            btn_lm_init.click(
                cb_learner_init,
                [lm_scenario, lm_names, lm_ctrs, lm_revs,
                 lm_algo, lm_eps, lm_c, lm_alpha, lm_gamma],
                [lm_state, lm_chart, lm_status, btn_lm_step],
            )
            btn_lm_step.click(
                cb_learner_step,
                [lm_state],
                [lm_state, lm_chart, lm_status],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 4 — Scenario Builder
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🔬 SCENARIO LAB"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">
                    SCENARIO BUILDER
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Preview banner expected values before running a campaign.
                    Build custom scenarios or explore presets.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    sb_scenario, sb_names, sb_ctrs, sb_revs = _scenario_inputs()
                    btn_sb = gr.Button("🔬 PREVIEW", variant="primary")

                    gr.HTML("""
                    <div style="background:#0f1520;border:1px solid #1e2a3d;border-radius:6px;
                                padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                    color:#4a6080;text-transform:uppercase;margin-bottom:0.5rem;">
                            PRESET SCENARIOS
                        </div>
                        <div style="font-size:0.8rem;color:#64748b;line-height:1.9;">
                            <div>🛒 <strong style="color:#e2e8f0">E-Commerce</strong> — Flash sale banners</div>
                            <div>📰 <strong style="color:#e2e8f0">News Feed</strong> — 5 headline variants</div>
                            <div>💰 <strong style="color:#e2e8f0">SaaS Pricing</strong> — High-value CTAs</div>
                            <div>🎮 <strong style="color:#e2e8f0">Mobile Game</strong> — 6 install creatives</div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=2):
                    sb_info  = gr.Markdown("*Select a scenario and click Preview.*")
                    sb_chart = gr.Plot(label="Banner Expected Values")

            btn_sb.click(cb_preview_scenario,
                         [sb_scenario, sb_names, sb_ctrs, sb_revs],
                         [sb_chart, sb_info])

        # ══════════════════════════════════════════════════════════════════
        # Tab 5 — How It Works
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📚 HOW IT WORKS"):
            gr.Markdown("""
## The Exploration vs Exploitation Dilemma

Every time a banner is shown, you must choose: **show the banner you think is best**
(exploitation) or **try a different one** to learn more about it (exploration).

Show the best banner too early → you might miss an even better one.
Explore too long → you waste impressions on bad banners.

Multi-Armed Bandit algorithms solve this dilemma systematically.

---

## The 6 Algorithms

### 🎲 ε-Greedy
Simplest approach: with probability ε, pick randomly; otherwise pick the best-known banner.
- **Fast**, easy to understand
- **Problem:** Fixed exploration forever, even after learning

### 📉 Decaying ε-Greedy
Same as ε-Greedy but ε decays as `1/√t` — explores more early, exploits later.
- **Better** than fixed ε for stationary environments
- **Natural** transition from exploration to exploitation

### 🔭 UCB1 (Upper Confidence Bound)
Always pick the banner with highest `Q(a) + c × √(log(t) / n(a))`.
The bonus decreases as an arm is pulled more — **principled optimism under uncertainty**.
- **Theoretically optimal** O(log T) regret bound
- **No randomness** at decision time — deterministic given history

### 🎰 Thompson Sampling
Bayesian: maintain a Beta(α, β) distribution for each banner's CTR.
Sample from each posterior, pick the highest sample.
- **Best practical performance** in most real ad systems
- **Natural uncertainty representation** — wide posteriors = more exploration

### 📈 Gradient Bandit
Learns a **preference H(a)** for each arm using stochastic gradient ascent.
Action probabilities are softmax(H). Reward above a moving baseline increases preference.
- **Works without knowing reward scale**
- **Converges** even in non-stationary environments

### ⚔️ EXP3 (Adversarial)
Designed for **adversarial** settings where the environment can be hostile.
Uses importance-weighted updates to guard against worst-case losses.
- **Strongest guarantees** — works even if an adversary controls rewards
- **Practical** when market dynamics are unpredictable

---

## The Environment

Each banner arm has:
- **True CTR** (hidden from agents): probability of a click per impression
- **Revenue** ($): value per conversion

**Reward** = `Bernoulli(CTR) × Revenue` — stochastic!

**Non-stationarity**: CTRs drift slowly each step (`N(0, σ²)`) — simulating
real-world effects like ad fatigue, seasonality, and competitor actions.

**Regret** = `Σ (optimal_EV - received_reward)` — cumulative opportunity cost.

---

## Reading the Charts

- **Cumulative Revenue**: higher = better. Optimal line = perfect foresight upper bound.
- **Cumulative Regret**: lower = better. Slope decreasing = agent is learning.
- **Arm Pull Distribution**: a well-calibrated agent should pull the best arm most often.
- **Beta PDFs** (Thompson): wide = uncertain; narrow = confident; true CTR shown as dashed line.
- **Arm Selection Timeline**: see when each algorithm commits to (or abandons) banners.

---

## Real-World Applications

| Industry | Arms | Reward |
|---|---|---|
| **Display Advertising** | Banner creatives | CTR × bid |
| **Email Marketing** | Subject line variants | Open rate × revenue |
| **Recommendation** | Content/product cards | Click × purchase |
| **Pricing** | Price points | Conversion × margin |
| **Clinical Trials** | Treatment arms | Patient outcome |
| **Search Ranking** | Result orderings | Click-through |
""")

    gr.HTML("""
    <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                color:#1e2a3d;padding:1.5rem 0 0.5rem;border-top:1px solid #0f1520;
                letter-spacing:0.1em;text-transform:uppercase;">
        Thompson Sampling · UCB1 · ε-Greedy · Gradient Bandit · EXP3 · Gradio
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
