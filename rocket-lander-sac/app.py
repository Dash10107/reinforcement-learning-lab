"""
SpaceX Mission Control — SAC Rocket Lander
Production Gradio application: simulate, visualise, analyse, and train
a Soft Actor-Critic agent on the LunarLander-v3 continuous control task.
"""

from __future__ import annotations
import os
import numpy as np
import gradio as gr
from stable_baselines3 import SAC

from core.mission import run_mission, MissionResult
from core.trainer import TrainingState, start_training
from viz.charts import (
    mission_overview, single_episode_detail,
    training_dashboard, empty_figure,
)
from viz.replay import make_episode_gif, make_comparison_gif

# ── Model loading ─────────────────────────────────────────────────────────────

_MODEL_PATHS = ["sac_finetuned.zip", "sac_rocket_lander.zip"]
_model: SAC | None = None

def _load_model(path: str | None = None) -> tuple[SAC, str]:
    candidates = ([path] if path else []) + _MODEL_PATHS
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return SAC.load(p), p
            except Exception:
                continue
    raise FileNotFoundError("No valid SAC checkpoint found.")

def _get_model() -> SAC:
    global _model
    if _model is None:
        _model, _ = _load_model()
    return _model

# ── Global training state ─────────────────────────────────────────────────────
_train_state = TrainingState()

# ── Callbacks ─────────────────────────────────────────────────────────────────

def cb_run_mission(
    n_episodes: int,
    gravity: float,
    enable_wind: bool,
    wind_power: float,
    turbulence: float,
    render_gif: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    try:
        model = _get_model()
    except FileNotFoundError as e:
        empty = empty_figure(str(e))
        return empty, None, empty, str(e), gr.update(choices=[])

    mission, all_frames = run_mission(
        model,
        n_episodes=int(n_episodes),
        gravity=float(gravity),
        enable_wind=bool(enable_wind),
        wind_power=float(wind_power),
        turbulence_power=float(turbulence),
        render=bool(render_gif),
        progress_cb=progress,
    )

    overview_fig = mission_overview(mission)

    gif_path = None
    if render_gif and all_frames:
        if n_episodes >= 2:
            gif_path = make_comparison_gif(all_frames, mission.episodes, fps=15)
        else:
            gif_path = make_episode_gif(all_frames[0], mission.episodes[0], fps=15)

    best = mission.best
    detail_fig = single_episode_detail(best)

    sr = mission.success_rate * 100
    icon = "🚀" if mission.avg_reward >= 150 else "💥"
    stats_md = f"""
### {icon} Mission Complete

| Metric | Value |
|---|---|
| **Avg Reward** | `{mission.avg_reward:+.2f}` |
| **Best** | `{best.total_reward:+.2f}` ({best.status_emoji} Ep {best.episode_idx+1}) |
| **Worst** | `{mission.worst.total_reward:+.2f}` ({mission.worst.status_emoji} Ep {mission.worst.episode_idx+1}) |
| **Success Rate** | `{sr:.1f}%` |
| **Episodes** | `{len(mission.episodes)}` |

**Per-Episode Scores:**
"""
    per_ep = "".join(
        f"- `#{e.episode_idx+1}` {e.status_emoji} **{e.status}** — {e.total_reward:+.1f} ({len(e.steps)} steps)\n"
        for e in mission.episodes
    )
    stats_md += per_ep

    ep_choices = [
        f"#{e.episode_idx+1} — {e.status_emoji} {e.status} ({e.total_reward:+.1f})"
        for e in mission.episodes
    ]

    _last_mission["data"] = mission
    _last_mission["frames"] = all_frames

    return overview_fig, gif_path, detail_fig, stats_md, gr.update(choices=ep_choices, value=ep_choices[0])


_last_mission: dict = {"data": None, "frames": []}


def cb_select_episode(selection: str) -> tuple:
    mission: MissionResult | None = _last_mission.get("data")
    all_frames = _last_mission.get("frames", [])
    if not mission or not selection:
        return empty_figure("Run a mission first."), None
    try:
        idx = int(selection.split("#")[1].split(" ")[0]) - 1
    except Exception:
        idx = 0
    ep = mission.episodes[idx]
    fig = single_episode_detail(ep)
    gif = None
    if all_frames and idx < len(all_frames):
        gif = make_episode_gif(all_frames[idx], ep, fps=15)
    return fig, gif


def cb_start_training(total_steps: int, lr: float, batch_size: int) -> str:
    global _train_state
    if _train_state.running:
        return "Training already in progress."
    _train_state = TrainingState()
    start_training(
        base_model_path="sac_rocket_lander.zip",
        total_timesteps=int(total_steps),
        learning_rate=float(lr),
        batch_size=int(batch_size),
        state=_train_state,
        save_path="sac_finetuned.zip",
    )
    return "Training started. Click **Refresh** to update charts."


def cb_stop_training() -> str:
    _train_state.running = False
    return "Stop signal sent."


def cb_refresh_training() -> tuple:
    fig = training_dashboard(_train_state)
    n_ep = len(_train_state.episode_rewards)
    rolling = float(np.mean(_train_state.episode_rewards[-20:])) if n_ep else 0.0
    progress_pct = (_train_state.timestep / max(_train_state.total_timesteps, 1)) * 100
    status_md = f"""
| | |
|---|---|
| **Status** | `{_train_state.status}` |
| **Progress** | `{progress_pct:.1f}%` |
| **Episodes** | `{n_ep}` |
| **Rolling Reward (20)** | `{rolling:+.1f}` |
| **Best Reward** | `{_train_state.best_reward:+.1f}` |
"""
    return fig, status_md


def cb_load_finetuned() -> str:
    global _model
    for path in _MODEL_PATHS:
        if os.path.exists(path):
            try:
                _model = SAC.load(path)
                return f"Model loaded from `{path}`."
            except Exception as e:
                return f"Failed to load `{path}`: {e}"
    return "No checkpoint found."


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #030b1a !important;
    color: #c8ddf0 !important;
    font-family: 'Exo 2', sans-serif !important;
}

.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

.tab-nav { background: #060f1e !important; border-bottom: 1px solid #0d2540 !important; }
.tab-nav button {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 0.18em !important;
    color: #3a6080 !important; background: transparent !important;
    border: none !important; text-transform: uppercase !important;
    padding: 0.7rem 1.4rem !important;
}
.tab-nav button.selected {
    color: #4fb3ff !important;
    border-bottom: 2px solid #4fb3ff !important;
}

.mc-header {
    text-align: center; padding: 2rem 1rem 1rem;
    border-bottom: 1px solid #0d2540; margin-bottom: 1.5rem;
}
.mc-header h1 {
    font-family: 'Orbitron', monospace !important;
    font-size: clamp(1.4rem, 3.5vw, 2.2rem) !important;
    font-weight: 900 !important; letter-spacing: 0.1em !important;
    color: #e8f4ff !important; margin: 0 !important;
}
.mc-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem; color: #2d6a9f;
    letter-spacing: 0.3em; text-transform: uppercase; margin-top: 0.3rem;
}

.status-strip {
    display: flex; gap: 0.6rem; justify-content: center;
    flex-wrap: wrap; margin: 1rem 0;
}
.badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.15em;
    padding: 4px 12px; border-radius: 3px; text-transform: uppercase;
}
.badge-green  { background:#041e12; color:#2ddb7c; border:1px solid #0a5530; }
.badge-blue   { background:#020f20; color:#4fb3ff; border:1px solid #0b3362; }
.badge-amber  { background:#1a1002; color:#f5a623; border:1px solid #5c3700; }
.badge-purple { background:#120920; color:#c77dff; border:1px solid #4a1a7a; }

button.primary {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.82rem !important; font-weight: 700 !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    background: linear-gradient(135deg,#0a2a52,#0d3a72) !important;
    color: #4fb3ff !important; border: 1px solid #1a5a9e !important;
    border-radius: 4px !important; transition: all 0.2s !important;
}
button.primary:hover {
    background: linear-gradient(135deg,#0d3a72,#1150a0) !important;
    border-color: #4fb3ff !important;
    box-shadow: 0 4px 20px rgba(79,179,255,0.25) !important;
}
button.stop {
    background: linear-gradient(135deg,#2a0a0a,#4a1010) !important;
    color: #ff4d6d !important; border: 1px solid #7a1a1a !important;
    font-family: 'Share Tech Mono', monospace !important;
}

label span, .gradio-container label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 0.15em !important;
    text-transform: uppercase !important; color: #4fb3ff !important;
}
input[type=range] {
    -webkit-appearance: none; height: 3px;
    background: #0d2540; border-radius: 2px; outline: none;
}
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px;
    border-radius: 50%; background: #4fb3ff; cursor: pointer;
    border: 2px solid #030b1a; box-shadow: 0 0 8px rgba(79,179,255,0.5);
}

textarea, .gradio-container textarea {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important; line-height: 1.7 !important;
    background: #020810 !important; color: #7fcfff !important;
    border: 1px solid #0d2540 !important; border-radius: 4px !important;
}

table { width: 100%; border-collapse: collapse; }
th { background: #060f1e; color: #4fb3ff;
     font-family: 'Share Tech Mono', monospace;
     font-size: 0.7rem; letter-spacing: 0.1em; padding: 6px 10px; }
td { border-top: 1px solid #0d2540; padding: 6px 10px;
     color: #c8ddf0; font-size: 0.85rem; }

footer { display: none !important; }
.gradio-container .block { background: transparent !important; border: none !important; }
"""

# ── Theory ────────────────────────────────────────────────────────────────────

THEORY_MD = """
## Soft Actor-Critic (SAC)

SAC is an **off-policy, maximum-entropy** deep RL algorithm for continuous
action spaces. It simultaneously maximises expected return *and* policy entropy,
encouraging exploration while converging to a stable policy.

### Objective
$$J(\\pi) = \\sum_t \\mathbb{E}_{(s_t,a_t)\\sim\\rho_\\pi}\\left[ r(s_t,a_t) + \\alpha\\,\\mathcal{H}(\\pi(\\cdot|s_t)) \\right]$$

The temperature $\\alpha$ is **auto-tuned** to a target entropy level.

### Architecture

| Component | Role |
|---|---|
| **Actor** $\\pi_\\phi(a\\|s)$ | Gaussian policy — outputs mean & log-std |
| **Critic 1** $Q_{\\theta_1}(s,a)$ | Q-value estimator |
| **Critic 2** $Q_{\\theta_2}(s,a)$ | Clipped double-Q: take min to reduce overestimation |
| **Target Critics** | Soft-updated copies ($\\tau=0.005$) for stable TD targets |

### Update Rules

**Critic** — minimise Bellman residual:
$$y = r + \\gamma\\min_i Q_{\\bar\\theta_i}(s',\\tilde a') - \\alpha\\log\\pi(\\tilde a'|s')$$

**Actor** — maximise Q + entropy:
$$\\mathcal{L}(\\phi) = \\mathbb{E}\\left[\\alpha\\log\\pi_\\phi(a|s) - \\min_i Q_{\\theta_i}(s,a)\\right]$$

**Temperature** — match target entropy $\\bar{\\mathcal{H}}$:
$$\\mathcal{L}(\\alpha) = \\mathbb{E}\\left[-\\alpha(\\log\\pi(a|s)+\\bar{\\mathcal{H}})\\right]$$

---

## LunarLander-v3 (Continuous)

| Property | Value |
|---|---|
| **State** | 8-dim: pos (x,y), vel (vx,vy), angle, angular vel, leg contacts |
| **Action** | 2-dim continuous: main throttle, lateral thrust ∈ [−1,1] |
| **Reward** | +100 each leg contact, +100 landing, −100 crash |
| **Solved** | Episode reward ≥ 200 |

---

## Model Hyperparameters

| Parameter | Value |
|---|---|
| `learning_rate` | 3×10⁻⁴ |
| `buffer_size` | 1,000,000 |
| `batch_size` | 256 |
| `tau` | 0.005 |
| `gamma` | 0.99 |
| `target_entropy` | −2.0 |

---

## Reading the Charts

- **Reward bars**: green ≥ 150, amber ≥ 0, red < 0
- **Trajectory plot**: `★` = successful landing, `×` = crash
- **Engine throttle**: main (blue) fires downward; lateral (amber) steers
- **Training reward**: smoothed line (solid) trends matter more than raw (faded)
- **Actor loss**: negative values normal — actor maximises Q, so loss = −Q
- **Entropy coef**: starts high, decreases as policy converges
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="SpaceX Mission Control — SAC Rocket Lander") as demo:

    gr.HTML("""
    <div class="mc-header">
        <div class="mc-sub">Autonomous Flight Intelligence System · SAC v2.0</div>
        <h1>⬡ SpaceX Mission Control</h1>
        <div class="mc-sub">Soft Actor-Critic · LunarLander-v3 · Continuous Control</div>
    </div>
    <div class="status-strip">
        <span class="badge badge-green">● SAC MODEL LOADED</span>
        <span class="badge badge-blue">● PHYSICS ENGINE READY</span>
        <span class="badge badge-amber">● TELEMETRY ONLINE</span>
        <span class="badge badge-purple">● TRAINING MODULE ARMED</span>
    </div>
    """)

    with gr.Tabs():

        # ── Mission Control ────────────────────────────────────────────────
        with gr.Tab("🚀 MISSION CONTROL"):

            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.HTML('<div class="mc-sub" style="margin-bottom:0.8rem">MISSION PARAMETERS</div>')

                    n_episodes = gr.Slider(1, 10, value=3, step=1,
                                           label="Landing Attempts")
                    gravity = gr.Slider(-20.0, -1.0, value=-10.0, step=0.5,
                                        label="Gravity (m/s²)")
                    enable_wind = gr.Checkbox(label="Enable Wind Disturbance", value=False)
                    wind_power = gr.Slider(0.0, 20.0, value=5.0, step=0.5,
                                           label="Wind Power", visible=False)
                    turbulence = gr.Slider(0.0, 2.0, value=0.5, step=0.1,
                                           label="Turbulence Power", visible=False)
                    render_gif = gr.Checkbox(label="Render Animated Replay", value=True)

                    enable_wind.change(
                        lambda v: (gr.update(visible=v), gr.update(visible=v)),
                        inputs=enable_wind,
                        outputs=[wind_power, turbulence],
                    )

                    launch_btn = gr.Button("🚀  INITIATE LAUNCH SEQUENCE", variant="primary")

                    gr.HTML('<div class="mc-sub" style="margin-top:1.2rem;margin-bottom:0.4rem">MODEL</div>')
                    load_btn = gr.Button("📂 Reload Checkpoint")
                    load_status = gr.Textbox(label="", lines=1, interactive=False,
                                             placeholder="Model status…")
                    load_btn.click(cb_load_finetuned, outputs=load_status)

                with gr.Column(scale=2):
                    stats_md = gr.Markdown("*Configure mission parameters and click Launch.*")
                    episode_selector = gr.Dropdown(
                        choices=[], label="Inspect Episode", interactive=True,
                    )

            with gr.Row():
                overview_plot = gr.Plot(label="Mission Overview Dashboard")

            with gr.Row():
                with gr.Column(scale=1):
                    detail_plot = gr.Plot(label="Episode Deep-Dive")
                with gr.Column(scale=1):
                    replay_gif = gr.Image(
                        label="Episode Replay (GIF with HUD)",
                        type="filepath",
                    )

            episode_selector.change(
                cb_select_episode,
                inputs=episode_selector,
                outputs=[detail_plot, replay_gif],
            )
            launch_btn.click(
                cb_run_mission,
                inputs=[n_episodes, gravity, enable_wind, wind_power, turbulence, render_gif],
                outputs=[overview_plot, replay_gif, detail_plot, stats_md, episode_selector],
            )

        # ── Training Lab ───────────────────────────────────────────────────
        with gr.Tab("🧪 TRAINING LAB"):

            gr.Markdown("### Fine-tune the SAC agent in your browser")
            gr.Markdown(
                "Runs in a background thread — click **Refresh Metrics** to pull updates. "
                "The fine-tuned model saves to `sac_finetuned.zip` and is used automatically."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="mc-sub" style="margin-bottom:0.8rem">HYPERPARAMETERS</div>')
                    train_steps = gr.Slider(5_000, 200_000, value=20_000, step=5_000,
                                            label="Total Timesteps")
                    train_lr = gr.Slider(1e-5, 1e-3, value=3e-4, step=1e-5,
                                         label="Learning Rate")
                    train_batch = gr.Slider(64, 512, value=256, step=64,
                                            label="Batch Size")

                    with gr.Row():
                        btn_train_start = gr.Button("▶ Start Training", variant="primary")
                        btn_train_stop  = gr.Button("⏹ Stop", variant="stop")
                    btn_refresh = gr.Button("🔄 Refresh Metrics")
                    train_msg = gr.Textbox(label="", lines=2, interactive=False)

                with gr.Column(scale=2):
                    train_status_md = gr.Markdown("*Start training to see live metrics.*")
                    train_plot = gr.Plot(label="Live Training Dashboard")

            btn_train_start.click(
                cb_start_training,
                inputs=[train_steps, train_lr, train_batch],
                outputs=train_msg,
            )
            btn_train_stop.click(cb_stop_training, outputs=train_msg)
            btn_refresh.click(cb_refresh_training, outputs=[train_plot, train_status_md])

        # ── Algorithm Guide ────────────────────────────────────────────────
        with gr.Tab("📚 ALGORITHM GUIDE"):
            gr.Markdown(THEORY_MD)

    gr.HTML("""
    <div style="text-align:center;font-family:'Share Tech Mono',monospace;
                font-size:0.65rem;color:#1e3d5c;letter-spacing:0.2em;
                text-transform:uppercase;padding:2rem 0 1rem;">
        Powered by Stable-Baselines3 · Soft Actor-Critic ·
        Gymnasium LunarLander-v3 · Gradio
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
