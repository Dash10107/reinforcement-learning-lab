"""
MBRL Pendulum Playground
A complete Model-Based Reinforcement Learning demonstration:
  — Train a neural dynamics model from real data
  — Visualise multi-step imagination vs reality (with uncertainty)
  — Plan with Model Predictive Control (Random Shooting)
  — Understand where and why the model fails
"""

from __future__ import annotations

import os

import gradio as gr
import gymnasium as gym
import numpy as np
import torch
from model.dynamics import (
    STATE_NAMES,
    DynamicsModel,
    EnsembleDynamics,
    load_pretrained,
)
from model.train import ENSEMBLE_PATH, TrainingState, start_training_thread
from planning.mpc import (
    run_mpc_episode,
    run_random_episode,
)
from viz.plots import (
    empty_fig,
    imagination_comparison,
    make_mpc_gif,
    mpc_episode_chart,
    single_step_comparison,
    training_curve,
    uncertainty_heatmap,
)

# ── Global state ──────────────────────────────────────────────────────────────

_train_state = TrainingState()
_ensemble: EnsembleDynamics | None = None
_pretrained: DynamicsModel | None = None

# Load pre-trained single model on startup
try:
    _pretrained = load_pretrained("dynamics_model.pth")
except Exception:
    _pretrained = None


# Load ensemble if already trained
def _load_ensemble() -> EnsembleDynamics | None:
    if os.path.exists(ENSEMBLE_PATH):
        e = EnsembleDynamics(n=5, hidden=128)
        e.load(ENSEMBLE_PATH)
        e.eval()
        return e
    return None


_ensemble = _load_ensemble()


def _best_model():
    """Return ensemble if available, else pretrained single model."""
    return _ensemble if _ensemble is not None else _pretrained


# ── Tab 1: Interactive Explorer ───────────────────────────────────────────────


def cb_explore(cos_th, sin_th, vel, torque):
    model = _best_model()

    state = np.array([float(cos_th), float(sin_th), float(vel)], dtype=np.float32)
    theta = float(np.arctan2(sin_th, cos_th))

    # Real env step
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset()
    env.unwrapped.state = np.array([theta, float(vel)])
    frame = env.render()
    real_next, _, _, _, _ = env.step(np.array([float(torque)]))
    env.close()

    # Model prediction
    if model is None:
        return (
            frame,
            empty_fig("No model loaded. Train one in the Training tab."),
            "*No model available.*",
        )

    if isinstance(model, EnsembleDynamics):
        pred_next, pred_std = model.predict_np(state, float(torque))
    else:
        pred_next = model.predict_np(state, np.array([float(torque)]))
        pred_std = None

    fig = single_step_comparison(real_next, pred_next, pred_std)

    # Markdown table
    rows = "\n".join(
        f"| `{n}` | `{r:.4f}` | `{p:.4f}` | `{abs(r - p):.4f}` |"
        for n, r, p in zip(STATE_NAMES, real_next, pred_next)
    )
    unc_col = ""
    if pred_std is not None:
        rows = "\n".join(
            f"| `{n}` | `{r:.4f}` | `{p:.4f}` | `{abs(r - p):.4f}` | `±{s:.4f}` |"
            for n, r, p, s in zip(STATE_NAMES, real_next, pred_next, pred_std)
        )
        unc_col = " Uncertainty |"

    table = f"""
| State | Real | Predicted | Error |{unc_col}
|---|---|---|---|{"---|" if pred_std is not None else ""}
{rows}
"""
    model_label = (
        "Ensemble (5 models)"
        if isinstance(model, EnsembleDynamics)
        else "Single model (pretrained)"
    )
    return frame, fig, f"**Model:** {model_label}\n{table}"


# ── Tab 2: Train ──────────────────────────────────────────────────────────────


def cb_start_train(n_steps, epochs):
    global _train_state
    if _train_state.running:
        return "⚠️ Training already running."
    _train_state = TrainingState()
    start_training_thread(int(n_steps), int(epochs), _train_state)
    return f"✅ Started: collecting {int(n_steps):,} transitions, then {int(epochs)} epochs."


def cb_stop_train():
    _train_state.running = False
    return "⏹ Stop requested."


def cb_refresh_train():
    global _ensemble
    fig = training_curve(_train_state.train_losses, _train_state.val_losses)
    # Reload ensemble if training just finished
    if _train_state.phase == "done" and _ensemble is None:
        _ensemble = _load_ensemble()
    status = (
        f"**Phase:** {_train_state.phase}  ·  "
        f"**Epoch:** {_train_state.epoch}/{_train_state.total_epochs}  ·  "
        f"**Transitions:** {_train_state.n_transitions:,}\n\n"
        f"`{_train_state.status}`"
    )
    return fig, status


# ── Tab 3: Imagination Rollout ────────────────────────────────────────────────


def cb_imagination(
    start_preset,
    horizon,
    action_val,
    progress: gr.Progress = gr.Progress(),
):
    model = _best_model()
    if model is None:
        return empty_fig("No model. Train one first."), "*No model available.*"

    PRESETS = {
        "⬇️ Hanging down (θ=π)": (np.pi, 0.0),
        "➡️ Horizontal (θ=π/2)": (np.pi / 2, 0.0),
        "⬆️ Near upright (θ=0.3)": (0.3, 0.5),
        "🌀 Spinning fast": (np.pi / 3, 6.0),
    }
    theta0, vel0 = PRESETS[start_preset]
    init_state = np.array([np.cos(theta0), np.sin(theta0), vel0], dtype=np.float32)

    # Run real environment with constant torque
    progress(0.1, desc="Running real environment…")
    env = gym.make("Pendulum-v1")
    env.reset()
    env.unwrapped.state = np.array([theta0, vel0])
    real_traj = [init_state.copy()]
    obs = init_state
    for _ in range(int(horizon)):
        obs, _, term, trunc, _ = env.step(np.array([float(action_val)]))
        real_traj.append(obs.copy())
        if term or trunc:
            env.reset()
    env.close()
    real_traj = np.array(real_traj[: int(horizon) + 1])

    # Roll out in model
    progress(0.5, desc="Rolling out in model…")
    imag_traj = [init_state.copy()]
    imag_stds = [np.zeros(3)]
    current = init_state.copy()

    model.eval() if hasattr(model, "eval") else None

    with torch.no_grad():
        for _ in range(int(horizon)):
            s = torch.FloatTensor(current).unsqueeze(0)
            a = torch.FloatTensor([[float(action_val)]])
            if isinstance(model, EnsembleDynamics):
                mean, std = model.forward_all(s, a)
                next_s = mean.squeeze(0).numpy()
                next_std = std.squeeze(0).numpy()
            else:
                next_s = model(s, a).squeeze(0).numpy()
                next_std = np.zeros(3)
            imag_traj.append(next_s)
            imag_stds.append(next_std)
            current = next_s

    imag_traj = np.array(imag_traj)
    imag_stds = np.array(imag_stds)

    progress(0.8, desc="Rendering…")
    fig = imagination_comparison(real_traj, imag_traj, imag_stds)

    mse_final = float(np.mean((real_traj[-1] - imag_traj[-1]) ** 2))
    mse_total = float(np.mean((real_traj - imag_traj) ** 2))
    status = f"""
**Horizon:** {horizon} steps  ·  **Start:** {start_preset}  ·  **Action:** {action_val:.2f} N·m

| Metric | Value |
|---|---|
| **Final MSE** | `{mse_final:.5f}` |
| **Mean MSE over horizon** | `{mse_total:.5f}` |
| **Model type** | `{"Ensemble" if isinstance(model, EnsembleDynamics) else "Single"}` |

> Small MSE = model closely tracks reality.
> Error typically grows with horizon — this is the fundamental MBRL challenge.
"""
    progress(1.0)
    return fig, status


# ── Tab 4: MPC Control ────────────────────────────────────────────────────────


def cb_run_mpc(horizon, n_samples, max_steps, progress: gr.Progress = gr.Progress()):
    model = _best_model()
    if model is None:
        return None, empty_fig("No model. Train one first."), "*No model.*"

    progress(0.1, desc="Running MPC episode…")
    mpc_res = run_mpc_episode(
        model,
        horizon=int(horizon),
        n_samples=int(n_samples),
        max_steps=int(max_steps),
        render=True,
        seed=0,
    )

    progress(0.6, desc="Running random baseline…")
    rand_res = run_random_episode(max_steps=int(max_steps), seed=0)

    progress(0.75, desc="Building GIF…")
    gif_path = make_mpc_gif(mpc_res["frames"], fps=20)

    progress(0.9, desc="Building charts…")
    chart = mpc_episode_chart(mpc_res, rand_res)

    improvement = mpc_res["total_reward"] - rand_res["total_reward"]
    status = f"""
### MPC Episode Complete

| Metric | MPC | Random | Improvement |
|---|---|---|---|
| **Total Reward** | `{mpc_res["total_reward"]:.1f}` | `{rand_res["total_reward"]:.1f}` | `{improvement:+.1f}` |
| **Steps** | `{mpc_res["steps"]}` | `{len(rand_res["rewards"])}` | — |
| **Planning horizon** | `{horizon}` | — | — |
| **Candidates/step** | `{n_samples}` | — | — |

> MPC samples {n_samples} random action sequences per step, rolls them out in the
> learned model, and executes the first action of the best one.
"""
    progress(1.0)
    return gif_path, chart, status


# ── Tab 5: Uncertainty ────────────────────────────────────────────────────────


def cb_uncertainty(progress: gr.Progress = gr.Progress()):
    global _ensemble
    _ensemble = _ensemble or _load_ensemble()
    if _ensemble is None:
        return empty_fig(
            "Train an ensemble first (Training tab)."
        ), "*Ensemble not trained yet.*"
    progress(0.3, desc="Computing uncertainty map…")
    fig = uncertainty_heatmap(_ensemble, n_grid=35)
    progress(1.0)
    return fig, "Uncertainty map computed from 5-model ensemble variance."


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0e1117 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1150px !important; margin: 0 auto !important; }

.mbrl-header {
    border-bottom: 1px solid #1c2333;
    padding: 1.5rem 1.5rem 1rem;
}
.mbrl-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.2rem, 3vw, 1.8rem);
    font-weight: 700; color: #7c3aed; margin: 0;
}
.mbrl-sub { color: #4a5568; font-size: 0.85rem; margin-top: 0.3rem; }
.mbrl-badges { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.7rem; }
.mbrl-badge {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    letter-spacing:0.08em; padding:3px 10px; border-radius:3px;
    text-transform:uppercase;
}
.b-purple { background:#1a0d30; color:#7c3aed; border:1px solid #4a1a8a; }
.b-cyan   { background:#071e26; color:#06b6d4; border:1px solid #0a4a5c; }
.b-green  { background:#061a0e; color:#10b981; border:1px solid #0a4a26; }
.b-amber  { background:#1e1206; color:#f59e0b; border:1px solid #5c3a06; }

.tab-nav { border-bottom:1px solid #1c2333 !important; background:transparent !important; }
.tab-nav button {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.72rem !important; letter-spacing:0.05em !important;
    color:#4a5568 !important; background:transparent !important;
    border:none !important; padding:0.7rem 1.1rem !important;
    text-transform:uppercase !important;
}
.tab-nav button.selected { color:#7c3aed !important; border-bottom:2px solid #7c3aed !important; }

button.primary {
    font-family:'JetBrains Mono',monospace !important; font-weight:600 !important;
    background:linear-gradient(135deg,#2d1b69,#3d2a8a) !important;
    color:#c4b5fd !important; border:1px solid #7c3aed !important;
    border-radius:5px !important; transition:all 0.2s !important;
}
button.primary:hover { box-shadow:0 0 14px rgba(124,58,237,0.4) !important; }
button.secondary {
    font-family:'JetBrains Mono',monospace !important;
    background:#161b27 !important; color:#06b6d4 !important;
    border:1px solid #06b6d4 !important; border-radius:5px !important;
}
button.stop {
    background:#180a0a !important; color:#ef4444 !important;
    border:1px solid #ef4444 !important; border-radius:5px !important;
}

label span, .gradio-container label {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.7rem !important; color:#4a5568 !important;
    text-transform:uppercase !important; letter-spacing:0.06em !important;
}
input[type=range] { -webkit-appearance:none; height:3px;
                    background:#1c2333; border-radius:2px; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:14px; height:14px;
    border-radius:50%; background:#7c3aed; cursor:pointer;
    border:2px solid #0e1117;
}
textarea, .gradio-container textarea {
    font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
    background:#060a12 !important; color:#7c3aed !important;
    border:1px solid #1c2333 !important; border-radius:4px !important;
}
.gradio-container h2 { color:#7c3aed !important; font-family:'JetBrains Mono',monospace !important; }
.gradio-container h3 { color:#06b6d4 !important; }
.gradio-container p  { color:#64748b !important; }
table { width:100%; border-collapse:collapse; }
th { background:#161b27; color:#7c3aed; font-family:'JetBrains Mono',monospace;
     font-size:0.7rem; text-align:left; padding:7px 12px;
     border-bottom:1px solid #1c2333; text-transform:uppercase; }
td { padding:7px 12px; border-bottom:1px solid #0e1117;
     color:#e2e8f0; font-size:0.83rem; }
code { font-family:'JetBrains Mono',monospace; background:#161b27;
       color:#f59e0b; padding:1px 5px; border-radius:3px; }
blockquote { border-left:3px solid #7c3aed; padding:0.6rem 1rem;
             background:#161b27; border-radius:0 4px 4px 0; margin:0.5rem 0; }
footer { display:none !important; }
.gradio-container .block { background:transparent !important; border:none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MBRL Pendulum Playground") as demo:
    gr.HTML("""
    <div class="mbrl-header">
        <div class="mbrl-title">🌀 MBRL Pendulum Playground</div>
        <div class="mbrl-sub">Model-Based Reinforcement Learning · Neural Dynamics · MPC Planning</div>
        <div class="mbrl-badges">
            <span class="mbrl-badge b-purple">● Neural Dynamics Model</span>
            <span class="mbrl-badge b-cyan">● Imagination Rollout</span>
            <span class="mbrl-badge b-green">● MPC Random Shooting</span>
            <span class="mbrl-badge b-amber">● Ensemble Uncertainty</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Interactive Explorer
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🎮 EXPLORER"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">
                    SINGLE-STEP STATE EXPLORER
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Set a pendulum state and action. See what the real physics engine predicts
                    vs what the learned dynamics model imagines.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    gr.HTML(
                        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#4a5568;text-transform:uppercase;margin-bottom:0.4rem;\">PENDULUM STATE</div>"
                    )
                    c_th = gr.Slider(
                        -1,
                        1,
                        value=1.0,
                        step=0.01,
                        label="cos θ  (1=upright, -1=hanging)",
                    )
                    s_th = gr.Slider(
                        -1, 1, value=0.0, step=0.01, label="sin θ  (0=upright)"
                    )
                    vel = gr.Slider(
                        -8, 8, value=0.0, step=0.1, label="Angular velocity  (rad/s)"
                    )

                    gr.HTML(
                        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#4a5568;text-transform:uppercase;margin:0.6rem 0 0.4rem;\">ACTION</div>"
                    )
                    trq = gr.Slider(
                        -2, 2, value=0.0, step=0.1, label="Torque applied  (N·m)"
                    )
                    btn_explore = gr.Button("▶ PREDICT", variant="primary")

                    gr.HTML("""
                    <div style="background:#161b27;border:1px solid #1c2333;border-radius:6px;padding:0.8rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#4a5568;text-transform:uppercase;margin-bottom:0.4rem;">STATE GUIDE</div>
                        <div style="font-size:0.8rem;color:#64748b;line-height:1.7;">
                            <div>⬆️ <strong style="color:#e2e8f0">Upright:</strong> cos=1, sin=0</div>
                            <div>⬇️ <strong style="color:#e2e8f0">Hanging:</strong> cos=-1, sin=0</div>
                            <div>➡️ <strong style="color:#e2e8f0">Right:</strong> cos=0, sin=1</div>
                            <div>⬅️ <strong style="color:#e2e8f0">Left:</strong> cos=0, sin=-1</div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=1):
                    exp_frame = gr.Image(label="Pendulum State", height=280)

                with gr.Column(scale=2):
                    exp_fig = gr.Plot(label="Real vs Predicted Next State")
                    exp_table = gr.Markdown("*Click Predict to compare real vs model.*")

            for inp in [c_th, s_th, vel, trq]:
                inp.change(
                    cb_explore, [c_th, s_th, vel, trq], [exp_frame, exp_fig, exp_table]
                )
            btn_explore.click(
                cb_explore, [c_th, s_th, vel, trq], [exp_frame, exp_fig, exp_table]
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Train Dynamics Model
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🧠 TRAIN MODEL"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">
                    ENSEMBLE DYNAMICS MODEL TRAINING
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Collect real transitions by running a random policy, then train
                    an ensemble of 5 neural networks to predict state transitions.
                    The ensemble variance gives us uncertainty estimates.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    train_steps = gr.Slider(
                        500, 8000, value=3000, step=500, label="Transitions to collect"
                    )
                    train_epochs = gr.Slider(
                        20, 150, value=80, step=10, label="Training epochs"
                    )
                    with gr.Row():
                        btn_train = gr.Button("▶ START TRAINING", variant="primary")
                        btn_stop = gr.Button("⏹ STOP", variant="stop")
                    btn_refresh = gr.Button("🔄 REFRESH", variant="secondary")
                    train_msg = gr.Textbox(label="Status", lines=2, interactive=False)

                    gr.HTML("""
                    <div style="background:#161b27;border:1px solid #1c2333;border-radius:6px;padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#4a5568;text-transform:uppercase;margin-bottom:0.6rem;">ENSEMBLE DESIGN</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;">
                            <div style="color:#4a5568;font-size:0.78rem;">Models</div>
                            <div style="color:#7c3aed;font-family:'JetBrains Mono',monospace;font-size:0.78rem;">5 × MLP (128-128-64)</div>
                            <div style="color:#4a5568;font-size:0.78rem;">Input</div>
                            <div style="color:#7c3aed;font-family:'JetBrains Mono',monospace;font-size:0.78rem;">[s, a] → 4D</div>
                            <div style="color:#4a5568;font-size:0.78rem;">Output</div>
                            <div style="color:#7c3aed;font-family:'JetBrains Mono',monospace;font-size:0.78rem;">s' → 3D</div>
                            <div style="color:#4a5568;font-size:0.78rem;">Uncertainty</div>
                            <div style="color:#7c3aed;font-family:'JetBrains Mono',monospace;font-size:0.78rem;">Ensemble std</div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=2):
                    train_status_md = gr.Markdown(
                        "*Start training to see live metrics.*"
                    )
                    train_fig = gr.Plot(label="Training Curves")

            btn_train.click(cb_start_train, [train_steps, train_epochs], [train_msg])
            btn_stop.click(cb_stop_train, outputs=[train_msg])
            btn_refresh.click(cb_refresh_train, outputs=[train_fig, train_status_md])

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Imagination Rollout
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🔮 IMAGINATION"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">
                    MULTI-STEP IMAGINATION vs REALITY
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    The model "imagines" a trajectory by chaining its own predictions.
                    Watch how errors compound over the planning horizon —
                    the core challenge of MBRL.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    imag_preset = gr.Dropdown(
                        [
                            "⬇️ Hanging down (θ=π)",
                            "➡️ Horizontal (θ=π/2)",
                            "⬆️ Near upright (θ=0.3)",
                            "🌀 Spinning fast",
                        ],
                        value="⬇️ Hanging down (θ=π)",
                        label="Starting state",
                    )
                    imag_horizon = gr.Slider(
                        5, 50, value=25, step=5, label="Rollout horizon (steps)"
                    )
                    imag_action = gr.Slider(
                        -2, 2, value=0.0, step=0.1, label="Constant action (torque)"
                    )
                    btn_imag = gr.Button("🔮 IMAGINE", variant="primary")
                    imag_status = gr.Markdown("")

                with gr.Column(scale=3):
                    imag_fig = gr.Plot(label="Real vs Imagined Trajectory")

            btn_imag.click(
                cb_imagination,
                [imag_preset, imag_horizon, imag_action],
                [imag_fig, imag_status],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 4 — MPC Control
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🎯 MPC CONTROL"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">
                    MODEL PREDICTIVE CONTROL  ·  RANDOM SHOOTING
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    At each step, sample many random action sequences, roll them out
                    in the learned model, and execute the best one.
                    This is MBRL in action — planning without learning an explicit policy.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    mpc_horizon = gr.Slider(
                        5, 30, value=15, step=5, label="Planning horizon  (steps)"
                    )
                    mpc_samples = gr.Slider(
                        64,
                        1024,
                        value=512,
                        step=64,
                        label="Candidate sequences per step",
                    )
                    mpc_maxsteps = gr.Slider(
                        50, 300, value=200, step=50, label="Episode length  (steps)"
                    )
                    btn_mpc = gr.Button("🎯 RUN MPC EPISODE", variant="primary")

                    gr.HTML("""
                    <div style="background:#161b27;border:1px solid #1c2333;border-radius:6px;padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#4a5568;text-transform:uppercase;margin-bottom:0.5rem;">HOW RANDOM SHOOTING WORKS</div>
                        <ol style="color:#64748b;font-size:0.8rem;line-height:1.8;margin:0;padding-left:1.2rem;">
                            <li>Sample <em>N</em> random torque sequences of length <em>H</em></li>
                            <li>Roll each out in the learned model</li>
                            <li>Compute total discounted reward for each</li>
                            <li>Execute <strong style="color:#e2e8f0">first action</strong> of the best sequence</li>
                            <li>Observe real next state, repeat</li>
                        </ol>
                    </div>
                    """)

                with gr.Column(scale=1):
                    mpc_gif = gr.Image(
                        label="MPC Episode Replay", type="filepath", height=320
                    )
                    mpc_status = gr.Markdown("")

            mpc_chart = gr.Plot(label="Episode Analysis")

            btn_mpc.click(
                cb_run_mpc,
                [mpc_horizon, mpc_samples, mpc_maxsteps],
                [mpc_gif, mpc_chart, mpc_status],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 5 — Uncertainty Map
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📊 UNCERTAINTY"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">
                    ENSEMBLE EPISTEMIC UNCERTAINTY
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Where does the ensemble disagree? High variance = regions the model
                    hasn't seen enough data from. Brighter = more uncertain.
                </div>
            </div>
            """)

            with gr.Row():
                btn_unc = gr.Button("📊 COMPUTE UNCERTAINTY MAP", variant="primary")
                unc_status = gr.Markdown("*Train an ensemble first, then compute.*")

            unc_fig = gr.Plot(label="Uncertainty Heatmap (θ vs angular velocity)")
            btn_unc.click(cb_uncertainty, outputs=[unc_fig, unc_status])

        # ══════════════════════════════════════════════════════════════════
        # Tab 6 — How MBRL Works
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📚 HOW IT WORKS"):
            gr.Markdown("""
## Model-Based RL — The Core Idea

Standard (model-free) RL learns a policy directly from environment interactions.
**MBRL learns a model of the environment first**, then uses that model to plan.

```
Model-Free RL:   environment → (s, a, r, s') → policy update
MBRL:            environment → (s, a, s') → world model
                 world model → planning (MPC) → action
```

**The key advantage:** A good model can generate unlimited synthetic experience.
One real transition can be "replayed" thousands of times through the model.

---

## The Pendulum Task

| Property | Value |
|---|---|
| **State** | `[cos θ, sin θ, θ̇]` — 3D continuous |
| **Action** | Torque ∈ [−2, 2] N·m — 1D continuous |
| **Reward** | `−(θ² + 0.1θ̇² + 0.001τ²)` — penalises deviation from upright |
| **Goal** | Swing up and balance at θ = 0 |

Why `cos θ` and `sin θ` instead of θ directly?
Because θ is cyclic — it wraps at ±π. The `cos/sin` encoding is smooth and
avoids a discontinuity the network would struggle to learn.

---

## The Dynamics Model

The model learns: **f(s, a) → s'**

Given current state `s` and action `a`, predict next state `s'`.

**Architecture:** 4 → 128 → 128 → 64 → 3 (SiLU activations)

Training objective:
```
L = (1/N) Σ ||f(sᵢ, aᵢ) − sᵢ'||²
```

Minimising MSE on real transitions collected from a random policy.

---

## Ensemble = Uncertainty

We train **5 independent models** (ensemble) on bootstrapped subsets of data.

```
Prediction mean  = average over 5 models     → what the model thinks
Prediction std   = variance across 5 models  → how uncertain the model is
```

High uncertainty → model hasn't seen similar states → plan conservatively.

---

## Model Predictive Control (Random Shooting)

```python
for each real step:
    candidates = sample_random_action_seqs(N=512, horizon=H)
    for each candidate:
        simulate in model → get trajectory
        compute sum of rewards
    best = argmax(total_reward)
    execute best_sequence[0] in real environment
    observe real next state
    repeat
```

**Key insight:** We never learn a policy network. The model IS the policy.
Planning replaces policy learning — the trade-off is compute per step.

---

## Why Imagination Fails Over Long Horizons

A model with 1% per-step error compounds:
```
After 10 steps:  ~10% total error
After 50 steps:  ~40% total error  (error compounds faster than linearly)
After 100 steps: model trajectory diverges completely
```

This is why MPC uses a **short receding horizon** (H=15-20) rather than
planning the entire episode at once. Re-planning at every step corrects drift.

---

## This vs Model-Free RL

| | MBRL (this app) | Model-Free (e.g. SAC) |
|---|---|---|
| **Data efficiency** | High — model can replay | Low — needs many real steps |
| **Compute** | High at inference (MPC) | Low at inference |
| **Planning** | Explicit (MPC) | Implicit (in policy network) |
| **Uncertainty** | Quantifiable (ensemble) | Hard to measure |
| **Long horizons** | Problematic (compounding error) | Handles naturally |
""")

    gr.HTML("""
    <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                color:#1c2333;padding:1.5rem 0 0.5rem;border-top:1px solid #161b27;
                letter-spacing:0.1em;text-transform:uppercase;">
        MBRL · Neural Dynamics · MPC Random Shooting · Ensemble Uncertainty · Gymnasium · PyTorch
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
