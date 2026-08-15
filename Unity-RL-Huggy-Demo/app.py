"""
Unity RL Huggy Demo — Interactive 3D Reinforcement Learning Showcase
A Gradio application presenting Huggy, a Unity ML-Agents trained character,
alongside a full educational guide to how deep RL works in 3D game engines.
"""

import gradio as gr

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0d14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

.tab-nav { border-bottom: 1px solid #1e293b !important; background: transparent !important; }
.tab-nav button {
    font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important;
    font-weight: 500 !important; color: #475569 !important;
    background: transparent !important; border: none !important;
    padding: 0.7rem 1.2rem !important;
}
.tab-nav button.selected { color: #f97316 !important; border-bottom: 2px solid #f97316 !important; }

.gradio-container h2 { color: #f97316 !important; font-family: 'JetBrains Mono', monospace !important; }
.gradio-container h3 { color: #fb923c !important; }
.gradio-container p, .gradio-container li { color: #94a3b8 !important; line-height: 1.7 !important; }
strong { color: #e2e8f0 !important; }
table { width: 100%; border-collapse: collapse; }
th { background: #111827; color: #f97316; font-family: 'JetBrains Mono', monospace;
     font-size: 0.72rem; text-align: left; padding: 8px 12px;
     border-bottom: 1px solid #1e293b; text-transform: uppercase; }
td { padding: 8px 12px; border-bottom: 1px solid #0a0d14; color: #e2e8f0; font-size: 0.85rem; }
code { font-family: 'JetBrains Mono', monospace; background: #111827;
       color: #fb923c; padding: 1px 6px; border-radius: 4px; }
blockquote { border-left: 3px solid #f97316; padding: 0.5rem 1rem;
             background: #111827; border-radius: 0 6px 6px 0; margin: 0.5rem 0; }

footer { display: none !important; }
.gradio-container .block { background: transparent !important; border: none !important; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Unity RL — Huggy Demo", css=CSS) as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 2rem 1rem 1.2rem;
                border-bottom: 1px solid #1e293b; margin-bottom: 1rem;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                    color:#475569; text-transform:uppercase; letter-spacing:0.15em;
                    margin-bottom:0.5rem;">
            Unity ML-Agents · Deep Reinforcement Learning · WebGL
        </div>
        <div style="font-size: clamp(1.6rem, 4vw, 2.4rem); font-weight: 700;
                    background: linear-gradient(135deg, #f97316, #fb923c, #fbbf24);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin: 0 0 0.4rem;">
            Huggy the RL Dog
        </div>
        <div style="color:#475569; font-size:0.9rem;">
            A 3D character that learned to fetch entirely through trial and error —
            no rules, no hardcoded movement, just rewards.
        </div>
    </div>
    """)

    with gr.Tabs():
        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Play
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🎮 Play with Huggy"):
            gr.HTML("""
            <div style="background:#111827; border:1px solid #1e293b; border-radius:10px;
                        padding:0.9rem 1.2rem; margin-bottom:1rem;
                        display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                <div style="font-size:1.3rem;">🐕</div>
                <div>
                    <div style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">
                        How to play
                    </div>
                    <div style="color:#64748b; font-size:0.82rem; margin-top:0.2rem;">
                        Click anywhere in the scene to throw the stick.
                        Huggy will run to fetch it. No installation needed — runs entirely in your browser.
                    </div>
                </div>
                <div style="margin-left:auto; display:flex; gap:0.5rem; flex-wrap:wrap;">
                    <span style="background:rgba(249,115,22,0.1); color:#f97316;
                                 border:1px solid rgba(249,115,22,0.25); border-radius:20px;
                                 padding:3px 10px; font-size:0.65rem; font-family:'JetBrains Mono',monospace;
                                 text-transform:uppercase; letter-spacing:0.08em;">
                        WebGL
                    </span>
                    <span style="background:rgba(249,115,22,0.1); color:#f97316;
                                 border:1px solid rgba(249,115,22,0.25); border-radius:20px;
                                 padding:3px 10px; font-size:0.65rem; font-family:'JetBrains Mono',monospace;
                                 text-transform:uppercase; letter-spacing:0.08em;">
                        PPO Policy
                    </span>
                    <span style="background:rgba(249,115,22,0.1); color:#f97316;
                                 border:1px solid rgba(249,115,22,0.25); border-radius:20px;
                                 padding:3px 10px; font-size:0.65rem; font-family:'JetBrains Mono',monospace;
                                 text-transform:uppercase; letter-spacing:0.08em;">
                        Real-time physics
                    </span>
                </div>
            </div>
            """)

            gr.HTML("""
            <div style="border-radius:12px; overflow:hidden; border:1px solid #1e293b;
                        background:#000; box-shadow: 0 8px 32px rgba(0,0,0,0.5);">
                <iframe
                    src="https://huggingface.co/spaces/ThomasSimonini/Huggy/embed/"
                    width="100%"
                    height="600px"
                    frameborder="0"
                    allow="autoplay; fullscreen; vr"
                    style="display:block;">
                </iframe>
            </div>
            """)

            gr.HTML("""
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; margin-top:1rem;">
                <div style="background:#111827; border:1px solid #1e293b; border-radius:8px; padding:1rem;">
                    <div style="font-size:1.4rem; margin-bottom:0.4rem;">🎯</div>
                    <div style="font-weight:600; color:#e2e8f0; font-size:0.85rem;">Sparse reward</div>
                    <div style="color:#64748b; font-size:0.78rem; margin-top:0.2rem;">
                        Huggy only gets reward when close to the stick.
                        No guidance on how to move or balance.
                    </div>
                </div>
                <div style="background:#111827; border:1px solid #1e293b; border-radius:8px; padding:1rem;">
                    <div style="font-size:1.4rem; margin-bottom:0.4rem;">🦴</div>
                    <div style="font-weight:600; color:#e2e8f0; font-size:0.85rem;">Joint torques</div>
                    <div style="color:#64748b; font-size:0.78rem; margin-top:0.2rem;">
                        The policy controls raw joint forces on a simulated
                        dog body — not pre-animated movements.
                    </div>
                </div>
                <div style="background:#111827; border:1px solid #1e293b; border-radius:8px; padding:1rem;">
                    <div style="font-size:1.4rem; margin-bottom:0.4rem;">🔁</div>
                    <div style="font-weight:600; color:#e2e8f0; font-size:0.85rem;">Millions of steps</div>
                    <div style="color:#64748b; font-size:0.78rem; margin-top:0.2rem;">
                        Trained across hundreds of parallel Unity simulations
                        running simultaneously at high speed.
                    </div>
                </div>
            </div>
            """)

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — How It Works
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🧠 How It Was Trained"):
            gr.Markdown("""
## From Zero to Fetch — How Huggy Learned

Huggy was not programmed with movement rules. No one wrote code that says
"move left leg forward when right leg is back." Instead, the character
discovered locomotion, balance, and targeting entirely through a process
of trial and error called reinforcement learning.

---

## The Algorithm: Proximal Policy Optimization (PPO)

PPO is a policy gradient algorithm that trains a neural network to map
observations to actions. At every physics step:

1. The network receives an observation (joint angles, velocities, stick position)
2. It outputs a **Gaussian distribution** over joint torque values
3. A torque is sampled from that distribution and applied to the body
4. The simulation steps forward and a reward is computed
5. The network updates to make good actions more likely

The "proximal" part prevents the network from updating too aggressively.
Each gradient step is constrained so the new policy cannot differ from
the old one by more than a fixed amount:

```
actor_loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
```

where `ratio = new_policy(action) / old_policy(action)`.
If ratio is clipped, the gradient contribution from that sample is zeroed out.
This keeps training stable even with noisy reward signals.

---

## The Observation Space

At each step Huggy's neural network receives:

| Input | What it tells the agent |
|---|---|
| Joint angles (all limbs) | Current body posture |
| Joint angular velocities | How fast each joint is moving |
| Body position and velocity | Where Huggy is and how fast it is moving |
| Body orientation | Is Huggy tilted or upright? |
| Stick position (relative) | Where is the target? |

These observations are raw numbers from the physics engine — no
visual processing, no camera. The agent learns entirely from proprioception
(body sense) plus target position.

---

## The Reward Function

The reward signal is deliberately minimal:

```
reward = max(0,  1 - distance_to_stick / max_distance)
```

Huggy gets a higher reward the closer it is to the stick. When it
touches the stick the episode ends with a success bonus.

There is no reward for:
- Walking correctly
- Maintaining balance
- Being efficient
- Avoiding falling

Everything Huggy knows about moving gracefully was discovered
because graceful movement happens to be the fastest way to
consistently reach a thrown stick.

---

## The Training Infrastructure

Training happens in **Unity ML-Agents**, a toolkit that connects
Unity's physics engine to Python training code:

```
Unity physics sim  ←→  Python PPO trainer
(joint forces)          (neural network)
```

To speed things up, hundreds of parallel copies of the simulation
run simultaneously. Each copy runs a different episode. All episodes
feed into the same shared replay buffer and policy update.

This is why training takes hours rather than years — even though
each individual episode is slow in real time, parallelism lets
the trainer process thousands of experiences per second.

---

## From Training to Browser

After training, the policy is a PyTorch model — just numbers
in a matrix. Unity ML-Agents exports this to a format that runs
in WebGL via browser-native machine learning APIs.

The demo above is running that exported policy **in your browser
tab right now**. The neural network is making real-time decisions
at every physics step, computing joint torques from observations,
just as it did during training — just without the Python training
loop around it.

---

## Why This Matters

Every other project in this portfolio uses simple 2D environments
or low-dimensional state spaces. Huggy shows what RL looks like
when the environment is a full 3D physics simulation:

- **Continuous, high-dimensional actions** (joint torques, not grid moves)
- **Physically simulated body** with dozens of coupled degrees of freedom
- **Sparse, delayed reward** — no step-by-step guidance
- **Real-time inference** at 60 fps in a browser

The algorithm (PPO) is identical to the Swarm Architect MARL
and Warehouse Sim projects in this repository. The only difference
is the environment complexity. This is one of the most powerful
ideas in modern RL: **a single general-purpose algorithm scales
from grid mazes to physically simulated 3D creatures.**
""")

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — About This Project
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📋 About"):
            gr.HTML("""
            <div style="max-width:720px; margin:0 auto;">

            <div style="background:#111827; border:1px solid #1e293b; border-radius:10px;
                        padding:1.4rem; margin-bottom:1rem;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                            color:#475569; text-transform:uppercase; letter-spacing:0.1em;
                            margin-bottom:0.8rem;">This Project</div>
                <div style="color:#94a3b8; font-size:0.9rem; line-height:1.7;">
                    This space embeds the official Huggy demo created by Thomas Simonini
                    as part of the Hugging Face Deep RL Course. It is presented here as
                    part of a portfolio of 11 end-to-end RL projects that together cover
                    the full range of modern reinforcement learning techniques.
                </div>
            </div>

            <div style="background:#111827; border:1px solid #1e293b; border-radius:10px;
                        padding:1.4rem; margin-bottom:1rem;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                            color:#475569; text-transform:uppercase; letter-spacing:0.1em;
                            margin-bottom:0.8rem;">Tech Stack</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
                    <div style="color:#64748b; font-size:0.82rem;">Game Engine</div>
                    <div style="color:#e2e8f0; font-size:0.82rem; font-family:'JetBrains Mono',monospace;">Unity 2022 + ML-Agents</div>
                    <div style="color:#64748b; font-size:0.82rem;">RL Algorithm</div>
                    <div style="color:#e2e8f0; font-size:0.82rem; font-family:'JetBrains Mono',monospace;">PPO (on-policy)</div>
                    <div style="color:#64748b; font-size:0.82rem;">Action space</div>
                    <div style="color:#e2e8f0; font-size:0.82rem; font-family:'JetBrains Mono',monospace;">Continuous (joint torques)</div>
                    <div style="color:#64748b; font-size:0.82rem;">Observation space</div>
                    <div style="color:#e2e8f0; font-size:0.82rem; font-family:'JetBrains Mono',monospace;">Proprioceptive + target</div>
                    <div style="color:#64748b; font-size:0.82rem;">Runtime</div>
                    <div style="color:#e2e8f0; font-size:0.82rem; font-family:'JetBrains Mono',monospace;">WebGL (browser-native)</div>
                    <div style="color:#64748b; font-size:0.82rem;">Interface</div>
                    <div style="color:#e2e8f0; font-size:0.82rem; font-family:'JetBrains Mono',monospace;">Gradio 6</div>
                </div>
            </div>

            <div style="background:#111827; border:1px solid #1e293b; border-radius:10px;
                        padding:1.4rem; margin-bottom:1rem;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                            color:#475569; text-transform:uppercase; letter-spacing:0.1em;
                            margin-bottom:0.8rem;">Want to train Huggy yourself?</div>
                <div style="color:#94a3b8; font-size:0.85rem; line-height:1.7;">
                    The full training pipeline is open source. You need Unity 2022,
                    the ML-Agents package, and Python with mlagents installed.
                </div>
                <div style="background:#0a0d14; border-radius:6px; padding:0.8rem;
                            font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                            color:#f97316; margin-top:0.8rem;">
                    pip install mlagents<br>
                    mlagents-learn config/ppo/Huggy.yaml --run-id=my_huggy
                </div>
                <div style="color:#64748b; font-size:0.8rem; margin-top:0.6rem;">
                    Full walkthrough in the
                    <a href="https://huggingface.co/learn/deep-rl-course/unit1/introduction"
                       style="color:#f97316;">Hugging Face Deep RL Course</a>.
                </div>
            </div>

            <div style="background:#111827; border:1px solid #1e293b; border-radius:10px;
                        padding:1.4rem;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                            color:#475569; text-transform:uppercase; letter-spacing:0.1em;
                            margin-bottom:0.8rem;">Rest of the Portfolio</div>
                <div style="color:#94a3b8; font-size:0.85rem; line-height:1.7; margin-bottom:0.8rem;">
                    This is one of 11 RL projects in this portfolio.
                    The others implement algorithms from scratch — Q-Learning,
                    SAC, DQN, IPPO, MBRL, RLHF, HMM, and more.
                </div>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                    <a href="https://huggingface.co/spaces/Dash10107/rocket-lander-sac"
                       style="background:rgba(249,115,22,0.1); color:#f97316;
                              border:1px solid rgba(249,115,22,0.2); border-radius:6px;
                              padding:4px 10px; font-size:0.75rem; text-decoration:none;">
                       🚀 Rocket Lander SAC
                    </a>
                    <a href="https://huggingface.co/spaces/Dash10107/marl-warehouse-sim"
                       style="background:rgba(249,115,22,0.1); color:#f97316;
                              border:1px solid rgba(249,115,22,0.2); border-radius:6px;
                              padding:4px 10px; font-size:0.75rem; text-decoration:none;">
                       📦 Warehouse MARL
                    </a>
                    <a href="https://huggingface.co/spaces/Dash10107/mbrl-pendulum-playground"
                       style="background:rgba(249,115,22,0.1); color:#f97316;
                              border:1px solid rgba(249,115,22,0.2); border-radius:6px;
                              padding:4px 10px; font-size:0.75rem; text-decoration:none;">
                       🌀 MBRL Pendulum
                    </a>
                    <a href="https://huggingface.co/spaces/Dash10107/rl_maze_solver"
                       style="background:rgba(249,115,22,0.1); color:#f97316;
                              border:1px solid rgba(249,115,22,0.2); border-radius:6px;
                              padding:4px 10px; font-size:0.75rem; text-decoration:none;">
                       🤖 RL Maze Solver
                    </a>
                </div>
            </div>

            </div>
            """)

    gr.HTML("""
    <div style="text-align:center; font-family:'JetBrains Mono',monospace; font-size:0.62rem;
                color:#1e293b; padding:1.5rem 0 0.5rem; border-top:1px solid #111827;
                margin-top:1rem; letter-spacing:0.1em; text-transform:uppercase;">
        Unity ML-Agents · PPO · WebGL · Huggy by Thomas Simonini · Gradio
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
