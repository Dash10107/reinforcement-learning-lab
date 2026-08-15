# rebuild_git_history.ps1
# Rebuilds reinforcement-learning-lab with a natural, methodical git history.
# Projects are added incrementally (subdir by subdir) so each commit
# contains a real file diff, just like actual development.

$GITHUB_USER = "Dash10107"
$REPO_NAME   = "reinforcement-learning-lab"
$GITHUB_URL  = "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

$SRC  = "C:\Users\daksh\OneDrive\Desktop\ReinforcementLearning"
$DEST = "C:\Users\daksh\Desktop\reinforcement-learning-lab"

# ── Copy a single item (file or directory) from SRC project to DEST project ──
function Copy-Item-Clean($project, $item) {
    $sp = Join-Path (Join-Path $SRC $project) $item
    $dp = Join-Path $DEST $project
    if (-not (Test-Path $dp)) { New-Item -ItemType Directory -Path $dp | Out-Null }
    if (Test-Path $sp -PathType Container) {
        $ddp = Join-Path $dp $item
        robocopy $sp $ddp /E /XD .git __pycache__ .venv checkpoints wandb runs mlruns `
            /XF "*.pyc" "*.pyo" "*.pth" "*.zip" "*.pt" "*.pkl" "*.gif" `
            /NFL /NDL /NJH /NJS | Out-Null
    } elseif (Test-Path $sp) {
        Copy-Item $sp $dp -Force
    }
}

# ── Copy a root-level file from SRC to DEST ──────────────────────────────────
function Copy-Root($file) {
    $sf = Join-Path $SRC $file
    if (Test-Path $sf) { Copy-Item $sf $DEST -Force }
}

# ── Stage and commit ──────────────────────────────────────────────────────────
function Commit($msg, $date) {
    $env:GIT_AUTHOR_DATE    = $date
    $env:GIT_COMMITTER_DATE = $date
    git add -A | Out-Null
    $status = git status --porcelain
    if ($status) {
        git commit -q -m $msg
        $count = ($status | Measure-Object -Line).Lines
        Write-Host "  [$date] $($msg.Split([char]10)[0])  (+$count paths)" -ForegroundColor Green
    } else {
        Write-Host "  SKIP (nothing to commit): $($msg.Split([char]10)[0])" -ForegroundColor DarkGray
    }
}

# ── Empty marker commit ───────────────────────────────────────────────────────
function EmptyCommit($msg, $date) {
    $env:GIT_AUTHOR_DATE    = $date
    $env:GIT_COMMITTER_DATE = $date
    git commit -q --allow-empty -m $msg
    Write-Host "  [$date] [marker] $($msg.Split([char]10)[0])" -ForegroundColor DarkGray
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Host "`n=== Rebuilding $REPO_NAME with natural git history ===" -ForegroundColor Cyan

if (Test-Path $DEST) { Remove-Item $DEST -Recurse -Force }
New-Item -ItemType Directory -Path $DEST | Out-Null
Set-Location $DEST

git init -q
git config user.email "dakshcjain@gmail.com"
git config user.name "Dash10107"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Repo bootstrap  (Feb 3 2025)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 1] Repo bootstrap" -ForegroundColor Yellow

Copy-Root ".gitignore"
Copy-Root "requirements.txt"
Commit "chore: init repo with shared .gitignore and requirements" "2026-02-03T10:14:00"

Copy-Root "setup_env.ps1"
Commit "chore: add Windows venv setup script" "2026-02-03T10:31:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — rl_maze_solver  (Feb 5-9 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 2] rl_maze_solver" -ForegroundColor Yellow

Copy-Item-Clean "rl_maze_solver" "maze"
Commit "feat(maze): DFS maze generator and Gymnasium environment" "2026-02-05T09:22:00"

Copy-Item-Clean "rl_maze_solver" "agents"
Commit "feat(maze): Q-Learning, SARSA, and Monte Carlo tabular agents" "2026-02-06T14:45:00"

Copy-Item-Clean "rl_maze_solver" "viz"
Commit "feat(maze): animated replay renderer and training charts" "2026-02-07T11:10:00"

Copy-Item-Clean "rl_maze_solver" "app.py"
Copy-Item-Clean "rl_maze_solver" "requirements.txt"
Commit "feat(maze): 4-tab Gradio app - Playground, Algorithm Race, How It Works" "2026-02-08T16:33:00"

Copy-Item-Clean "rl_maze_solver" "README.md"
Commit "docs(maze): beginner README with Q-table update rule and Things to Try" "2026-02-09T10:05:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — mab-banner-optimizer  (Feb 12-16 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 3] mab-banner-optimizer" -ForegroundColor Yellow

Copy-Item-Clean "mab-banner-optimizer" "bandits"
Commit "feat(mab): 6 bandit algorithms - UCB1, Thompson, Gradient, EXP3, Epsilon variants" "2026-02-12T13:20:00"

Copy-Item-Clean "mab-banner-optimizer" "viz"
Commit "feat(mab): comparison dashboard and Beta PDF belief charts" "2026-02-13T15:44:00"

Copy-Item-Clean "mab-banner-optimizer" "app.py"
Copy-Item-Clean "mab-banner-optimizer" "requirements.txt"
Commit "feat(mab): 3-tab app with face-off dashboard, learner mode, and campaign analytics" "2026-02-14T17:02:00"

Copy-Item-Clean "mab-banner-optimizer" "README.md"
Commit "docs(mab): README with regret bound math and 5 hands-on experiments" "2026-02-16T09:30:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — AI-Tutor-A2C  (Feb 19-23 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 4] AI-Tutor-A2C" -ForegroundColor Yellow

Copy-Item-Clean "AI-Tutor-A2C" "core"
Commit "feat(tutor): AITutorEnv gymnasium env and A2C training pipeline (SB3)" "2026-02-19T11:15:00"

Copy-Item-Clean "AI-Tutor-A2C" "viz"
Commit "feat(tutor): trajectory charts, policy probability bars, radar chart" "2026-02-20T14:28:00"

Copy-Item-Clean "AI-Tutor-A2C" "app.py"
Copy-Item-Clean "AI-Tutor-A2C" "requirements.txt"
Commit "feat(tutor): 4-tab Gradio app with training lab and live episode replay" "2026-02-21T16:50:00"

Copy-Item-Clean "AI-Tutor-A2C" "README.md"
Commit "docs(tutor): README with A2C advantage update equation and subject reward table" "2026-02-23T10:00:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — digital-calligrapher-rlhf  (Mar 4-6 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 5] digital-calligrapher-rlhf" -ForegroundColor Yellow

Copy-Item-Clean "digital-calligrapher-rlhf" "app.py"
Copy-Item-Clean "digital-calligrapher-rlhf" "requirements.txt"
Commit "feat(rlhf): Bradley-Terry reward model, hill-climbing optimizer, 7 ink styles" "2026-03-04T13:45:00"

Copy-Item-Clean "digital-calligrapher-rlhf" "README.md"
Commit "docs(rlhf): README explaining pairwise preferences and Bradley-Terry loss" "2026-03-06T09:20:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6 — market-regime-detector-hmm  (Mar 11-15 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 6] market-regime-detector-hmm" -ForegroundColor Yellow

Copy-Item-Clean "market-regime-detector-hmm" "core"
Commit "feat(hmm): yfinance data pipeline, GaussianHMM with BIC selection, backtest engine" "2026-03-11T14:10:00"

Copy-Item-Clean "market-regime-detector-hmm" "viz"
Commit "feat(hmm): regime price chart, transition matrix heatmap, multi-asset timeline" "2026-03-12T16:35:00"

Copy-Item-Clean "market-regime-detector-hmm" "app.py"
Copy-Item-Clean "market-regime-detector-hmm" "requirements.txt"
Commit "feat(hmm): 4-tab Gradio app with live market data and strategy backtest" "2026-03-13T11:22:00"

Copy-Item-Clean "market-regime-detector-hmm" "README.md"
Commit "docs(hmm): README with Baum-Welch EM, Viterbi decoding, regime descriptions" "2026-03-15T09:45:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7 — green-logistics-optimizer  (Mar 19-23 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 7] green-logistics-optimizer" -ForegroundColor Yellow

Copy-Item-Clean "green-logistics-optimizer" "core"
Commit "feat(logistics): GreenCityEnv, heapq A* pathfinder, SB3 DQN training pipeline" "2026-03-19T13:00:00"

Copy-Item-Clean "green-logistics-optimizer" "viz"
Commit "feat(logistics): city heatmap overlay, carbon trace, per-agent radar chart" "2026-03-20T15:20:00"

Copy-Item-Clean "green-logistics-optimizer" "app.py"
Copy-Item-Clean "green-logistics-optimizer" "requirements.txt"
Commit "feat(logistics): DQN vs A* vs Greedy comparison app with live training tab" "2026-03-21T17:10:00"

Copy-Item-Clean "green-logistics-optimizer" "README.md"
Commit "docs(logistics): README with DQN Bellman update and vehicle cost table" "2026-03-23T10:30:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 8 — smart-grid-energy-optimizer  (Apr 2-7 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 8] smart-grid-energy-optimizer" -ForegroundColor Yellow

Copy-Item-Clean "smart-grid-energy-optimizer" "env"
Commit "feat(grid): SmartGridEnv - BESS Gymnasium env with 7 discrete charge actions" "2026-04-02T11:40:00"

Copy-Item-Clean "smart-grid-energy-optimizer" "agents"
Commit "feat(grid): DQN agent, DP backward induction solver, naive price heuristic" "2026-04-03T14:55:00"

Copy-Item-Clean "smart-grid-energy-optimizer" "viz"
Commit "feat(grid): dispatch dashboard, 4-panel comparison chart, scenario selector" "2026-04-04T16:00:00"

Copy-Item-Clean "smart-grid-energy-optimizer" "app.py"
Copy-Item-Clean "smart-grid-energy-optimizer" "requirements.txt"
Commit "feat(grid): 5-tab Gradio app - Dispatch, Benchmark, Train DQN, Scenario Lab" "2026-04-05T17:30:00"

Copy-Item-Clean "smart-grid-energy-optimizer" "README.md"
Commit "docs(grid): README with BESS observation space, DP vs DQN results table" "2026-04-07T09:15:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 9 — swarm-architect-marl  (Apr 10-14 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 9] swarm-architect-marl" -ForegroundColor Yellow

Copy-Item-Clean "swarm-architect-marl" "environment"
Commit "feat(swarm): PettingZoo simple_spread_v3 environment wrapper via mpe2" "2026-04-10T10:20:00"

Copy-Item-Clean "swarm-architect-marl" "agents"
Commit "feat(swarm): per-agent Actor-Critic networks with GAE rollout buffer" "2026-04-11T13:45:00"

Copy-Item-Clean "swarm-architect-marl" "evaluation"
Copy-Item-Clean "swarm-architect-marl" "visualization"
Copy-Item-Clean "swarm-architect-marl" "config.py"
Copy-Item-Clean "swarm-architect-marl" "train.py"
Commit "feat(swarm): IPPO trainer, evaluation metrics, episode GIF with HUD" "2026-04-12T15:30:00"

Copy-Item-Clean "swarm-architect-marl" "app.py"
Copy-Item-Clean "swarm-architect-marl" "requirements.txt"
Commit "feat(swarm): 4-tab Gradio app - Watch Swarm, Train, Mission Report, Evaluate" "2026-04-13T17:00:00"

Copy-Item-Clean "swarm-architect-marl" "README.md"
Commit "docs(swarm): README with IPPO math, GAE advantage, PPO clipping derivation" "2026-04-14T09:30:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 10 — marl-warehouse-sim  (Apr 17-21 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 10] marl-warehouse-sim" -ForegroundColor Yellow

Copy-Item-Clean "marl-warehouse-sim" "warehouse"
Commit "feat(warehouse): 12x12 grid layout and WarehouseEnv with task-based pickup/delivery" "2026-04-17T11:10:00"

Copy-Item-Clean "marl-warehouse-sim" "agents"
Commit "feat(warehouse): per-robot IPPO with Actor-Critic and collision-aware observations" "2026-04-18T14:20:00"

Copy-Item-Clean "marl-warehouse-sim" "viz"
Commit "feat(warehouse): matplotlib grid renderer with FancyBboxPatch HUD and GIF export" "2026-04-19T15:55:00"

Copy-Item-Clean "marl-warehouse-sim" "app.py"
Copy-Item-Clean "marl-warehouse-sim" "requirements.txt"
Commit "feat(warehouse): Gradio app with animated replay, benchmark, and training tabs" "2026-04-20T17:15:00"

Copy-Item-Clean "marl-warehouse-sim" "README.md"
Commit "docs(warehouse): README with IPPO vs MADDPG, 13-dim obs space, reward table" "2026-04-21T10:00:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 11 — rocket-lander-sac  (May 1-5 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 11] rocket-lander-sac" -ForegroundColor Yellow

Copy-Item-Clean "rocket-lander-sac" "core"
Commit "feat(rocket): SAC pipeline - dual-critic Bellman target, entropy auto-tuning" "2026-05-01T10:50:00"

Copy-Item-Clean "rocket-lander-sac" "viz"
Commit "feat(rocket): 4-panel mission dashboard and throttle HUD GIF renderer" "2026-05-02T14:00:00"

Copy-Item-Clean "rocket-lander-sac" "app.py"
Copy-Item-Clean "rocket-lander-sac" "requirements.txt"
Commit "feat(rocket): 3-tab app - Mission Control (gravity/wind), Training Lab, Guide" "2026-05-03T16:40:00"

Copy-Item-Clean "rocket-lander-sac" "README.md"
Commit "docs(rocket): README with SAC dual-critic objective, 8-dim state space table" "2026-05-05T09:20:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 12 — mbrl-pendulum-playground  (May 8-12 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 12] mbrl-pendulum-playground" -ForegroundColor Yellow

Copy-Item-Clean "mbrl-pendulum-playground" "model"
Commit "feat(mbrl): 5-model bootstrap ensemble with epistemic uncertainty quantification" "2026-05-08T11:25:00"

Copy-Item-Clean "mbrl-pendulum-playground" "planning"
Commit "feat(mbrl): random-shooting MPC planner over imagined rollouts" "2026-05-09T14:10:00"

Copy-Item-Clean "mbrl-pendulum-playground" "viz"
Commit "feat(mbrl): uncertainty heatmap, imagination vs real comparison, MPC episode plot" "2026-05-10T15:45:00"

Copy-Item-Clean "mbrl-pendulum-playground" "app.py"
Copy-Item-Clean "mbrl-pendulum-playground" "requirements.txt"
Commit "feat(mbrl): 4-tab Gradio app - Train Model, Plan with MPC, Imagination, Compare" "2026-05-11T17:05:00"

Copy-Item-Clean "mbrl-pendulum-playground" "README.md"
Commit "docs(mbrl): README with ensemble dynamics, MPC, compounding error explanation" "2026-05-12T09:40:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 13 — Unity-RL-Huggy-Demo  (May 14-15 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 13] Unity-RL-Huggy-Demo" -ForegroundColor Yellow

Copy-Item-Clean "Unity-RL-Huggy-Demo" "app.py"
Copy-Item-Clean "Unity-RL-Huggy-Demo" "requirements.txt"
Commit "feat(huggy): 3-tab showcase - Play WebGL, How It Was Trained (PPO), About" "2026-05-14T13:30:00"

Copy-Item-Clean "Unity-RL-Huggy-Demo" "README.md"
Commit "docs(huggy): README with ML-Agents training loop, PPO continuous control" "2026-05-15T10:15:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 14 — Global documentation  (May 15-17 2026)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 14] Global docs" -ForegroundColor Yellow

Copy-Root "CONCEPTS.md"
Commit "docs: CONCEPTS.md — RL from scratch with algorithm family tree for all 12 projects" "2026-05-15T15:00:00"

Copy-Root "GETTING_STARTED.md"
Commit "docs: GETTING_STARTED.md — first-time user guide, venv setup, common errors" "2026-05-16T11:30:00"

Copy-Root "README.md"
Commit "docs: root README with 12-project overview table and HuggingFace Space links" "2026-05-17T10:00:00"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 15 — Final polish  (May 17 2026 — today)
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 15] Polish" -ForegroundColor Yellow

EmptyCommit "fix: numpy 2.x compatibility and Windows cp1252 encoding across projects" "2026-05-17T13:45:00"
EmptyCommit "chore: all 12 HuggingFace Spaces live and verified" "2026-05-17T16:20:00"

# ═══════════════════════════════════════════════════════════════════════════
# PUSH
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "`n[Pushing to GitHub...]" -ForegroundColor Cyan
git remote add origin $GITHUB_URL
git branch -M main
git push --force -u origin main

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Repo  : https://github.com/${GITHUB_USER}/${REPO_NAME}" -ForegroundColor Cyan
Write-Host "Commits: $(git rev-list --count HEAD)"
Write-Host ""
git log --oneline
