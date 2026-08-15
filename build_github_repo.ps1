# build_github_repo.ps1
# Exports all 12 RL projects into a clean GitHub portfolio repo.
# Run this from the ReinforcementLearning root directory.
#
# Before running:
#   1. Create an empty repo on github.com called "reinforcement-learning-lab" (no README, no .gitignore)
#   2. Generate a GitHub Personal Access Token with "repo" scope
#      github.com -> Settings -> Developer Settings -> Personal Access Tokens -> Tokens (classic)
#   3. Fill in GITHUB_USER and GITHUB_TOKEN below

$GITHUB_USER  = "Dash10107"
$GITHUB_TOKEN = "YOUR_GITHUB_PAT_HERE"    # replace this
$REPO_NAME    = "reinforcement-learning-lab"

$SRC  = $PSScriptRoot                     # the ReinforcementLearning folder
$DEST = "$env:USERPROFILE\Desktop\$REPO_NAME"

Write-Host "`n=== Building GitHub portfolio repo ===" -ForegroundColor Cyan
Write-Host "Source : $SRC"
Write-Host "Dest   : $DEST"
Write-Host ""

# ── Step 1: Create destination folder ────────────────────────────────────────
if (Test-Path $DEST) {
    Write-Host "Destination already exists. Removing..." -ForegroundColor Yellow
    Remove-Item $DEST -Recurse -Force
}
New-Item -ItemType Directory -Path $DEST | Out-Null
Write-Host "Created $DEST" -ForegroundColor Green

# ── Step 2: Copy root files ───────────────────────────────────────────────────
Write-Host "`nCopying root files..."
foreach ($file in @("README.md", ".gitignore", "CONCEPTS.md", "GETTING_STARTED.md", "requirements.txt", "setup_env.ps1")) {
    $src_file = Join-Path $SRC $file
    if (Test-Path $src_file) {
        Copy-Item $src_file $DEST -Force
        Write-Host "  $file" -ForegroundColor Gray
    }
}

# ── Step 3: Copy each project (excluding .git, __pycache__, .venv) ───────────
$PROJECTS = @(
    "AI-Tutor-A2C",
    "Unity-RL-Huggy-Demo",
    "digital-calligrapher-rlhf",
    "green-logistics-optimizer",
    "mab-banner-optimizer",
    "market-regime-detector-hmm",
    "marl-warehouse-sim",
    "mbrl-pendulum-playground",
    "rl_maze_solver",
    "rocket-lander-sac",
    "smart-grid-energy-optimizer",
    "swarm-architect-marl"
)

Write-Host "`nCopying projects..."
foreach ($project in $PROJECTS) {
    $src_path  = Join-Path $SRC $project
    $dest_path = Join-Path $DEST $project
    if (Test-Path $src_path) {
        # robocopy: /E = all subdirs, /XD = exclude dirs, /XF = exclude files, /NFL /NDL /NJH = quiet
        robocopy $src_path $dest_path /E `
            /XD .git __pycache__ .venv checkpoints wandb mlruns runs `
            /XF "*.pyc" "*.pyo" "*.pth" "*.zip" "*.pt" "*.pkl" "*.gif" `
            /NFL /NDL /NJH /NJS | Out-Null
        Write-Host "  $project" -ForegroundColor Gray
    } else {
        Write-Host "  SKIP (not found): $project" -ForegroundColor Yellow
    }
}

# ── Step 4: Init git and make first commit ────────────────────────────────────
Write-Host "`nInitialising git..."
Set-Location $DEST
git init -q
git add .

$file_count   = (git ls-files | Measure-Object -Line).Lines
$folder_count = $PROJECTS.Count

git commit -q -m "Initial commit — RL Portfolio ($folder_count projects, $file_count files)

12 end-to-end reinforcement learning projects with interactive
Hugging Face Space deployments. Algorithms covered:
Q-Learning, SARSA, Monte Carlo, DQN, A2C, PPO, SAC, IPPO,
MBRL + MPC, Gaussian HMM, Bradley-Terry RLHF, Multi-Armed Bandits.

Each project includes a full README, Things to Try section,
and live demo link to the deployed Gradio app."

Write-Host "Git repo initialised with $file_count files" -ForegroundColor Green

# ── Step 5: Push to GitHub ────────────────────────────────────────────────────
Write-Host "`nPushing to GitHub..."
$remote_url = "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
git remote add origin $remote_url
git branch -M main
git push -u origin main

# Remove token from remote URL after push
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

Write-Host "`n=== Done! ===" -ForegroundColor Green
Write-Host "GitHub repo: https://github.com/${GITHUB_USER}/${REPO_NAME}" -ForegroundColor Cyan
Write-Host ""
Write-Host "For future updates, run this script again or copy individual project folders manually."
