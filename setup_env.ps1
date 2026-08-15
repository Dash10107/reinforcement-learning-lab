# PowerShell script to set up a unified virtual environment for Reinforcement Learning projects

Write-Host "--- Setting up Unified Virtual Environment ---" -ForegroundColor Cyan

# 1. Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
} else {
    Write-Host "Virtual environment (.venv) already exists."
}

# 2. Upgrade pip and install requirements
Write-Host "Installing dependencies from master requirements.txt..." -ForegroundColor Yellow

# Use the python executable inside the venv to ensure we install there
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "--- Setup Complete! ---" -ForegroundColor Green
Write-Host "To activate the environment, run:"
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
