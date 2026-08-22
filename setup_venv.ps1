# ============================================================
# setup_venv.ps1 — PowerShell Virtual Environment Setup Script
# ============================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Neuronotes Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Verify Python is accessible
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] 'python' was not found in PATH. Please install Python 3.10+." -ForegroundColor Red
    Exit 1
}

# 2. Create .venv if not present
if (-not (Test-Path -Path ".venv")) {
    Write-Host "[1/3] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        Exit 1
    }
} else {
    Write-Host "[1/3] Virtual environment (.venv) already exists." -ForegroundColor Green
}

# 3. Activate venv in current session
Write-Host "[2/3] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "[WARNING] Could not find $activateScript. You can activate using CMD or bash." -ForegroundColor Yellow
}

# 4. Install dependencies
Write-Host "[3/3] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n============================================================" -ForegroundColor Green
    Write-Host "[SUCCESS] Virtual environment configured and active!" -ForegroundColor Green
    Write-Host "Activate in new PowerShell sessions with:" -ForegroundColor White
    Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "`nTo run the experiment pipeline:" -ForegroundColor White
    Write-Host "    python main.py" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to install dependencies from requirements.txt." -ForegroundColor Red
}
