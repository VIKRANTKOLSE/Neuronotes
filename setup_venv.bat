@echo off
REM ============================================================
REM setup_venv.bat — Windows CMD / Batch Virtual Environment Setup
REM ============================================================

echo [Neuronotes] Setting up Python virtual environment...

REM Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

REM Create virtual environment if it does not exist
if not exist ".venv" (
    echo [1/3] Creating virtual environment in .venv ...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment (.venv) already exists.
)

REM Activate virtual environment
echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip and install requirements
echo [3/3] Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [SUCCESS] Virtual environment ready and dependencies installed!
    echo To activate in future terminal sessions, run:
    echo     .venv\Scripts\activate
    echo.
    echo To run the default experiment:
    echo     python main.py
    echo ============================================================
) else (
    echo [ERROR] Dependency installation encountered an issue.
)

cmd /k
