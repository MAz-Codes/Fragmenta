@echo off
REM Fragmenta Desktop - Windows Launcher Script

echo Fragmenta Desktop
echo ===================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Project root: %PROJECT_ROOT%

python --version 2>nul
if errorlevel 1 (
    echo Python not found. Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

if not exist "%PROJECT_ROOT%\venv" (
    echo Creating Python virtual environment...
    python -m venv "%PROJECT_ROOT%\venv"
)

echo Activating virtual environment...
call "%PROJECT_ROOT%\venv\Scripts\activate.bat"

cd /d "%PROJECT_ROOT%"

echo Updating pip...
python -m pip install --upgrade pip setuptools wheel build --quiet

echo Installing all dependencies from requirements.txt...
echo This may take a few minutes...
python -m pip install -r requirements.txt --quiet

echo Installing bundled stable-audio-tools...
cd stable-audio-tools && python -m pip install -e . --quiet && cd ..

echo Verifying PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo PyTorch not found, reinstalling...
    python -m pip uninstall torch torchvision torchaudio -y --quiet
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
)

echo Starting Fragmenta Desktop...
python main.py

pause
