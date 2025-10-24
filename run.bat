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
python -m pip install --upgrade pip setuptools wheel build

echo Installing PyTorch...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo Installing all other dependencies from requirements.txt...
echo This may take a few minutes...
echo Note: Skipping flash-attn on Windows (Linux-only optimization)
findstr /V "flash-attn" requirements.txt > "%TEMP%\requirements_windows.txt"
python -m pip install -r "%TEMP%\requirements_windows.txt"
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    del "%TEMP%\requirements_windows.txt"
    pause
    exit /b 1
)
del "%TEMP%\requirements_windows.txt"
echo Dependencies installed successfully

echo Installing bundled stable-audio-tools...
cd stable-audio-tools
python -m pip install -e .
if errorlevel 1 (
    echo ERROR: Failed to install stable-audio-tools
    cd ..
    pause
    exit /b 1
)
cd ..
echo stable-audio-tools installed successfully

echo Checking if React frontend is built...
if not exist "app\frontend\build\index.html" (
    echo React frontend not built. Building now...
    cd app\frontend
    
    echo Installing Node.js dependencies...
    call npm install
    
    echo Building React app...
    call npm run build
    
    cd ..\..
    echo Frontend build complete!
) else (
    echo Frontend already built. Skipping build step.
)

echo Starting Fragmenta...
python main.py

pause