@echo off


chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo Fragmenta Desktop
echo ===================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Project root: %PROJECT_ROOT%

set "PY_LAUNCHER="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 --version >nul 2>nul
    if not errorlevel 1 set "PY_LAUNCHER=py -3.11"
)

if not defined PY_LAUNCHER (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>nul') do (
        echo %%v | findstr /b "3.11." >nul && set "PY_LAUNCHER=python"
    )
)

if not defined PY_LAUNCHER (
    echo.
    echo ERROR: Python 3.11 was not found.
    echo.
    echo This project requires Python 3.11 specifically — newer versions
    echo (3.12, 3.13) are NOT compatible with the pinned numpy/pandas.
    echo.
    echo Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
    echo Make sure to check "Add python.exe to PATH" during install,
    echo or rely on the "py" launcher that ships with the installer.
    echo.
    pause
    exit /b 1
)

echo Using Python 3.11 via: %PY_LAUNCHER%

if not exist "%PROJECT_ROOT%\venv" (
    echo Creating Python virtual environment...
    %PY_LAUNCHER% -m venv "%PROJECT_ROOT%\venv"
)

echo Activating virtual environment...
call "%PROJECT_ROOT%\venv\Scripts\activate.bat"

cd /d "%PROJECT_ROOT%"

echo Updating pip...
python -m pip install --upgrade pip "setuptools<70" wheel build

echo Installing PyTorch...
python -m pip install "torch>=2.5,<=2.8" "torchvision<0.24" "torchaudio>=2.5,<=2.8" "numpy==1.23.5" --index-url https://download.pytorch.org/whl/cu128

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

echo Starting Fragmenta...
python start.py

pause