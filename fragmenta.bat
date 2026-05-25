@echo off
setlocal

chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo Fragmenta Desktop
echo ===================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Project root: %PROJECT_ROOT%

REM ---------------------------------------------------------------------------
REM  Locate Python 3.11
REM ---------------------------------------------------------------------------
set "PY_LAUNCHER="

where py >nul 2>&1
if errorlevel 1 goto :try_python
py -3.11 --version >nul 2>&1
if errorlevel 1 goto :try_python
set "PY_LAUNCHER=py -3.11"
goto :have_python

:try_python
where python >nul 2>&1
if errorlevel 1 goto :missing_python
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
    echo %%v | findstr /b /c:"3.11." >nul && set "PY_LAUNCHER=python"
)
if not defined PY_LAUNCHER goto :missing_python
goto :have_python

:missing_python
echo.
echo ERROR: Python 3.11 was not found.
echo.
echo This project requires Python 3.11 specifically. Newer versions
echo (3.12, 3.13) are NOT compatible.
echo.
echo Install Python 3.11 from:
echo    https://www.python.org/downloads/release/python-3119/
echo.
echo During install, tick "Add python.exe to PATH" and keep the
echo "py launcher" option checked. Then re-run this script.
echo.
goto :end_fail

:have_python
echo Using Python 3.11 via: %PY_LAUNCHER%
for /f "delims=" %%v in ('%PY_LAUNCHER% --version 2^>^&1') do echo Detected: %%v

REM ---------------------------------------------------------------------------
REM  Reuse / rebuild the venv
REM ---------------------------------------------------------------------------
set "VENV=%PROJECT_ROOT%\venv"
set "VENV_PY=%VENV%\Scripts\python.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo Existing venv was not built with Python 3.11 - removing and recreating...
        rmdir /s /q "%VENV%"
    )
)

if not exist "%VENV%" (
    echo Creating Python virtual environment...
    %PY_LAUNCHER% -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: failed to create virtual environment.
        goto :end_fail
    )
)

echo Activating virtual environment...
call "%VENV%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: failed to activate virtual environment at %VENV%
    goto :end_fail
)

cd /d "%PROJECT_ROOT%"

REM ---------------------------------------------------------------------------
REM  Dependencies
REM ---------------------------------------------------------------------------
echo Updating pip...
python -m pip install --upgrade pip "setuptools<70" wheel build
if errorlevel 1 goto :end_fail

REM requirements.txt declares --extra-index-url for the CUDA 12.8 torch wheels
REM at the top of the file, and gates flash-attn behind sys_platform == 'linux'
REM so pip skips it cleanly on Windows. One install resolves the whole graph;
REM no manual torch/numpy pre-installs needed. The Stable Audio 3 vendor lives
REM at vendor\stable-audio-3 and is loaded via sys.path from Python — nothing
REM to pip-install for it here.
echo Installing dependencies from requirements.txt...
echo (first run takes several minutes - torch + transformers are large)
python -m pip install -r requirements.txt ^
    --find-links "%PROJECT_ROOT%\utils\vendor\wheels" --prefer-binary
if errorlevel 1 (
    echo ERROR: failed to install dependencies. Check the log above.
    goto :end_fail
)
echo Dependencies installed successfully.

REM ---------------------------------------------------------------------------
REM  Launch
REM ---------------------------------------------------------------------------
echo Starting Fragmenta...
python start.py
set "RUN_RESULT=%ERRORLEVEL%"
if not "%RUN_RESULT%"=="0" (
    echo.
    echo Fragmenta exited with code %RUN_RESULT%.
)

:end_fail
echo.
echo Press any key to close this window...
pause >nul
endlocal
