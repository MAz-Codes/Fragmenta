@echo off
REM ===========================================================================
REM  Build a Flash Attention 2 wheel for Fragmenta on Windows.
REM
REM  Why: there is no official PyPI Windows wheel for flash-attn. The sa3-medium
REM  model needs Flash Attention 2 for its long-form (up to 380s) sliding-window
REM  attention. This compiles a wheel matching Fragmenta's pinned stack so it can
REM  be installed (and later shipped/hosted) for Windows + NVIDIA users.
REM
REM  PREREQUISITES (read before running):
REM    1. Visual Studio 2022 Build Tools with "Desktop development with C++"
REM       (MSVC v143). RUN THIS SCRIPT FROM the "x64 Native Tools Command Prompt
REM       for VS 2022" so cl.exe is on PATH.
REM    2. CUDA Toolkit matching your torch build (Fragmenta pins cu128 -> CUDA
REM       12.8). `nvcc --version` must report that version.
REM    3. The Fragmenta venv ACTIVATED with the pinned torch already installed
REM       (torch==2.7.1+cu128).
REM
REM  Usage (from the x64 Native Tools prompt, venv active, repo root):
REM    packaging\windows\build_flash_attn.bat
REM    packaging\windows\build_flash_attn.bat 2.7.4.post1 12.0 4
REM      arg1 = flash-attn version   (default 2.7.4.post1 — the version proven
REM                                   to work on Blackwell/cu128; Linux pins
REM                                   2.8.3 separately in requirements.txt)
REM      arg2 = TORCH_CUDA_ARCH_LIST (default 12.0 = Blackwell sm_120 / RTX 50xx;
REM                                   use 8.0 for Ampere, 8.9 for Ada, etc.)
REM      arg3 = MAX_JOBS             (default 4; lower to 2 if the build OOMs)
REM
REM  Output: dist_wheels\flash_attn-<ver>+...-cp311-cp311-win_amd64.whl
REM  Build time is typically 1-3 hours. Be patient; do not close the window.
REM ===========================================================================
setlocal enabledelayedexpansion

set FA_VERSION=%1
if "%FA_VERSION%"=="" set FA_VERSION=2.7.4.post1
set FA_ARCH=%2
if "%FA_ARCH%"=="" set FA_ARCH=12.0
set FA_JOBS=%3
if "%FA_JOBS%"=="" set FA_JOBS=4

echo(
echo === Fragmenta flash-attn Windows build ===
echo   flash-attn version : %FA_VERSION%
echo   CUDA arch list     : %FA_ARCH%
echo   MAX_JOBS           : %FA_JOBS%
echo(

REM --- Toolchain checks -----------------------------------------------------
where cl >nul 2>&1
if errorlevel 1 (
  echo [ERROR] cl.exe not found. Run this from the "x64 Native Tools Command
  echo         Prompt for VS 2022" so MSVC is on PATH.
  exit /b 1
)
where nvcc >nul 2>&1
if errorlevel 1 (
  echo [ERROR] nvcc not found. Install the CUDA Toolkit matching your torch
  echo         build ^(cu128 -^> CUDA 12.8^) and ensure it is on PATH.
  exit /b 1
)
echo --- nvcc version ---
nvcc --version | findstr /C:"release"
echo(
echo --- torch / CUDA visible to Python ---
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| is_available', torch.cuda.is_available())"
if errorlevel 1 (
  echo [ERROR] Could not import torch. Activate the Fragmenta venv with the
  echo         pinned torch installed before running this script.
  exit /b 1
)

echo(
echo Installing build deps (ninja, packaging, wheel, setuptools)...
python -m pip install -q ninja packaging wheel setuptools
if errorlevel 1 (
  echo [ERROR] Failed to install build dependencies.
  exit /b 1
)

REM --- Build ----------------------------------------------------------------
set FLASH_ATTENTION_FORCE_BUILD=TRUE
set TORCH_CUDA_ARCH_LIST=%FA_ARCH%
set MAX_JOBS=%FA_JOBS%
if not exist dist_wheels mkdir dist_wheels

echo(
echo Building flash-attn==%FA_VERSION% (this can take 1-3 hours)...
echo   FLASH_ATTENTION_FORCE_BUILD=%FLASH_ATTENTION_FORCE_BUILD%
echo   TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
echo   MAX_JOBS=%MAX_JOBS%
echo(
python -m pip wheel flash-attn==%FA_VERSION% --no-build-isolation --no-deps -w dist_wheels
if errorlevel 1 (
  echo(
  echo [ERROR] Build failed. Common fixes:
  echo   * OOM during compile  -^> rerun with a lower MAX_JOBS, e.g. ... 2
  echo   * arch rejected       -^> confirm CUDA 12.8 + TORCH_CUDA_ARCH_LIST=12.0
  echo   * 2.8.3 incompatible  -^> try 2.7.4.post1 (strongest Blackwell support)
  exit /b 1
)

echo(
echo === DONE ===
echo Wheel(s) written to dist_wheels\ :
dir /b dist_wheels\flash_attn-*.whl
echo(
echo Install it with:
echo   pip install --no-deps dist_wheels\<the-wheel>.whl
echo Then verify:
echo   python -c "import flash_attn; print(flash_attn.__version__)"
endlocal
