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
REM    2. CUDA Toolkit 12.8 or 12.9 (Fragmenta pins cu128; 12.8+ is what can
REM       target sm_120). Set FA_CUDA_HOME to force a specific toolkit folder.
REM    3. The Fragmenta venv ACTIVATED with the pinned torch already installed
REM       (Python 3.14, torch==2.11.0+cu128): run fragmenta.bat once on
REM       dev/1.3 so it rebuilds venv\, then venv\Scripts\activate.
REM
REM  Usage (from the x64 Native Tools prompt, venv active, repo root):
REM    packaging\windows\build_flash_attn.bat
REM    packaging\windows\build_flash_attn.bat 2.8.3.post2 "80;120" 4
REM      arg1 = flash-attn version   (default 2.8.3.post2, same as Linux;
REM                                   2.7.4.post1 is the fallback proven on
REM                                   Blackwell if 2.8.x won't build)
REM      arg2 = CUDA archs           (default "80;120": RTX 30xx/40xx run the
REM                                   sm_80 code, 120 = RTX 50xx. Quote it.)
REM      arg3 = MAX_JOBS             (default 4; lower to 2 if the build OOMs)
REM      arg4 = TORCH_CUDA_ARCH_LIST (2.7.x only; default "8.0;12.0")
REM
REM  Output: dist_wheels\flash_attn-<ver>-cp314-cp314-win_amd64.whl + its sha256
REM  Build time is several hours (two archs). Do not close the window.
REM ===========================================================================
setlocal enabledelayedexpansion

set FA_VERSION=%~1
if "%FA_VERSION%"=="" set FA_VERSION=2.8.3.post2
set FA_ARCH=%~2
if "%FA_ARCH%"=="" set FA_ARCH=80;120
set FA_JOBS=%3
if "%FA_JOBS%"=="" set FA_JOBS=4
set FA_TORCH_ARCH=%~4
if "%FA_TORCH_ARCH%"=="" set FA_TORCH_ARCH=8.0;12.0

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

REM torch refuses to build inside an activated VC env without this.
set DISTUTILS_USE_SDK=1

REM The default CUDA_PATH is often an older toolkit (e.g. 12.1), which cannot
REM target sm_120. Prefer a 12.8 or 12.9 install when one is present; set
REM FA_CUDA_HOME beforehand to force a specific toolkit.
if "%FA_CUDA_HOME%"=="" if defined CUDA_PATH_V12_8 set "FA_CUDA_HOME=%CUDA_PATH_V12_8%"
if "%FA_CUDA_HOME%"=="" if defined CUDA_PATH_V12_9 set "FA_CUDA_HOME=%CUDA_PATH_V12_9%"
if not "%FA_CUDA_HOME%"=="" (
  set "CUDA_HOME=%FA_CUDA_HOME%"
  set "CUDA_PATH=%FA_CUDA_HOME%"
  set "PATH=%FA_CUDA_HOME%\bin;%PATH%"
)

where nvcc >nul 2>&1
if errorlevel 1 (
  echo [ERROR] nvcc not found. Install the CUDA Toolkit matching your torch
  echo         build ^(cu128 -^> CUDA 12.8^) and ensure it is on PATH.
  exit /b 1
)
echo --- nvcc version ---
nvcc --version | findstr /C:"release"

REM sm_120 (RTX 50xx) needs CUDA 12.8+; an older nvcc fails hours into the build.
set FA_NVCC_REL=
for /f "tokens=5 delims=, " %%v in ('nvcc --version ^| findstr /C:"release"') do set FA_NVCC_REL=%%v
for /f "tokens=1,2 delims=." %%a in ("%FA_NVCC_REL%") do set /a FA_NVCC_NUM=%%a*100+%%b
if %FA_NVCC_NUM% LSS 1208 (
  echo [ERROR] nvcc is CUDA %FA_NVCC_REL%; this build needs CUDA 12.8 or newer.
  echo         Install CUDA 12.8, or set FA_CUDA_HOME to a 12.8+ toolkit folder.
  exit /b 1
)
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
REM 2.8.x reads FLASH_ATTN_CUDA_ARCHS; 2.7.x read TORCH_CUDA_ARCH_LIST
REM (arg4, dotted form). Set both.
set FLASH_ATTN_CUDA_ARCHS=%FA_ARCH%
set TORCH_CUDA_ARCH_LIST=%FA_TORCH_ARCH%
set MAX_JOBS=%FA_JOBS%
if not exist dist_wheels mkdir dist_wheels

echo(
echo Building flash-attn==%FA_VERSION% (this takes several hours)...
echo   FLASH_ATTENTION_FORCE_BUILD=%FLASH_ATTENTION_FORCE_BUILD%
echo   FLASH_ATTN_CUDA_ARCHS=%FLASH_ATTN_CUDA_ARCHS%
echo   TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
echo   MAX_JOBS=%MAX_JOBS%
echo(
python -m pip wheel flash-attn==%FA_VERSION% --no-build-isolation --no-deps -w dist_wheels
if errorlevel 1 (
  echo(
  echo [ERROR] Build failed. Common fixes:
  echo   * OOM during compile  -^> rerun with a lower MAX_JOBS, e.g. ... 2
  echo   * arch rejected       -^> confirm CUDA 12.8 is the nvcc on PATH
  echo   * 2.8.x incompatible  -^> try 2.7.4.post1 ^(proven on Blackwell^)
  exit /b 1
)

echo(
echo === DONE ===
echo Wheel(s) written to dist_wheels\ :
for %%w in (dist_wheels\flash_attn-*.whl) do certutil -hashfile "%%w" SHA256
echo(
echo Install it with:
echo   pip install --no-deps dist_wheels\^<the-wheel^>.whl
echo Then verify:
echo   python -c "import flash_attn; print(flash_attn.__version__)"
endlocal
