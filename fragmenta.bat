@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo Fragmenta Desktop
echo ===================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
echo Project root: %PROJECT_ROOT%

set "PY_LAUNCHER="

where py >nul 2>&1
if errorlevel 1 goto :try_python
py -3.14 --version >nul 2>&1
if errorlevel 1 goto :try_python
set "PY_LAUNCHER=py -3.14"
goto :have_python

:try_python
where python >nul 2>&1
if errorlevel 1 goto :missing_python
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
    echo %%v | findstr /b /c:"3.14." >nul && set "PY_LAUNCHER=python"
)
if not defined PY_LAUNCHER goto :missing_python
goto :have_python

:missing_python
echo.
echo ERROR: Python 3.14 was not found.
echo.
echo This project requires Python 3.14 specifically.
echo.
echo Install Python 3.14 from:
echo    https://www.python.org/downloads/windows/
echo With the Python install manager, run: py install 3.14
echo With the classic installer, tick "Add python.exe to PATH" and keep
echo the "py launcher" option checked. Then re-run this script.
echo.
goto :end_fail

:have_python
echo Using Python 3.14 via: %PY_LAUNCHER%
for /f "delims=" %%v in ('%PY_LAUNCHER% --version 2^>^&1') do echo Detected: %%v

%PY_LAUNCHER% "%PROJECT_ROOT%\install.py" --launch
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
