<#
Build Fragmenta-<ver>-Setup.exe (Windows x64).

Mirrors macos/build_dmg.sh. Steps:
  1. Assemble the payload (app code via git archive + standalone Python).
  2. Generate the .ico PyInstaller needs at build time.
  3. Freeze packaging/launcher.py -> fragmenta.exe.
  4. Wrap the frozen launcher + payload into a per-user installer with Inno Setup.

The launcher's first-run progress splash uses Tkinter. Rather than depend on the
build box's Python shipping Tk, we freeze with the SAME standalone CPython we
ship: python-build-standalone bundles _tkinter + Tcl/Tk, so PyInstaller captures
the splash for free and the build needs nothing preinstalled. PyInstaller lives
in a throwaway venv built from that interpreter so the shipped payload Python
stays pristine (no PyInstaller riding along).

Prereqs: git, Inno Setup (iscc on PATH). Run from anywhere; paths resolve to the
repo root. assemble.py uses `git archive HEAD`, so commit changes first.

Signing is left to the caller: sign packaging\build\Fragmenta-<ver>-Setup.exe
with signtool using your code-signing cert.
#>
$ErrorActionPreference = "Stop"

$REPO    = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VERSION = (Get-Content (Join-Path $REPO "VERSION") -Raw).Trim()
$DIST    = Join-Path $REPO "packaging\build"
$PAYLOAD = Join-Path $DIST "payload"
$WIN     = Join-Path $DIST "win"

Write-Host "==> Fragmenta $VERSION - Windows x64 build"
Set-Location $REPO

# 1. Assemble the payload (app code + standalone Python).
python packaging\assemble.py --target windows-x64 --out $PAYLOAD

# 2. Build tooling in a throwaway venv built from the SAME standalone CPython we
# ship (it bundles Tkinter, so PyInstaller captures the splash). Keeping the
# tools here means the build box needs no preinstalled PyInstaller/Pillow and the
# shipped payload Python stays pristine.
$PbsPy     = Join-Path $PAYLOAD "python-3.11\python.exe"
$BuildVenv = Join-Path $DIST "_buildvenv"
& $PbsPy -c "import tkinter" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error ("the bundled standalone Python lacks Tkinter - the first-run splash`n" +
        "would be missing. Bump PBS_RELEASE in packaging/python_standalone.py to a`n" +
        "build that ships Tcl/Tk and rebuild.")
    exit 1
}
if (Test-Path $BuildVenv) { Remove-Item -Recurse -Force $BuildVenv }
& $PbsPy -m venv $BuildVenv
$VenvPy = Join-Path $BuildVenv "Scripts\python.exe"
& $VenvPy -m pip install --quiet --upgrade pip pyinstaller pillow

# 3. App icon (PyInstaller needs the .ico at build time) — via the venv's Pillow,
# so the build box itself needs nothing preinstalled.
& $VenvPy -c "from PIL import Image; Image.open('app/frontend/public/fragmenta_icon_1024.png').save('app/frontend/public/fragmenta.ico', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"

# 4. Freeze the launcher with the bundled standalone Python (ships Tkinter).
if (Test-Path $WIN) { Remove-Item -Recurse -Force $WIN }
& $VenvPy -m PyInstaller packaging\windows\launcher.spec `
    --distpath $WIN --workpath (Join-Path $DIST "_pyi")

# 5. Inno Setup installer.
iscc /DAppVersion=$VERSION packaging\windows\fragmenta.iss
Write-Host "==> done: $DIST\Fragmenta-$VERSION-Setup.exe"
