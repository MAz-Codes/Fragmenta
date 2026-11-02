<#
Build Fragmenta-<ver>-Windows-x64-Setup.exe. Runs on Windows; CI runs it too.

Steps:
  1. Assemble the bundle (runtime\python + runtime\uv.exe + payload\).
  2. Freeze packaging/launcher.py -> fragmenta.exe, with a version resource.
  3. Wrap the frozen launcher + bundle into a per-user installer with Inno Setup.

The launcher's first-run window uses Tkinter, so it is frozen with the SAME
standalone CPython we ship (python-build-standalone carries Tcl/Tk) and the
build box needs no Python with Tk. PyInstaller lives in a throwaway venv built
from that interpreter, so it never rides along in the shipped Python.

Prereqs: git, any Python 3.11+ on PATH (runs assemble.py), Inno Setup 6. Run
from anywhere; paths resolve to the repo root. assemble.py uses `git archive
HEAD`, so commit changes first. Unsigned (D22).
#>
$ErrorActionPreference = "Stop"

$REPO    = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VERSION = (Get-Content (Join-Path $REPO "VERSION") -Raw).Trim()
$DIST    = Join-Path $REPO "packaging\build"
$BUNDLE  = Join-Path $DIST "bundle"
$WIN     = Join-Path $DIST "win"
$Ref     = if ($env:FRAGMENTA_REF) { $env:FRAGMENTA_REF } else { "HEAD" }

function Invoke-Checked {
    # Native commands don't stop the script on failure by themselves.
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "$($args[0]) failed with exit $LASTEXITCODE" }
}

Write-Host "==> Fragmenta $VERSION - Windows x64 build"
Set-Location $REPO

$Iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $Iscc) { $Iscc = Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe" }
if (-not (Test-Path $Iscc)) { throw "Inno Setup 6 not found (iscc)" }

# 1. Bundle.
Invoke-Checked python packaging\assemble.py --target windows-x64 --out $BUNDLE --ref $Ref

# 2. Freeze the launcher.
$PbsPy     = Join-Path $BUNDLE "runtime\python\python.exe"
$BuildVenv = Join-Path $DIST "_buildvenv"
& $PbsPy -c "import tkinter" 2>$null
if ($LASTEXITCODE -ne 0) { throw "the bundled Python has no Tkinter, so the launcher window would be missing" }
if (Test-Path $BuildVenv) { Remove-Item -Recurse -Force $BuildVenv }
Invoke-Checked $PbsPy -m venv $BuildVenv
$VenvPy = Join-Path $BuildVenv "Scripts\python.exe"
Invoke-Checked $VenvPy -m pip install --quiet pyinstaller==6.22.3
if (Test-Path $WIN) { Remove-Item -Recurse -Force $WIN }
Invoke-Checked $VenvPy -m PyInstaller --noconfirm packaging\windows\launcher.spec `
    --distpath $WIN --workpath (Join-Path $DIST "_pyi")

# 3. Installer. Windows' file version is X.Y.Z.N (N = rc number, 100 = final),
# the same numbers launcher.spec writes into fragmenta.exe.
if ($VERSION -notmatch '^(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$') { throw "VERSION $VERSION isn't X.Y.Z or X.Y.Z-rc.N" }
$N = if ($Matches[4]) { $Matches[4] } else { "100" }
$VersionInfo = "$($Matches[1]).$($Matches[2]).$($Matches[3]).$N"
Invoke-Checked $Iscc /Q "/DAppVersion=$VERSION" "/DVersionInfo=$VersionInfo" packaging\windows\fragmenta.iss

$Setup = Join-Path $DIST "Fragmenta-$VERSION-Windows-x64-Setup.exe"
(Get-FileHash -Algorithm SHA256 $Setup).Hash.ToLower() + "  " + (Split-Path $Setup -Leaf)
Write-Host "==> done: $Setup"
