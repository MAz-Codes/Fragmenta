# PyInstaller spec — freezes packaging/launcher.py into fragmenta.exe (onedir).
#
# Build from the repo root via packaging/windows/build_exe.ps1, which freezes
# with the bundled standalone Python (it ships Tkinter, so the launcher's
# first-run splash is captured) rather than the build box's Python. Equivalent
# manual call, run under that interpreter's venv:
#   <bundle>/runtime/python/python.exe -m PyInstaller packaging/windows/launcher.spec --distpath packaging/build/win
#
# Produces packaging/build/win/fragmenta/fragmenta.exe (+ _internal/). The Inno
# Setup script then puts that folder together with the assembled bundle
# (runtime\ + payload\) so the launcher finds them beside the exe.
#
# Icon: app/frontend/public/fragmenta.ico (committed).
import os
import re

from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
)

# PyInstaller reads relative paths from this file's folder; SPECPATH is that folder.
ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
ICON = os.path.join(ROOT, "app", "frontend", "public", "fragmenta.ico")

# Windows wants four numbers: X.Y.Z.N, N = the rc number, 100 for a final release,
# so 2.0.0-rc.1 < 2.0.0-rc.2 < 2.0.0.
VERSION = open(os.path.join(ROOT, "VERSION"), encoding="utf-8").read().strip()
_m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?", VERSION)
if not _m:
    raise SystemExit(f"VERSION {VERSION!r} isn't X.Y.Z or X.Y.Z-rc.N")
NUMBERS = (int(_m[1]), int(_m[2]), int(_m[3]), int(_m[4] or 100))

VERSION_INFO = VSVersionInfo(
    ffi=FixedFileInfo(filevers=NUMBERS, prodvers=NUMBERS),
    kids=[
        StringFileInfo([StringTable("040904B0", [
            StringStruct("CompanyName", "Misagh Azimi"),
            StringStruct("FileDescription", "Fragmenta"),
            StringStruct("FileVersion", VERSION),
            StringStruct("InternalName", "fragmenta"),
            StringStruct("OriginalFilename", "fragmenta.exe"),
            StringStruct("ProductName", "Fragmenta"),
            StringStruct("ProductVersion", VERSION),
        ])]),
        VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
    ],
)

a = Analysis(
    [os.path.join(ROOT, "packaging", "launcher.py")],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fragmenta",
    console=False,            # windowed launcher (no console flash)
    icon=ICON,
    version=VERSION_INFO,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="fragmenta",
)
