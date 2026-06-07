# PyInstaller spec — freezes packaging/launcher.py into fragmenta.exe (onedir).
#
# Build from the repo root via packaging/windows/build_exe.ps1, which freezes
# with the bundled standalone Python (it ships Tkinter, so the launcher's
# first-run splash is captured) rather than the build box's Python. Equivalent
# manual call, run under that interpreter's venv:
#   <payload>/python-3.11/python.exe -m PyInstaller packaging/windows/launcher.spec --distpath packaging/build/win
#
# Produces packaging/build/win/fragmenta/fragmenta.exe (+ _internal/). The Inno
# Setup script then bundles that folder together with the assembled payload
# (app code + python-3.11) so the launcher finds the payload beside the exe.
#
# Icon: app/frontend/public/fragmenta.ico — generate it from fragmenta_icon_1024.png
# before building (see packaging/README.md). If absent, the exe builds without an icon.
import os

ICON = os.path.join("app", "frontend", "public", "fragmenta.ico")

a = Analysis(
    ["packaging/launcher.py"],
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
    icon=ICON if os.path.exists(ICON) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="fragmenta",
)
