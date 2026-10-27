# Fragmenta packaging

Builds the **macOS `.dmg`** (Apple Silicon) and **Windows `.exe`** installer for
the desktop distribution described in [`../distribution.md`](../distribution.md).

Linux is not packaged — it builds from source (`./fragmenta.sh`).

## The model

A **thin native launcher** is frozen with PyInstaller. It carries no torch — it
just locates the bundled **standalone Python 3.11** and the **app code** next to
itself, sets `FRAGMENTA_PACKAGED=1`, and runs `install.py --launch`. On first
run `install.py` builds a venv in the user-data dir and pip-installs the deps
(torch/SA3/flash-attn resolved for the real machine); later launches are the
fast idempotent path.

```
launcher.py ──frozen──▶ Fragmenta.app / fragmenta.exe
                        └─ runs ─▶ python-3.11/  install.py --launch
                                                 └─ venv in user-data dir ─▶ start.py
```

Read-only bundle code + writable user-data dir (`~/Library/Application Support/FragmentaDesktop`,
`%APPDATA%\FragmentaDesktop`) is handled by `install.py` / `app/core/config.py`
keying off `FRAGMENTA_PACKAGED`.

## Pieces

| File | Role |
|---|---|
| `launcher.py` | Stdlib-only launcher; PyInstaller target. |
| `python_standalone.py` | Resolve/download `python-build-standalone` CPython 3.11. Pins at the top — bump there. |
| `assemble.py` | Build the staged **payload** (app code via `git archive` + standalone Python). |
| `macos/build_dmg.sh` | `.app` + sign + notarize + `.dmg`. |
| `macos/Info.plist.in`, `macos/entitlements.plist` | App metadata; hardened-runtime entitlements (needed for unsigned pip dylibs). |
| `windows/build_exe.ps1` | Assemble → freeze → Inno Setup, end to end. |
| `windows/launcher.spec` | PyInstaller spec → `fragmenta.exe`. |
| `windows/fragmenta.iss` | Inno Setup per-user installer (+ WebView2 check). |
| `../VERSION` | Single source of truth for the version. |

Build output goes to `packaging/build/` (gitignored).

## Building (run on the target OS)

Builds are produced **locally on each OS** — a Mac for the `.dmg`, a Windows
box for the `.exe`. `assemble.py` uses `git archive HEAD`, so commit your
changes on the build machine first (`git pull` the repo there).

```bash
# macOS (on a Mac):
bash packaging/macos/build_dmg.sh            # -> packaging/build/Fragmenta-<ver>-macOS-arm64.dmg
```

```powershell
# Windows (PowerShell; needs git + Inno Setup with iscc on PATH):
powershell -ExecutionPolicy Bypass -File packaging\windows\build_exe.ps1
# -> packaging\build\Fragmenta-<ver>-Setup.exe
```

The script assembles the payload, generates the `.ico`, freezes the launcher,
and runs Inno Setup. Like the macOS build it freezes with the **bundled
standalone Python** (which ships Tkinter) via a throwaway venv, so the first-run
splash is captured without the build box needing Tk — and PyInstaller never rides
along in the shipped payload Python.

```bash
# Inspect the payload only (any OS, no Python download):
python packaging/assemble.py --skip-python --out /tmp/payload
```

### Signing (optional)

Set these env vars before building for a signed/notarized artifact; leave them
unset for an unsigned local build:

- macOS: `MAC_SIGN_IDENTITY` (and `AC_APPLE_ID`, `AC_TEAM_ID`, `AC_PASSWORD` to notarize)
- Windows: sign `Fragmenta-<ver>-Setup.exe` with `signtool` using your code-signing cert.

## First-run note for users

First launch needs internet and a few minutes (pip pulls the GBs; models
download via the Checkpoint Manager). User data (models, projects, output)
lives outside the install dir and survives uninstall.
