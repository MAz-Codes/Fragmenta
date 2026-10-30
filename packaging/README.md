# Fragmenta packaging

Builds the **macOS `.dmg`** (Apple Silicon) and **Windows `.exe`** installer for
the desktop distribution described in [`../distribution.md`](../distribution.md).

Linux is not packaged — it builds from source (`./fragmenta.sh`).

## The model

A **thin native launcher** is frozen with PyInstaller. It carries no torch — it
just locates the bundle next to itself (`runtime/python`, `runtime/uv`,
`payload/`), sets `FRAGMENTA_PACKAGED=1`, and runs `payload/install.py --launch`.
On first run `install.py` builds a venv in the user-data dir and syncs it with
the bundled uv to the platform's hashed lock (`payload/locks/`); later launches
are the fast idempotent path.

```
launcher.py ──frozen──▶ Fragmenta.app / fragmenta.exe
                        └─ runs ─▶ runtime/python  payload/install.py --launch
                                   └─ runtime/uv pip sync locks/<platform>.txt
                                                 └─ venv in user-data dir ─▶ start.py
```

No directory name carries a version, so an update replaces files in place.
Read-only bundle code + writable user-data dir (`~/Library/Application Support/FragmentaDesktop`,
`%LOCALAPPDATA%\FragmentaDesktop`, `$XDG_DATA_HOME/FragmentaDesktop`) is handled by
`install.py` / `app/core/config.py` keying off `FRAGMENTA_PACKAGED`.

## Pieces

| File | Role |
|---|---|
| `launcher.py` | Stdlib-only launcher; PyInstaller target. `--self-test` installs, checks the app works and exits 0 or not, with no window and no model download (`logs/self-test.json` has the details). |
| `python_standalone.py` | Download `python-build-standalone` CPython 3.14 into `runtime/python`. Version + per-target SHA-256 pinned at the top; a mismatch is refused. |
| `uv_standalone.py` | Download the pinned uv into `runtime/uv`, same pinning. |
| `fetch.py` | Checksum-verified download with retries; safe extraction (`filter="data"`). |
| `lock.py` | Regenerate the hashed dependency locks in `../locks/` from `requirements.txt`. |
| `assemble.py` | Build the staged **bundle**: `runtime/` (Python + uv) and `payload/` (app code via `git archive`, locks, `manifest.json`). |
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

The script assembles the bundle, generates the `.ico`, freezes the launcher,
and runs Inno Setup. Like the macOS build it freezes with the **bundled
standalone Python** (which ships Tkinter) via a throwaway venv, so the first-run
splash is captured without the build box needing Tk — and PyInstaller never rides
along in the shipped payload Python.

```bash
# Inspect the bundle only (any OS, no Python/uv download):
python packaging/assemble.py --skip-runtime --out /tmp/bundle
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
