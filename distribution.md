# Fragmenta — Desktop Distribution Plan (.dmg + .exe)

**Status:** Plan / not yet built. Revisit when ready to ship.
**Scope:** macOS `.dmg` and Windows `.exe` only. **Linux = build from source** (the existing `fragmenta.sh` + `install.py` path; documented in the README).
**Companion:** [SA3_INTEGRATION_PLAN.md](SA3_INTEGRATION_PLAN.md), [SA3_COMPLETION_PLAN.md](SA3_COMPLETION_PLAN.md).

---

## 0. TL;DR

Ship a **thin bootstrapper launcher** (not a frozen torch bundle) that brings its **own Python 3.11**, lays the app code down next to it, and on first run does exactly what `install.py` + `start.py` already do: create a venv, `pip install` the deps (pip resolves the right wheels for the actual machine), then launch.

- The launcher is a tiny native executable (a frozen, stdlib-only `install.py`), so there's **no torch-freezing nightmare** — torch/SA3/flash-attn are installed by pip at runtime, exactly as today.
- A bundled **standalone Python 3.11** means the user needs nothing preinstalled.
- The app code (vendored SA3, built frontend, `start.py`, `install.py`, `requirements.txt`, …) is **shipped inside the installer/zip next to the launcher** (decided).
- macOS → `Fragmenta.app` inside a `.dmg`; Windows → an Inno Setup/NSIS installer `.exe`.

Trade-off accepted: **first run needs internet + a few minutes** (pip pulls the GBs; models download via the Checkpoint Manager). Every later launch is the fast idempotent path (the `requirements.txt` hash stamp in `install.py`).

---

## 1. Why this approach (and not a frozen app)

Freezing the whole app is painful **because of torch**: collecting its binaries/CUDA libs, hidden-import hooks, a 2.5–5 GB artifact, and a CUDA build that can't match an arbitrary user's driver. The bootstrapper sidesteps all of it:

| | Frozen torch bundle | **Bootstrapper launcher (this plan)** |
|---|---|---|
| Artifact size | 2.5–5 GB | Launcher ~10–40 MB (+ bundled app code) |
| torch / CUDA correctness | Frozen at build time; can't match user GPU | **pip resolves at runtime for the real machine** |
| flash-attn | Fragile to bundle | Not bundled; pip handles it where applicable |
| Hidden-import hell | Severe (torch/transformers/SA3) | None — launcher is stdlib-only |
| Signing | Sign a 2.5 GB blob | Sign a small launcher (same effort, faster) |
| First run | Offline, instant | Needs internet + minutes (pip + models) |

GPU still works: pip installs the CUDA torch wheel on NVIDIA machines, and `AudioGenerator._autodetect_device()` (`cuda → mps → cpu`) falls back automatically for everyone else. Small models need no flash-attn on any platform; medium stays Linux+CUDA+flash-attn (source/Docker), which the UI already greys out elsewhere.

---

## 2. Runtime architecture (read-only code, writable data)

A signed `.app` and a Program-Files install are **read-only** where the bundled code sits, so the split is:

```
<bundle / install dir>            (read-only after install/sign)
├── launcher (native exe)         ← frozen install.py + bundled Python 3.11
├── python-3.11/                  ← standalone Python (python-build-standalone)
├── app/  vendor/  utils/         ← shipped app code
├── start.py  install.py  requirements.txt  models/config ...
└── app/frontend/build/           ← prebuilt React (served by Flask)

<user data dir>                   (writable; created at runtime)
├── venv/                         ← created by install.py with the bundled Python
├── models/pretrained/            ← downloaded via Checkpoint Manager
├── output/  logs/  config/  projects/
```

User data dir (already computed by [config.py](app/core/config.py) in frozen mode):
- macOS: `~/Library/Application Support/FragmentaDesktop`
- Windows: `%APPDATA%\FragmentaDesktop`

---

## 3. Prerequisite code change (the one real dependency)

**`install.py` must put the venv in the user data dir, not `PROJECT_ROOT/venv`,** when running from a packaged build — otherwise venv creation fails against the read-only bundle.

- Detect packaged mode (e.g. an env var the launcher sets, or `sys.frozen` / a sentinel file beside the launcher).
- Resolve `VENV_PATH` to `<user_data_dir>/venv` in that mode; keep `PROJECT_ROOT/venv` for source/dev runs.
- Reuse `config.py`'s `user_data_dir` logic so the path matches where models/output already go.
- The bundled standalone Python becomes the interpreter `install.py` uses to build the venv (instead of "find Python 3.11 on PATH").
- `start.py` is launched with the venv's Python, unchanged.

Everything else in `install.py` (idempotent stamp, dep install, laion-clap `--no-deps`, verify) is already correct and reused as-is.

---

## 4. Bundled Python

Use **`python-build-standalone`** (Astral) — self-contained, relocatable CPython 3.11 builds (~25–40 MB per platform):
- macOS: `aarch64-apple-darwin` (Apple Silicon; add `x86_64-apple-darwin` only if Intel support is needed).
- Windows: `x86_64-pc-windows-msvc`.

The launcher invokes this interpreter to run `install.py`. No "install Python first" step for the user. (Linux-from-source keeps the current find/auto-install-3.11 logic.)

---

## 5. macOS `.dmg` pipeline

1. **Launcher → `Fragmenta.app`.** Build a small `.app` whose executable runs the bundled Python on `install.py --launch`. Tooling: PyInstaller (`--windowed`) on a stdlib-only entry script, or a hand-rolled `.app` bundle. Set the icon (`app/frontend/public/fragmenta_icon_1024.png` → `.icns`), bundle id, `LSMinimumSystemVersion`.
2. **Lay app code into `Fragmenta.app/Contents/Resources/`** (vendor, app, utils, start.py, install.py, requirements.txt, frontend/build, models/config) + the standalone Python under `Resources/python-3.11/`.
3. **Wrap in `.dmg`** with `create-dmg` (background image, drag-to-Applications layout).
4. **Sign + notarize** (see §8).
5. Webview: macOS uses WKWebView (built in) — no extra runtime.

Output: `Fragmenta-<version>-macOS-arm64.dmg`.

---

## 6. Windows `.exe` pipeline

1. **Launcher → `fragmenta.exe`.** PyInstaller one-file/one-dir on the stdlib-only entry script that runs bundled Python on `install.py --launch`. Icon from `app/frontend/public/fragmenta.ico`.
2. **Installer `.exe`** via **Inno Setup** (or NSIS): bundles `fragmenta.exe` + the standalone Python + the app code; installs to a user-writable location (per-user `%LOCALAPPDATA%\Programs\Fragmenta` to avoid admin/Program-Files read-only issues); creates Start-menu + optional desktop shortcuts; registers an uninstaller.
3. **WebView2 runtime:** present on most Win10/11. The installer should check for the Evergreen WebView2 runtime and offer to fetch the bootstrapper if missing.
4. **Code-sign** the launcher + installer (see §8) to avoid SmartScreen friction.

Output: `Fragmenta-<version>-Setup.exe`.

---

## 7. Linux (out of scope for binaries)

No `.AppImage`/`.deb`. Document in the README:
- Clone/download the repo, run `./fragmenta.sh` (auto-installs Python 3.11 + WebKitGTK via apt/dnf/pacman, then `install.py --launch`).
- This is also the path for the **medium** model (Linux + CUDA + flash-attn).

---

## 8. Signing & notarization

- **macOS:** Apple Developer ID cert ($99/yr). `codesign` the `.app` (hardened runtime), `notarytool` submit the `.dmg`, `stapler staple`. Without this, Gatekeeper blocks the app.
- **Windows:** Authenticode code-signing cert (OV/EV). Unsigned runs but triggers SmartScreen warnings; EV clears reputation faster. Optional for a first internal release, recommended for public.
- Secrets (certs, Apple app-specific password / API key) live in CI secrets, never in the repo.

---

## 9. Build infrastructure

- **Must build on each OS** (no cross-compile). Use a GitHub Actions matrix: `macos-latest` (Apple Silicon runners) for the `.dmg`, `windows-latest` for the `.exe`.
- Per-job: check out → fetch standalone Python → assemble bundle (app code + Python + launcher) → package (`create-dmg` / Inno Setup) → sign/notarize → upload artifact / attach to a GitHub Release.
- Keep these build scripts in a tracked location (NOT `scripts/`, which is gitignored) — e.g. `packaging/macos/` and `packaging/windows/`.

---

## 10. First-run UX

1. User downloads `Fragmenta-...dmg` / `...Setup.exe`, installs, double-clicks.
2. Launcher shows a minimal "Setting up Fragmenta (first run, a few minutes)…" state while `install.py` builds the venv + pip-installs.
3. App opens to the Checkpoint Manager (no model bundled) → user signs into HF, accepts the license, downloads Small — Music (~2.3 GB).
4. Subsequent launches: stamp matches → no reinstall → straight to the app.

Consider surfacing pip/setup progress in the launcher window (or a console) so the first-run wait isn't a blank screen.

---

## 11. Open decisions / risks

- [ ] **Packaged-mode detection** in `install.py`: env var set by the launcher vs `sys.frozen` vs sentinel file. (Env var is simplest and explicit.)
- [ ] **macOS Intel:** ship `x86_64` too, or Apple-Silicon-only? (Arm-only is simpler; most current Macs.)
- [ ] **First-run progress UI:** plain console window vs a small native splash vs a pre-`start.py` minimal webview.
- [ ] **Version stamping:** single source of truth for the version across `.dmg`/`.exe` names, the app, and the installer.
- [ ] **Offline/air-gapped users:** out of scope — first run requires internet (pip + models). Document clearly.
- [ ] **Disk space:** venv + models can exceed 5–10 GB; the installer should note this.
- [ ] **WebView2 absence on Windows:** decide auto-fetch vs prompt.

---

## 12. Phased checklist (when ready)

1. [ ] `install.py`: packaged-mode detection + venv → user data dir; use bundled Python as the venv interpreter. Verify on a plain unzipped folder first (no packaging).
2. [ ] Standalone Python fetch + bundle assembly script (shared logic, per-OS thin wrappers) under `packaging/`.
3. [ ] macOS: `.app` build + `create-dmg` + codesign/notarize (on a Mac / `macos-latest`).
4. [ ] Windows: launcher `.exe` + Inno Setup installer + WebView2 check + code-sign (on Windows / `windows-latest`).
5. [ ] First-run progress UX.
6. [ ] GitHub Actions release workflow (matrix → signed artifacts → Release).
7. [ ] README: "Download" section (mac/Windows) + "Run from source (Linux / advanced)".
8. [ ] Smoke-test each artifact on a clean machine/VM: install → first-run setup → download Small — Music → generate.

**Definition of done:** on a clean macOS and a clean Windows machine, a non-technical user double-clicks the downloaded artifact, waits out first-run setup, downloads Small — Music from the Checkpoint Manager, and generates a clip — with no terminal, no Python install, no manual steps.
