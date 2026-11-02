# Fragmenta packaging

Builds the downloads for all three systems:

| System | File |
|---|---|
| macOS (Apple Silicon, 14+) | `Fragmenta-<ver>-macOS-arm64.dmg` |
| Windows (x64, 10 1809+) | `Fragmenta-<ver>-Windows-x64-Setup.exe` (per-user, no admin) |
| Linux (x86_64, glibc 2.28+) | `Fragmenta-<ver>-Linux-x86_64.AppImage` + `.tar.gz` of the same tree |

All unsigned (D22): macOS gets an ad-hoc signature, which Apple Silicon needs to
run anything; Windows shows SmartScreen's "More info → Run anyway".

## The model

A thin launcher carries no torch. It finds the bundle next to itself
(`runtime/python`, `runtime/uv`, `payload/`), sets `FRAGMENTA_PACKAGED=1`, and runs
`payload/install.py --launch`. On first run `install.py` builds a venv in the
user-data dir and syncs it with the bundled uv to the platform's hashed lock
(`payload/locks/`); later launches take the fast path.

```
launcher ─▶ runtime/python  payload/install.py --launch
            └─ runtime/uv pip sync locks/<platform>.txt ─▶ venv in user-data dir ─▶ start.py
```

The launcher is `packaging/launcher.py`: frozen with PyInstaller on macOS
(`Fragmenta.app/Contents/MacOS/fragmenta`, bundle in `Contents/Resources/fragmenta/`)
and Windows (`fragmenta.exe`, bundle beside it); run as a script by the AppImage's
`AppRun` on Linux. No directory name carries a version, so an update replaces
files in place. User data lives outside the install: `~/Library/Application Support/FragmentaDesktop`,
`%LOCALAPPDATA%\FragmentaDesktop`, `$XDG_DATA_HOME/FragmentaDesktop`.

## Pieces

| File | Role |
|---|---|
| `launcher.py` | Stdlib-only launcher. `--self-test` installs, checks the app works and exits 0 or not, with no window and no model download (`logs/self-test.json` has the details). |
| `python_standalone.py`, `uv_standalone.py` | Download the pinned standalone CPython 3.14 and uv; per-target SHA-256 pinned, a mismatch is refused. |
| `fetch.py` | Checksum-verified download with retries; safe extraction. |
| `lock.py` | Regenerate the hashed locks in `../locks/` from `requirements.txt`; `--check` fails if they're stale. |
| `assemble.py` | Build the bundle: `runtime/` + `payload/` (app code via `git archive`, locks, `manifest.json`). |
| `make_manifest.py` | `latest.json` for the in-app update notifier. |
| `macos/build_dmg.sh` | `.app` + ad-hoc signature + `.dmg`. `entitlements.plist` is only for a future Developer ID build. |
| `windows/build_exe.ps1` | Bundle → freeze (`launcher.spec`, with version info) → Inno Setup (`fragmenta.iss`). |
| `linux/build_appimage.sh` | AppDir → tarball + AppImage (pinned appimagetool + runtime). |
| `../.github/workflows/release.yml` | CI: all three builds, a self-test of each, draft release. |
| `../VERSION` | Single source of the version. |

Build output goes to `packaging/build/` (gitignored).

## Building with CI (normal way)

Pushing a tag on a `dev/2.0` commit starts `release.yml`. It checks that the
locks and the committed frontend build are current, builds the three downloads,
installs each on a fresh GitHub machine and runs `--self-test` there.

- `build-dry-<anything>`: files only, attached to the run (Actions → the run →
  Artifacts). Delete the tag afterwards.
- `build-v<VERSION>` (e.g. `build-v2.0.0-rc.1`): also a **draft** release
  `v<VERSION>` with the files, `SHA256SUMS` and `latest.json`. An `-rc.N` version
  is marked pre-release.

```bash
git tag build-dry-1 && git push origin build-dry-1
gh run watch                               # or the Actions tab
git push origin :build-dry-1 && git tag -d build-dry-1
```

## Release checklist

1. On `dev/2.0`: set `VERSION`, add the `## [<VERSION>]` section to `CHANGELOG.md`
   (`latest.json` takes its highlights from it), `npm run build` in `app/frontend`
   and commit the build.
2. If `requirements.txt` changed: `python packaging/lock.py`, commit `locks/`.
3. `git tag build-v<VERSION> && git push origin build-v<VERSION>` → draft release.
4. Test the downloads on real machines (Mac, Windows + NVIDIA, Linux + NVIDIA).
5. rc: write the notes and publish the draft as a pre-release on the `dev/2.0` commit.
   Final: open the `release/X.Y` PR into `main` without the dev-only paths; after
   it's merged, retarget the draft to the merge commit, write the notes, publish.
   Check `latest.json` first: CI writes it like 1.3's, with no `requires`.
6. Update the HF Space in place.

## Building by hand (same scripts)

Each runs on its own system. `assemble.py` uses `git archive HEAD`, so commit first.

```bash
bash packaging/macos/build_dmg.sh          # Mac
bash packaging/linux/build_appimage.sh     # any x86_64 Linux
```

```powershell
# Windows: needs git, Python 3.11+ and Inno Setup 6
powershell -ExecutionPolicy Bypass -File packaging\windows\build_exe.ps1
```

Then `--self-test` on the result: `Fragmenta.app/Contents/MacOS/fragmenta --self-test`,
`fragmenta.exe --self-test` (prints nothing; read `logs\self-test.json`),
`./Fragmenta-*.AppImage --self-test`.
