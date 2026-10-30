"""In-app update notices.

Once a day (and on demand) the backend asks GitHub whether a newer Fragmenta
release exists, caches the answer in ``<user_data>/config/update_state.json``,
and the UI reads only that cache. Nothing personal is sent: one unauthenticated
GET with no IDs or query parameters. Never blocks startup, never downloads or
installs anything, and is off in the Hugging Face Space and with
``FRAGMENTA_NO_UPDATE_CHECK=1``.

Source of truth is the ``latest.json`` manifest attached to each release (see
redesign/UPDATE_NOTIFIER.md for the schema), with GitHub's releases API as the
fallback for releases that have none. The client shipped in a release keeps
running unchanged for months, so anything that may need to change later is
manifest *data* — but only narrow data: the manifest can switch between
predefined messages and supply URLs that pass ``is_allowed_url``; it can't
inject text or commands.

Test hooks (env): FRAGMENTA_UPDATE_MANIFEST_URL, FRAGMENTA_UPDATE_API_URL and
FRAGMENTA_UPDATE_DOWNLOAD_BASE override the GitHub endpoints; ``file://`` URLs
are read from disk.
"""
import json
import os
import platform
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlsplit

import requests
from packaging.version import InvalidVersion, Version

from utils.logger import get_logger

logger = get_logger("UpdateCheck")

REPO_PATH = "/MAz-Codes/Fragmenta/"
MANIFEST_URL = os.environ.get(
    "FRAGMENTA_UPDATE_MANIFEST_URL",
    "https://github.com/MAz-Codes/Fragmenta/releases/latest/download/latest.json")
API_URL = os.environ.get(
    "FRAGMENTA_UPDATE_API_URL", "https://api.github.com/repos/MAz-Codes/Fragmenta")
DOWNLOAD_BASE = os.environ.get(
    "FRAGMENTA_UPDATE_DOWNLOAD_BASE", "https://github.com/MAz-Codes/Fragmenta/releases/download")

CHECK_INTERVAL = 24 * 3600
FIRST_CHECK_DELAY = 30           # don't compete with startup / model load
MANUAL_CHECK_MIN_GAP = 60        # rate limit for POST /api/update/check
SNOOZE_SECONDS = 3 * 24 * 3600
TIMEOUT = (3.05, 6)
RELEASE_TAG = re.compile(r"^v\d+\.\d+\.\d+")   # skips deps-… and flash-attn-win-… tags

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_lock = threading.RLock()
_last_manual_check = 0.0


# --- version / environment ---------------------------------------------------

def current_version() -> Optional[str]:
    try:
        v = (_PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
        Version(v)
        return v
    except (OSError, InvalidVersion):
        return None


def _running_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def channel() -> str:
    if os.environ.get("FRAGMENTA_PACKAGED") == "1":
        return "packaged"
    if os.environ.get("FRAGMENTA_DOCKER") == "1":
        return "docker"
    # A worktree's .git is a file, a clone's a directory — both count.
    return "source-git" if (_PROJECT_ROOT / ".git").exists() else "source-zip"


def artifact_key() -> str:
    if sys.platform == "darwin":
        return "macos-arm64"
    if sys.platform == "win32":
        return "windows-x64"
    return "linux-x86_64-appimage" if os.environ.get("APPIMAGE") else "linux-x86_64-tar"


def _launcher_name() -> str:
    return {"win32": "fragmenta.bat", "darwin": "fragmenta.command"}.get(sys.platform, "./fragmenta.sh")


# --- state file --------------------------------------------------------------

_DEFAULT_STATE: Dict[str, Any] = {
    "enabled": True, "include_prereleases": False,
    "last_checked": 0, "last_ok": 0, "last_error": None,
    "latest": None, "skipped_version": None, "snoozed_until": 0,
    "last_seen_version": None,
}


def _state_path() -> Path:
    from app.core.config import get_config
    return get_config().user_data_dir / "config" / "update_state.json"


def load_state() -> Dict[str, Any]:
    with _lock:
        try:
            data = json.loads(_state_path().read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not an object")
        except FileNotFoundError:
            data = {}
        except (OSError, ValueError) as exc:
            logger.warning(f"Update state unreadable ({exc}); starting fresh")
            data = {}
        return {**_DEFAULT_STATE, **{k: v for k, v in data.items() if k in _DEFAULT_STATE}}


def save_state(state: Dict[str, Any]) -> None:
    with _lock:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def _update_state(**changes: Any) -> Dict[str, Any]:
    with _lock:
        state = load_state()
        state.update(changes)
        save_state(state)
        return state


# --- disabled reasons --------------------------------------------------------

def is_disabled(state: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Why update checks can't run at all, or None. The user's switch in
    About is not one of these: it only turns off the *automatic* daily
    check — "Check now" still works (see check())."""
    if os.environ.get("FRAGMENTA_HF_SPACE") == "1" or os.environ.get("SPACE_ID"):
        return "hf_space"
    if os.environ.get("FRAGMENTA_NO_UPDATE_CHECK") == "1":
        return "env"
    if current_version() is None:
        return "unknown_version"
    return None


# --- URL allowlist -----------------------------------------------------------

def is_allowed_url(url: Any) -> bool:
    """Only https://github.com/MAz-Codes/Fragmenta/… URLs, parsed and
    normalised. A prefix check would accept …/Fragmenta/../../evil/repo,
    which browsers resolve to another repository."""
    if not isinstance(url, str) or len(url) > 500 or "\\" in url:
        return False
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    if parts.scheme != "https" or parts.hostname != "github.com":
        return False
    if parts.username or parts.password or port is not None or "@" in parts.netloc:
        return False
    raw = parts.path
    if re.search(r"%2e|%2f|%5c", raw, re.IGNORECASE):
        return False
    segments = unquote(raw).split("/")
    if any(s in (".", "..") for s in segments):
        return False
    return raw.lower().startswith(REPO_PATH.lower())


# --- manifest ----------------------------------------------------------------

def _clip(value: Any, limit: int = 200) -> Optional[str]:
    return value.strip()[:limit] if isinstance(value, str) and value.strip() else None


def validate(manifest: Any) -> Optional[Dict[str, Any]]:
    """The subset of a manifest we trust, or None. URLs that fail the
    allowlist are dropped rather than failing the whole manifest."""
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        return None
    try:
        version = str(Version(str(manifest.get("version", ""))))
    except InvalidVersion:
        return None
    out: Dict[str, Any] = {"schema": 1, "version": version}
    for key in ("notes_url", "upgrade_guide_url"):
        if is_allowed_url(manifest.get(key)):
            out[key] = manifest[key]
    released = _clip(manifest.get("released_at"), 40)
    if released:
        out["released_at"] = released
    highlights = manifest.get("highlights")
    if isinstance(highlights, list):
        out["highlights"] = [h for h in (_clip(x, 300) for x in highlights[:5]) if h]
    out["critical"] = manifest.get("critical") is True
    out["reinstall_required"] = manifest.get("reinstall_required") is True
    try:
        if manifest.get("min_supported"):
            out["min_supported"] = str(Version(str(manifest["min_supported"])))
    except InvalidVersion:
        pass
    requires = manifest.get("requires")
    if isinstance(requires, dict) and re.fullmatch(r"\d+\.\d+", str(requires.get("python", ""))):
        out["requires"] = {"python": str(requires["python"])}
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        clean = {}
        for key, art in list(artifacts.items())[:10]:
            if isinstance(art, dict) and is_allowed_url(art.get("url")):
                entry = {"url": art["url"]}
                if re.fullmatch(r"[0-9a-f]{64}", str(art.get("sha256", ""))):
                    entry["sha256"] = art["sha256"]
                if isinstance(art.get("size"), int):
                    entry["size"] = art["size"]
                clean[str(key)[:40]] = entry
        out["artifacts"] = clean
    return out


def _get(url: str, accept: str = "application/json") -> requests.Response:
    headers = {"User-Agent": f"Fragmenta/{current_version()} ({platform.system()})", "Accept": accept}
    return requests.get(url, timeout=TIMEOUT, headers=headers)


def _get_json(url: str, accept: str = "application/json") -> Any:
    """JSON from a URL; raises on HTTP errors or bad JSON. file:// for tests."""
    if url.startswith("file://"):
        return json.loads(Path(url[len("file://"):]).read_text(encoding="utf-8"))
    resp = _get(url, accept)
    resp.raise_for_status()
    return resp.json()


def _manifest_from_release(rel: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal manifest for a release without latest.json: version, notes link
    and the first bullet lines of its description."""
    bullets = [ln.strip()[2:] for ln in str(rel.get("body") or "").splitlines()
               if ln.strip().startswith(("- ", "* "))]
    return {"schema": 1, "version": str(rel.get("tag_name", "")).lstrip("v"),
            "notes_url": rel.get("html_url"), "released_at": rel.get("published_at"),
            "highlights": bullets[:5]}


def fetch_manifest() -> Dict[str, Any]:
    try:
        return _get_json(MANIFEST_URL)
    except (requests.RequestException, ValueError, OSError) as exc:
        # 404 / not JSON: this release predates latest.json — use the API.
        logger.info(f"latest.json unavailable ({exc}); using the releases API")
        return _manifest_from_release(_get_json(f"{API_URL}/releases/latest", "application/vnd.github+json"))


def fetch_prerelease_manifest() -> Optional[Dict[str, Any]]:
    """Highest v* release including prereleases. releases/latest never returns
    a prerelease, so the beta toggle needs the list (ordered by date, not
    version — hence the sort)."""
    releases = _get_json(f"{API_URL}/releases?per_page=20", "application/vnd.github+json")
    candidates = []
    for rel in releases if isinstance(releases, list) else []:
        tag = str(rel.get("tag_name", ""))
        if rel.get("draft") or not RELEASE_TAG.match(tag):
            continue
        try:
            candidates.append((Version(tag.lstrip("v")), rel))
        except InvalidVersion:
            continue
    if not candidates:
        return None
    _, rel = max(candidates, key=lambda c: c[0])
    if any(a.get("name") == "latest.json" for a in rel.get("assets") or []):
        try:
            return _get_json(f"{DOWNLOAD_BASE}/{rel['tag_name']}/latest.json")
        except (requests.RequestException, ValueError, OSError):
            pass
    return _manifest_from_release(rel)


# --- check -------------------------------------------------------------------

def check(force: bool = False) -> Dict[str, Any]:
    """Refresh the cached answer. Never raises; on failure keeps the previous
    good result and records the error."""
    state = load_state()
    if is_disabled(state):
        return state
    if not force and (not state.get("enabled", True)
                      or time.time() - state.get("last_checked", 0) < CHECK_INTERVAL):
        return state   # automatic checks off, or checked recently
    try:
        raw = fetch_prerelease_manifest() if state.get("include_prereleases") else fetch_manifest()
        latest = validate(raw)
        if latest is None:
            raise ValueError("release manifest failed validation")
        state = _update_state(latest=latest, last_checked=time.time(),
                              last_ok=time.time(), last_error=None)
        logger.info(f"Update check: latest {latest['version']}, running {current_version()}")
    except Exception as exc:   # network, HTTP, JSON, anything — stay silent
        state = _update_state(last_checked=time.time(), last_error=str(exc)[:200])
        logger.info(f"Update check failed: {exc}")
    return state


def manual_check() -> Dict[str, Any]:
    """POST /api/update/check — forced, but at most once a minute."""
    global _last_manual_check
    with _lock:
        if time.time() - _last_manual_check >= MANUAL_CHECK_MIN_GAP:
            _last_manual_check = time.time()
            check(force=True)
    return status()


def upgrade_kind(latest: Dict[str, Any], current: Optional[str] = None) -> str:
    """'major' when a source install can't simply pull the new release: it
    needs a different Python, says reinstall_required, or bumps the major
    version (the safety net for API-fallback manifests, which carry no
    `requires`). Packaged and Docker installs bring their own Python. Errs
    towards 'major' — a needless guide only costs convenience."""
    if not channel().startswith("source"):
        return "in_place"
    if latest.get("reinstall_required"):
        return "major"
    need = (latest.get("requires") or {}).get("python")
    if need and need != _running_python():
        return "major"
    try:
        if Version(latest["version"]).major > Version(current or current_version() or "0").major:
            return "major"
    except InvalidVersion:
        return "major"
    return "in_place"


def instructions(latest: Dict[str, Any], kind: str) -> str:
    ver, ch = latest["version"], channel()
    if ch == "packaged":
        return (f"Download Fragmenta {ver} and install it over this version. "
                "Your models, projects and outputs are kept.")
    if ch == "docker":
        return "Rebuild your Docker image from the new release's source."
    if kind == "major":
        need = (latest.get("requires") or {}).get("python")
        why = f" (it needs Python {need})" if need else ""
        return (f"Fragmenta {ver} is a major upgrade and can't be installed over this "
                f"version with git pull{why}. Download the installer for your system or "
                "follow the upgrade guide. Keep this folder until your models, adapters "
                "and projects are in the new version — the guide explains how.")
    if ch == "source-git":
        return f"Update with `git pull`, then run `{_launcher_name()}`."
    return ("Download the new source release and replace this folder "
            "(keep models/, projects/ and output/).")


def status() -> Dict[str, Any]:
    """What the UI shows. Cache only — never touches the network."""
    state = load_state()
    current = current_version()
    reason = is_disabled(state)
    out: Dict[str, Any] = {
        "enabled": state.get("enabled", True), "disabled_reason": reason,
        "include_prereleases": state.get("include_prereleases", False),
        "current": current, "channel": channel(), "available": False,
        "checked_at": state.get("last_ok") or None, "last_error": state.get("last_error"),
    }
    latest = state.get("latest")
    if reason or not latest or not current:
        return out
    try:
        newer_version = Version(latest["version"])
        newer = newer_version > Version(current)
        if newer_version.is_prerelease and not state.get("include_prereleases"):
            newer = False
        unsupported = bool(latest.get("min_supported")) and Version(current) < Version(latest["min_supported"])
    except InvalidVersion:
        return out
    kind = upgrade_kind(latest, current)
    skipped = state.get("skipped_version") == latest["version"]
    snoozed = time.time() < state.get("snoozed_until", 0)
    out.update({
        "latest": latest["version"], "available": newer, "skipped": skipped, "snoozed": snoozed,
        "critical": latest.get("critical", False), "unsupported": unsupported,
        "highlights": latest.get("highlights", []), "upgrade_kind": kind,
        "instructions": instructions(latest, kind) if newer else None,
        "notes_available": bool(latest.get("notes_url")),
        "guide_available": bool(latest.get("upgrade_guide_url") or latest.get("notes_url")),
        "download_available": bool((latest.get("artifacts") or {}).get(artifact_key())
                                   or latest.get("notes_url")),
    })
    return out


# --- user actions ------------------------------------------------------------

def dismiss(action: str) -> Dict[str, Any]:
    latest = load_state().get("latest") or {}
    if action == "skip" and latest.get("version"):
        _update_state(skipped_version=latest["version"])
    elif action == "snooze":
        _update_state(snoozed_until=time.time() + SNOOZE_SECONDS)
    else:
        raise ValueError("action must be 'skip' or 'snooze'")
    return status()


def update_settings(enabled: Optional[bool] = None, include_prereleases: Optional[bool] = None) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    if isinstance(enabled, bool):
        changes["enabled"] = enabled
    if isinstance(include_prereleases, bool):
        changes["include_prereleases"] = include_prereleases
        changes["last_checked"] = 0   # the answer depends on it — re-check soon
    if changes:
        _update_state(**changes)
    return status()


def resolve_url(target: str) -> Optional[str]:
    """The URL behind a UI button, chosen server-side from validated state —
    the client never supplies a URL."""
    latest = load_state().get("latest") or {}
    notes = latest.get("notes_url")
    if target == "notes":
        url = notes
    elif target == "guide":
        url = latest.get("upgrade_guide_url") or notes
    elif target == "download":
        art = (latest.get("artifacts") or {}).get(artifact_key()) if channel() == "packaged" else None
        url = (art or {}).get("url") or notes
    else:
        return None
    return url if is_allowed_url(url) else None


# --- What's new (shown once after an upgrade) --------------------------------

def changelog_highlights(version: str) -> list:
    """First ≤5 bullets of the bundled CHANGELOG.md section — works offline."""
    try:
        text = (_PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    except OSError:
        return []
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
    if not m:
        return []
    bullets, cur = [], None
    for line in m.group(1).splitlines():
        if line.startswith("- "):
            cur = line[2:].strip()
            bullets.append(cur)
        elif cur is not None and line.startswith("  ") and line.strip():
            bullets[-1] = f"{bullets[-1]} {line.strip()}"
        elif not line.strip():
            cur = None
    return bullets[:5]


def whats_new() -> Dict[str, Any]:
    """{show, version, highlights} once after the running version changes. A
    fresh install (nothing seen yet) records the version without showing."""
    current = current_version()
    seen = load_state().get("last_seen_version")
    if not current or seen == current:
        return {"show": False, "version": current, "highlights": []}
    _update_state(last_seen_version=current)
    highlights = changelog_highlights(current) if seen else []
    return {"show": bool(seen and highlights), "version": current, "highlights": highlights}


# --- background thread -------------------------------------------------------

_stop = threading.Event()


def start_background_checker() -> Optional[threading.Thread]:
    """First check ~30 s after start, then hourly wake-ups; check() itself
    enforces the 24 h interval and the user's automatic-check switch (so
    turning it back on works without a restart). Not started at all in the HF
    Space, with FRAGMENTA_NO_UPDATE_CHECK=1, or without a readable VERSION."""
    reason = is_disabled()
    if reason:
        logger.info(f"Update check disabled ({reason})")
        return None

    def loop() -> None:
        if _stop.wait(FIRST_CHECK_DELAY):
            return
        while not _stop.is_set():
            check()
            _stop.wait(3600)

    t = threading.Thread(target=loop, name="update-check", daemon=True)
    t.start()
    return t
