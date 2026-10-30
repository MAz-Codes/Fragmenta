"""Update notifier (app/backend/update_check.py) against a local fake GitHub.

Covers the U1.4 list in redesign/UPDATE_NOTIFIER.md: newer/equal/older,
prereleases, API fallback, timeouts, URL allowlist, skip/snooze, disabled
reasons, channels, corrupt state, What's new, the major-upgrade rule and
the 415 on non-JSON POSTs. Never touches the real state file or network.

Run:  ./venv/bin/python tests/test_update_check.py
"""
import http.server
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- fake GitHub -------------------------------------------------------------
ROUTES = {}   # path -> (status, body, delay)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        status, body, delay = ROUTES.get(self.path, (404, "not found", 0))
        if delay:
            time.sleep(delay)
        data = body if isinstance(body, str) else json.dumps(body)
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data.encode())
        except (BrokenPipeError, ConnectionResetError):
            pass   # the client already timed out — that's the point of the test

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_port}"
os.environ["FRAGMENTA_UPDATE_MANIFEST_URL"] = f"{BASE}/latest/latest.json"
os.environ["FRAGMENTA_UPDATE_API_URL"] = f"{BASE}/api"
os.environ["FRAGMENTA_UPDATE_DOWNLOAD_BASE"] = f"{BASE}/download"
for k in ("FRAGMENTA_HF_SPACE", "SPACE_ID", "FRAGMENTA_NO_UPDATE_CHECK", "FRAGMENTA_PACKAGED", "FRAGMENTA_DOCKER"):
    os.environ.pop(k, None)

from app.backend import update_check as uc  # noqa: E402

TMP = Path(tempfile.mkdtemp())
uc._state_path = lambda: TMP / "update_state.json"
uc.current_version = lambda: "1.2.0"
uc._running_python = lambda: "3.11"
REAL_CHANNEL = uc.channel
uc.channel = lambda: CHANNEL[0]
CHANNEL = ["source-git"]
GH = "https://github.com/MAz-Codes/Fragmenta"

fails = 0


def chk(name, cond, detail=""):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{' — ' + str(detail) if detail else ''}")
    fails += 0 if cond else 1


def reset(**state):
    (TMP / "update_state.json").unlink(missing_ok=True)
    if state:
        uc.save_state({**uc._DEFAULT_STATE, **state})
    ROUTES.clear()


def manifest(version, **extra):
    return {"schema": 1, "version": version, "notes_url": f"{GH}/releases/tag/v{version}",
            "highlights": ["One", "Two"], **extra}


def run(m=None, **state):
    reset(**state)
    if m is not None:
        ROUTES["/latest/latest.json"] = (200, m, 0)
    uc.check(force=True)
    return uc.status()


print("=== newer / equal / older ===")
s = run(manifest("1.2.1"))
chk("newer → available", s["available"] and s["latest"] == "1.2.1")
chk("same Python, minor bump → in place, git pull text", s["upgrade_kind"] == "in_place" and "git pull" in s["instructions"])
chk("equal → not available", not run(manifest("1.2.0"))["available"])
chk("older → not available", not run(manifest("1.1.0"))["available"])

print("=== major-upgrade rule (frozen 1.2 client meets 1.3) ===")
s = run(manifest("1.3.0", requires={"python": "3.14"}))
chk("requires.python 3.14 on a 3.11 source install → major", s["upgrade_kind"] == "major")
chk("major text names Python 3.14 and never says 'git pull' as an instruction",
    "Python 3.14" in s["instructions"] and "Update with" not in s["instructions"], s["instructions"])
chk("same requires.python → in place", run(manifest("1.2.1", requires={"python": "3.11"}))["upgrade_kind"] == "in_place")
chk("reinstall_required → major", run(manifest("1.2.1", reinstall_required=True))["upgrade_kind"] == "major")
reset()
ROUTES["/api/releases/latest"] = (200, {"tag_name": "v2.0.0", "html_url": f"{GH}/releases/tag/v2.0.0", "body": "- Big"}, 0)
uc.check(force=True)
s = uc.status()
chk("major version bump via API fallback (no `requires`) → major", s["upgrade_kind"] == "major" and s["latest"] == "2.0.0")
CHANNEL[0] = "docker"
s = run(manifest("1.3.0", requires={"python": "3.14"}, reinstall_required=True))
chk("same manifest on Docker → in place, rebuild line", s["upgrade_kind"] == "in_place" and "Rebuild" in s["instructions"])
CHANNEL[0] = "packaged"
s = run(manifest("1.3.0", requires={"python": "3.14"}))
chk("packaged → in place, install-over text", s["upgrade_kind"] == "in_place" and "install it over" in s["instructions"])
CHANNEL[0] = "source-zip"
chk("zip source, in place → replace-folder text", "replace this folder" in run(manifest("1.2.1"))["instructions"])
CHANNEL[0] = "source-git"

print("=== API fallback, errors, timeouts ===")
reset()
ROUTES["/latest/latest.json"] = (200, "this is not json", 0)
ROUTES["/api/releases/latest"] = (200, {"tag_name": "v1.2.5", "html_url": f"{GH}/releases/tag/v1.2.5", "body": "- A\n- B"}, 0)
uc.check(force=True)
s = uc.status()
chk("malformed latest.json → API fallback", s["latest"] == "1.2.5" and s["highlights"] == ["A", "B"])
reset()
ROUTES["/api/releases/latest"] = (200, {"tag_name": "v1.2.6", "html_url": f"{GH}/releases/tag/v1.2.6", "body": ""}, 0)
uc.check(force=True)
chk("404 latest.json → API fallback", uc.status()["latest"] == "1.2.6")
uc.TIMEOUT = (0.5, 0.5)
ROUTES["/latest/latest.json"] = (200, manifest("9.9.9"), 2)
ROUTES["/api/releases/latest"] = (200, {}, 2)
t0 = time.time(); st = uc.check(force=True)
chk("timeout → no exception, error recorded, previous result kept",
    st["last_error"] and uc.status()["latest"] == "1.2.6" and time.time() - t0 < 4)
uc.TIMEOUT = (3.05, 6)
reset(); ROUTES["/latest/latest.json"] = (200, {"schema": 2, "version": "9.0.0"}, 0)
uc.check(force=True)
chk("wrong schema → rejected, nothing shown", not uc.status()["available"])

print("=== prereleases ===")
rel = lambda tag, pre=False, assets=(): {"tag_name": tag, "prerelease": pre, "draft": False,
                                          "html_url": f"{GH}/releases/tag/{tag}", "body": "- rc note",
                                          "assets": [{"name": a} for a in assets]}
reset(include_prereleases=True)
ROUTES["/api/releases?per_page=20"] = (200, [rel("deps-flash-attn-2.8.3-torch2.11-cu128", True),
                                             rel("v1.2.0"), rel("v1.3.0-rc.1", True, ["latest.json"]),
                                             rel("flash-attn-win-v2.7.4.post1"), rel("v1.2.1")], 0)
ROUTES["/download/v1.3.0-rc.1/latest.json"] = (200, manifest("1.3.0rc1", requires={"python": "3.14"}), 0)
uc.check(force=True)
s = uc.status()
chk("toggle on: highest v* tag regardless of list order (rc beats 1.2.1)", s["latest"] == "1.3.0rc1" and s["available"])
chk("rc with latest.json → its manifest (major upgrade)", s["upgrade_kind"] == "major")
ROUTES["/api/releases?per_page=20"] = (200, [rel("v1.3.0-rc.2", True), rel("deps-x", True)], 0)
uc.check(force=True)
chk("rc without latest.json → minimal manifest; deps-… never chosen", uc.status()["latest"] == "1.3.0rc2")
reset(); ROUTES["/latest/latest.json"] = (200, manifest("1.3.0rc1"), 0)
uc.check(force=True)
chk("toggle off: a prerelease manifest is not offered", not uc.status()["available"])

print("=== URL allowlist ===")
for url, ok in [(f"{GH}/releases/tag/v1.3.0", True), (f"{GH}/blob/main/docs/UPGRADING.md", True),
                ("https://github.com/maz-codes/fragmenta/releases", True),
                (f"{GH}/../../evil/repo", False), (f"{GH}/%2e%2e/%2e%2e/evil", False),
                ("https://github.com.evil.io/MAz-Codes/Fragmenta/x", False), ("http://github.com/MAz-Codes/Fragmenta/x", False),
                ("https://user@github.com/MAz-Codes/Fragmenta/x", False), ("https://github.com:444/MAz-Codes/Fragmenta/x", False),
                ("https://github.com/MAz-Codes/FragmentaEvil/x", False), ("https://github.com/other/repo", False),
                ("javascript:alert(1)", False), (f"{GH}\\..\\evil", False)]:
    chk(f"{'allow ' if ok else 'reject'} {url}", uc.is_allowed_url(url) == ok)
s = run(manifest("1.2.1", upgrade_guide_url="https://evil.example/guide",
                 artifacts={"linux-x86_64-tar": {"url": "https://evil.example/x.tgz"}}))
latest = uc.load_state()["latest"]
chk("off-list upgrade_guide_url and artifact dropped", "upgrade_guide_url" not in latest and not latest.get("artifacts"))
chk("guide falls back to notes_url", uc.resolve_url("guide") == f"{GH}/releases/tag/v1.2.1")
chk("unknown open target → nothing", uc.resolve_url("anything") is None)

print("=== skip / snooze / disabled ===")
run(manifest("1.2.1"))
s = uc.dismiss("skip")
chk("skip hides this version", s["skipped"])
run(manifest("1.2.2"), skipped_version="1.2.1")
chk("a newer version shows again after a skip", not uc.status()["skipped"])
s = uc.dismiss("snooze")
chk("snooze hides it", s["snoozed"])
uc.save_state({**uc.load_state(), "snoozed_until": time.time() - 1})
chk("snooze expires", not uc.status()["snoozed"])
os.environ["SPACE_ID"] = "x"; chk("HF Space → disabled", uc.status()["disabled_reason"] == "hf_space"); del os.environ["SPACE_ID"]
os.environ["FRAGMENTA_NO_UPDATE_CHECK"] = "1"; chk("env opt-out → disabled", uc.status()["disabled_reason"] == "env"); del os.environ["FRAGMENTA_NO_UPDATE_CHECK"]
run(manifest("1.2.1"))
uc.update_settings(enabled=False)
uc.save_state({**uc.load_state(), "last_checked": 0})
ROUTES["/latest/latest.json"] = (200, manifest("1.2.9"), 0)
uc.check()
chk("automatic checks off → the daily check fetches nothing", uc.status()["latest"] == "1.2.1")
uc.check(force=True)
chk("…but Check now still works", uc.status()["latest"] == "1.2.9" and uc.status()["disabled_reason"] is None)
uc.update_settings(enabled=True)
chk("switch back on → automatic again", uc.status()["enabled"] is True)

print("=== channels, corrupt state, What's new ===")
os.environ["FRAGMENTA_PACKAGED"] = "1"; chk("FRAGMENTA_PACKAGED=1 → packaged", REAL_CHANNEL() == "packaged"); del os.environ["FRAGMENTA_PACKAGED"]
os.environ["FRAGMENTA_DOCKER"] = "1"; chk("FRAGMENTA_DOCKER=1 → docker", REAL_CHANNEL() == "docker"); del os.environ["FRAGMENTA_DOCKER"]
chk("this checkout → source-git", REAL_CHANNEL() == "source-git")
real_root, uc._PROJECT_ROOT = uc._PROJECT_ROOT, TMP
chk("no .git → source-zip", REAL_CHANNEL() == "source-zip")
(TMP / "update_state.json").write_text("{corrupt")
chk("corrupt state file → defaults", uc.load_state()["enabled"] is True)
reset()
(TMP / "CHANGELOG.md").write_text("# Changelog\n\n## [1.2.0] - 2026-11-20\n\n### Added\n- First thing\n  wrapped line\n- Second thing\n\n## [1.1.1] - x\n- old\n")
uc._PROJECT_ROOT = TMP
w = uc.whats_new()
chk("fresh install: records the version, shows nothing", not w["show"] and uc.load_state()["last_seen_version"] == "1.2.0")
uc.save_state({**uc.load_state(), "last_seen_version": "1.1.1"})
w = uc.whats_new()
chk("after an upgrade: shows once with bundled highlights", w["show"] and w["highlights"] == ["First thing wrapped line", "Second thing"], w)
chk("second call: not again", not uc.whats_new()["show"])

print("=== HTTP routes ===")
from app.backend.app import app  # noqa: E402
c = app.test_client()
r = c.post("/api/update/settings", data="enabled=false", content_type="text/plain")
chk("POST with a text/plain body → 415", r.status_code == 415, r.status_code)
r = c.post("/api/update/dismiss", json={"action": "nope"})
chk("bad dismiss action → 400", r.status_code == 400)
chk("GET /api/update/status → 200 JSON", c.get("/api/update/status").get_json() is not None)

srv.shutdown()
print(f"\n{'All update-check tests passed.' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
