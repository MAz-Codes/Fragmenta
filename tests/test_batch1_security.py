"""Batch 1 security-fix regression suite.

Exercises every fix in redesign/task.md Batch 1 (1.1–1.7) plus regression
checks that normal behaviour still works. Pure stdlib + the project venv; no
pytest required. Run:  venv/bin/python tests/test_batch1_security.py

The suite isolates state into temp dirs via FRAGMENTA_PROJECTS_DIR /
FRAGMENTA_UPLOADS_DIR (set BEFORE importing the app) and toggles
FRAGMENTA_DOCKER per-test to simulate web vs desktop (both are read at call
time by _is_headless()).
"""
import os
import sys
import tempfile
import wave
import struct
import threading
from pathlib import Path

# --- Isolation: must be set before importing the app / config singleton ------
_TMP = Path(tempfile.mkdtemp(prefix="fragmenta-batch1-"))
os.environ["FRAGMENTA_PROJECTS_DIR"] = str(_TMP / "projects")
os.environ["FRAGMENTA_UPLOADS_DIR"] = str(_TMP / "uploads")
os.environ.setdefault("DISPLAY", ":0")          # default = desktop on Linux
os.environ.pop("FRAGMENTA_DOCKER", None)
os.environ.pop("FRAGMENTA_ALLOWED_ORIGINS", None)
os.environ.pop("FRAGMENTA_MAX_DISK_BYTES", None)
os.environ.pop("FRAGMENTA_MAX_CONCURRENT_JOBS", None)
(_TMP / "projects").mkdir(parents=True, exist_ok=True)
(_TMP / "uploads").mkdir(parents=True, exist_ok=True)

# Repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.backend.app as A  # noqa: E402
from app.backend.data import projects as P  # noqa: E402

client = A.app.test_client()

# --- Tiny assert framework ---------------------------------------------------
_PASS = 0
_FAIL = 0


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def set_web():
    os.environ["FRAGMENTA_DOCKER"] = "1"


def set_desktop():
    os.environ.pop("FRAGMENTA_DOCKER", None)
    os.environ["DISPLAY"] = ":0"


def make_wav(path: Path, seconds=0.05, sr=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", 0) for _ in range(n)))


# =============================================================================
print("\n=== 1.1 pick-folder command injection (macOS branch) ===")
import subprocess as _sub  # noqa: E402

PAYLOAD = '/x" & (do shell script "touch /tmp/fragmenta_pwned") & "'


def run_pick(start_dir, platform):
    captured = {}
    real_run = _sub.run
    real_platform = sys.platform

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd

        class R:
            returncode = 1
            stdout = ""
            stderr = ""
        return R()
    try:
        _sub.run = fake_run
        # pick_folder imports subprocess locally → same module object, patched.
        # Force the platform branch we want to inspect.
        sys.platform = platform
        client.post("/api/pick-folder", json={"start_dir": start_dir})
    finally:
        _sub.run = real_run
        sys.platform = real_platform
    return " ".join(captured.get("cmd", []))


cmd = run_pick(PAYLOAD, "darwin")
check("injection payload neutralised (not present verbatim in osascript)",
      PAYLOAD not in cmd, f"cmd={cmd!r}")
check("no 'do shell script' breakout reaches the command",
      "do shell script" not in cmd, f"cmd={cmd!r}")

# A legitimate existing dir survives (escaped). Use a dir whose name has a quote
# to prove the escaping path runs.
weird = _TMP / 'we"ird'
weird.mkdir(exist_ok=True)
cmd2 = run_pick(str(weird), "darwin")
check("existing dir is used (appears, with escaped quote)",
      'we\\"ird' in cmd2, f"cmd={cmd2!r}")

# Nonexistent path falls back to home (not the attacker string).
cmd3 = run_pick("/this/does/not/exist/xyz", "darwin")
check("nonexistent start_dir falls back (attacker path absent)",
      "/this/does/not/exist/xyz" not in cmd3, f"cmd={cmd3!r}")


# =============================================================================
print("\n=== 1.2 path-traversal file deletion (delete_clip + slice_clip) ===")
# Create a project and a real clip via the ingest path (desktop mode).
set_desktop()
proj_name = "trav_proj"
src_dir = _TMP / "src_clips"
make_wav(src_dir / "good.wav")
P.create_project(proj_name)
P.ingest_folder(proj_name, src_dir, "copy")

# Sentinel OUTSIDE the project tree.
sentinel = _TMP / "sentinel_target.wav"
sentinel.write_bytes(b"do-not-delete")

# _safe_clip_path rejects traversal.
proj_path = P.project_path(proj_name)
raised = False
try:
    P._safe_clip_path(proj_path, "../../sentinel_target.wav")
except ValueError:
    raised = True
check("_safe_clip_path raises ValueError on traversal", raised)

raised = False
try:
    P._safe_clip_path(proj_path, "../../sentinel_target.wav".replace("/", os.sep))
except ValueError:
    raised = True
check("_safe_clip_path raises on os-sep traversal too", raised)

# delete_clip with a traversal name → FileNotFoundError (membership check) and
# the sentinel is untouched.
raised = False
try:
    P.delete_clip(proj_name, "../../sentinel_target.wav")
except FileNotFoundError:
    raised = True
except ValueError:
    raised = True
check("delete_clip refuses traversal name", raised)
check("sentinel still on disk after delete attempt", sentinel.exists())

# slice_clip with a traversal name → refused, sentinel untouched.
raised = False
try:
    P.slice_clip(proj_name, "../../sentinel_target.wav", 1.0, 0.0, "hard")
except (FileNotFoundError, ValueError):
    raised = True
check("slice_clip refuses traversal name", raised)
check("sentinel still on disk after slice attempt", sentinel.exists())

# HTTP-level: encoded traversal returns 4xx and leaves the sentinel.
r = client.delete(f"/api/projects/{proj_name}/clip/..%2f..%2fsentinel_target.wav")
check("HTTP delete traversal → 4xx", 400 <= r.status_code < 500, f"status={r.status_code}")
check("sentinel survives HTTP traversal delete", sentinel.exists())

# Regression: a NORMAL delete of a real clip still works.
r = client.delete(f"/api/projects/{proj_name}/clip/good.wav")
check("normal clip delete → 200", r.status_code == 200, f"status={r.status_code}")
check("normal clip removed from disk", not (proj_path / "good.wav").exists())


# =============================================================================
print("\n=== 1.3 CORS allow-list (no wildcard, no credentials) ===")
r = client.get("/api/health", headers={"Origin": "http://evil.example"})
acao = r.headers.get("Access-Control-Allow-Origin")
check("evil origin not reflected", acao in (None, "") and acao != "http://evil.example",
      f"acao={acao!r}")
check("no wildcard ACAO", acao != "*", f"acao={acao!r}")

r = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
check("dev-server origin allowed", r.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000",
      f"acao={r.headers.get('Access-Control-Allow-Origin')!r}")
check("credentials not enabled",
      r.headers.get("Access-Control-Allow-Credentials") not in ("true", "True"),
      f"acac={r.headers.get('Access-Control-Allow-Credentials')!r}")

# Preflight from an allowed origin still returns the exposed fragment headers.
r = client.options("/api/health", headers={
    "Origin": "http://localhost:3000",
    "Access-Control-Request-Method": "GET",
})
check("preflight from allowed origin ok", r.status_code < 400, f"status={r.status_code}")


# =============================================================================
print("\n=== 1.4 free-gpu-memory: cross-process kill gated to desktop ===")
import time as _time  # noqa: E402


def run_free_gpu(docker):
    """Drive /api/free-gpu-memory with subprocess.run faked to present a foreign
    GPU python process; capture whether a `kill` was attempted on it."""
    kills = []
    real_run = _sub.run
    real_sleep = _time.sleep

    def fake_run(cmd, *a, **k):
        joined = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        if "--query-compute-apps=pid,used_memory,process_name" in joined:
            R.stdout = "99999, 4096, python\n"   # foreign pid, 4 GB
        elif "--query-gpu=memory.used,memory.total" in joined:
            R.stdout = "1024, 8192\n"
        if cmd and cmd[0] == "kill":
            kills.append(cmd)
        return R()
    try:
        _sub.run = fake_run
        _time.sleep = lambda *a, **k: None
        if docker:
            os.environ["FRAGMENTA_DOCKER"] = "1"
        else:
            os.environ.pop("FRAGMENTA_DOCKER", None)
            os.environ["DISPLAY"] = ":0"
        client.post("/api/free-gpu-memory", json={})
    finally:
        _sub.run = real_run
        _time.sleep = real_sleep
    return kills


kills_web = run_free_gpu(docker=True)
check("web/headless: no foreign PID killed",
      not any("99999" in " ".join(k) for k in kills_web), f"kills={kills_web}")

kills_desktop = run_free_gpu(docker=False)
check("desktop: foreign PID kill attempted",
      any("99999" in " ".join(k) for k in kills_desktop), f"kills={kills_desktop}")
set_desktop()


# =============================================================================
print("\n=== 1.5 ingest path restriction + symlink disable (web mode) ===")
set_web()
P.create_project("web_proj")

# Arbitrary server path rejected.
r = client.post("/api/projects/web_proj/ingest", json={"folder_path": "/etc", "mode": "copy"})
check("web: ingest /etc → 400", r.status_code == 400, f"status={r.status_code} body={r.get_data(as_text=True)}")

# Symlink mode rejected in web mode.
staging = A.get_config().get_path("uploads") / "staged_folder"
make_wav(staging / "clip1.wav")
r = client.post("/api/projects/web_proj/ingest", json={"folder_path": str(staging), "mode": "symlink"})
check("web: symlink mode → 400", r.status_code == 400, f"status={r.status_code}")

# Ingest from the staging root works...
r = client.post("/api/projects/web_proj/ingest", json={"folder_path": str(staging), "mode": "copy"})
check("web: ingest from staging → 200", r.status_code == 200, f"status={r.status_code} body={r.get_data(as_text=True)}")
body = r.get_json()
check("web: staging clip added", body and body.get("added", 0) >= 1, f"body={body}")
# ... and 1.7 staging cleanup removed the staging dir afterwards.
check("1.7: staging dir removed after ingest", not staging.exists(), f"still exists: {staging}")

# Desktop mode: arbitrary path still works.
set_desktop()
P.create_project("desk_proj")
desk_src = _TMP / "desktop_music"
make_wav(desk_src / "song.wav")
r = client.post("/api/projects/desk_proj/ingest", json={"folder_path": str(desk_src), "mode": "copy"})
check("desktop: arbitrary path ingest → 200", r.status_code == 200, f"status={r.status_code}")
check("desktop: source folder NOT deleted (not under staging)", desk_src.exists())


# =============================================================================
print("\n=== 1.6 torch.load monkey-patch removed (vendored loader) ===")
import torch  # noqa: E402
from app.backend.data import auto_annotator as AA  # noqa: E402

orig_load = torch.load
# Build a fake CLAP-style checkpoint with a module-prefixed position_ids key.
ckpt = _TMP / "fake_clap.pt"
state = {
    "module.text_branch.embeddings.position_ids": torch.tensor([0, 1, 2]),
    "module.audio_branch.weight": torch.tensor([1.0, 2.0]),
}
torch.save(state, str(ckpt))

result = AA._clap_load_state_dict_trusted(str(ckpt), skip_params=True)
check("torch.load global unchanged after trusted load", torch.load is orig_load)
check("module. prefix stripped", "audio_branch.weight" in result, f"keys={list(result)}")
check("position_ids removed", "text_branch.embeddings.position_ids" not in result, f"keys={list(result)}")

# Concurrency intent: while the trusted loader runs, a parallel torch.load on
# another thread must NOT see weights_only forced to False. Since there is no
# global patch at all, this is structurally guaranteed; assert the loader holds
# the shared lock type and that no _trusted_load wrapper leaks onto torch.load.
check("CHECKPOINT_LOAD_LOCK is a Lock", isinstance(AA.CHECKPOINT_LOAD_LOCK, type(threading.Lock())))
check("torch.load is not a wrapper named _trusted_load",
      getattr(torch.load, "__name__", "") != "_trusted_load")


# =============================================================================
print("\n=== 1.7 disk quota + global job cap (opt-in, headless-only) ===")
# Disk quota helper.
set_desktop()
os.environ["FRAGMENTA_MAX_DISK_BYTES"] = "1"
check("desktop: quota never enforced", A._check_disk_quota() is None)
set_web()
# uploads dir has bytes from earlier; limit=1 → over quota.
(A.get_config().get_path("uploads") / "filler.bin").write_bytes(b"x" * 1024)
msg = A._check_disk_quota()
check("web: over-quota returns a message", isinstance(msg, str) and "quota" in msg.lower(), f"msg={msg!r}")
os.environ.pop("FRAGMENTA_MAX_DISK_BYTES", None)
check("web: quota unset → None", A._check_disk_quota() is None)

# Job cap helper.
os.environ["FRAGMENTA_MAX_CONCURRENT_JOBS"] = "1"
set_desktop()
check("desktop: job cap never enforced", A._check_job_quota() is None)
set_web()
check("web: under cap → None (no jobs running)", A._check_job_quota() is None)
# Inject a fake running annotate job.
with A._project_annotate_jobs_lock:
    A._project_annotate_jobs["fake"] = {"state": "running"}
msg = A._check_job_quota()
check("web: at cap → message", isinstance(msg, str) and "busy" in msg.lower(), f"msg={msg!r}")
with A._project_annotate_jobs_lock:
    A._project_annotate_jobs.pop("fake", None)
os.environ.pop("FRAGMENTA_MAX_CONCURRENT_JOBS", None)
set_desktop()

# pre_encoder helper exists and counts.
from app.backend.data.pre_encoder import count_running_pre_encode_jobs  # noqa: E402
check("count_running_pre_encode_jobs() == 0 at rest", count_running_pre_encode_jobs() == 0)


# =============================================================================
print("\n=== Regression: core read routes still 200 ===")
for route in ["/api/health", "/api/projects"]:
    r = client.get(route)
    check(f"GET {route} ok", r.status_code == 200, f"status={r.status_code}")

# Clean up temp tree.
import shutil  # noqa: E402
shutil.rmtree(_TMP, ignore_errors=True)

print(f"\n==== RESULT: {_PASS} passed, {_FAIL} failed ====")
sys.exit(1 if _FAIL else 0)
