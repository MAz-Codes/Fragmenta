"""Bulk annotation injection regression suite.

Covers PATCH /api/projects/<name>/clips/prompt and the underlying
projects.inject_clip_prompts: modes, scoping, separator behaviour on empty
prompts, repeat injection, validation and the in-memory (unsaved) contract.

Pure stdlib + the project venv; no pytest required. Run:
    venv/bin/python tests/test_prompt_injection.py
"""
import os
import sys
import tempfile
import wave
import struct
from pathlib import Path

# --- Isolation: must be set before importing the app / config singleton ------
_TMP = Path(tempfile.mkdtemp(prefix="fragmenta-inject-"))
os.environ["FRAGMENTA_PROJECTS_DIR"] = str(_TMP / "projects")
os.environ["FRAGMENTA_UPLOADS_DIR"] = str(_TMP / "uploads")
os.environ.pop("FRAGMENTA_DOCKER", None)
(_TMP / "projects").mkdir(parents=True, exist_ok=True)
(_TMP / "uploads").mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.backend.app as A  # noqa: E402
from app.backend.data import projects as P  # noqa: E402

client = A.app.test_client()

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


def make_wav(path: Path, seconds=0.05, sr=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", 0) for _ in range(n)))


def fresh_project(name, prompts):
    """Create a project with clips carrying the given {file: prompt}."""
    proj = P.project_path(name)
    proj.mkdir(parents=True, exist_ok=True)
    for fname, prompt in prompts.items():
        make_wav(proj / fname)
        (proj / f"{Path(fname).stem}.txt").write_text(prompt, encoding="utf-8")
    # Drop any cached session so the sidecars are re-read.
    P._SESSIONS.pop(name, None) if hasattr(P, "_SESSIONS") else None
    return P.get_project(name)


def prompts_of(name):
    return {c["file_name"]: c["prompt"] for c in P.get_project(name)["clips"]}


def inject(name, **body):
    return client.patch(f"/api/projects/{name}/clips/prompt", json=body)


def undo(name):
    return client.post(f"/api/projects/{name}/clips/prompt/undo")


def set_prompt(name, fname, prompt):
    return client.patch(f"/api/projects/{name}/clip/{fname}", json={"prompt": prompt})


# =============================================================================
print("\n=== setup ===")
fresh_project("p1", {"a.wav": "kick loop", "b.wav": "snare hit", "c.wav": ""})
base = prompts_of("p1")
check("project has 3 clips", len(base) == 3, f"got {base}")
check("empty prompt round-trips as empty", base.get("c.wav") == "", f"got {base.get('c.wav')!r}")

# =============================================================================
print("\n=== append (default mode) across all clips ===")
r = inject("p1", text="120bpm")
check("200 OK", r.status_code == 200, f"status={r.status_code} body={r.get_json()}")
check("reports 3 updated", r.get_json().get("updated") == 3, f"got {r.get_json()}")
p = prompts_of("p1")
check("appends after existing text with separator",
      p["a.wav"] == "kick loop, 120bpm", f"got {p['a.wav']!r}")
check("second clip likewise",
      p["b.wav"] == "snare hit, 120bpm", f"got {p['b.wav']!r}")
check("empty prompt gets bare text, no leading comma",
      p["c.wav"] == "120bpm", f"got {p['c.wav']!r}")

# =============================================================================
print("\n=== repeat injection stacks (the 'infinite times' contract) ===")
inject("p1", text="analog synth")
p = prompts_of("p1")
check("second pass keeps the first",
      p["a.wav"] == "kick loop, 120bpm, analog synth", f"got {p['a.wav']!r}")
check("previously-empty clip now has both",
      p["c.wav"] == "120bpm, analog synth", f"got {p['c.wav']!r}")

# =============================================================================
print("\n=== prepend ===")
fresh_project("p2", {"a.wav": "kick loop", "c.wav": ""})
inject("p2", text="lo-fi", mode="prepend")
p = prompts_of("p2")
check("prepends before existing text", p["a.wav"] == "lo-fi, kick loop", f"got {p['a.wav']!r}")
check("empty prompt gets bare text, no trailing comma",
      p["c.wav"] == "lo-fi", f"got {p['c.wav']!r}")

# =============================================================================
print("\n=== replace ===")
fresh_project("p3", {"a.wav": "kick loop", "c.wav": ""})
inject("p3", text="field recording", mode="replace")
p = prompts_of("p3")
check("replace discards existing", p["a.wav"] == "field recording", f"got {p['a.wav']!r}")
check("replace fills empty too", p["c.wav"] == "field recording", f"got {p['c.wav']!r}")

# =============================================================================
print("\n=== scoping to a subset ===")
fresh_project("p4", {"a.wav": "one", "b.wav": "two", "c.wav": "three"})
r = inject("p4", text="TAG", files=["a.wav", "c.wav"])
check("reports 2 updated", r.get_json().get("updated") == 2, f"got {r.get_json()}")
p = prompts_of("p4")
check("selected clip a touched", p["a.wav"] == "one, TAG", f"got {p['a.wav']!r}")
check("unselected clip b untouched", p["b.wav"] == "two", f"got {p['b.wav']!r}")
check("selected clip c touched", p["c.wav"] == "three, TAG", f"got {p['c.wav']!r}")

print("\n=== empty file list touches nothing (not 'all') ===")
r = inject("p4", text="NOPE", files=[])
check("reports 0 updated", r.get_json().get("updated") == 0, f"got {r.get_json()}")
p = prompts_of("p4")
check("no clip changed", "NOPE" not in "".join(p.values()), f"got {p}")

print("\n=== stale selection is reported, not fatal ===")
r = inject("p4", text="X", files=["a.wav", "ghost.wav"])
body = r.get_json()
check("200 OK despite unknown name", r.status_code == 200, f"status={r.status_code}")
check("updated counts only real clips", body.get("updated") == 1, f"got {body}")
check("missing lists the ghost", body.get("missing") == ["ghost.wav"], f"got {body}")

# =============================================================================
print("\n=== validation ===")
r = inject("p4")
check("missing text → 400", r.status_code == 400, f"status={r.status_code}")
r = inject("p4", text="   ")
check("whitespace-only text → 400", r.status_code == 400, f"status={r.status_code}")
r = inject("p4", text="x", mode="obliterate")
check("unknown mode → 400", r.status_code == 400, f"status={r.status_code}")
r = inject("p4", text="x", files="a.wav")
check("files as a string → 400", r.status_code == 400, f"status={r.status_code}")
r = client.patch("/api/projects/does-not-exist/clips/prompt", json={"text": "x"})
check("unknown project → 404", r.status_code == 404, f"status={r.status_code}")

print("\n=== text is trimmed ===")
fresh_project("p5", {"a.wav": "one"})
inject("p5", text="  spaced  ")
check("surrounding whitespace stripped",
      prompts_of("p5")["a.wav"] == "one, spaced", f"got {prompts_of('p5')['a.wav']!r}")

# =============================================================================
print("\n=== in-memory only: sidecars untouched until Save/Commit ===")
fresh_project("p6", {"a.wav": "original"})
inject("p6", text="INJECTED")
sidecar = P.project_path("p6") / "a.txt"
check("session shows the injection",
      prompts_of("p6")["a.wav"] == "original, INJECTED", f"got {prompts_of('p6')['a.wav']!r}")
check("on-disk sidecar still holds the committed text",
      sidecar.read_text(encoding="utf-8").strip() == "original",
      f"got {sidecar.read_text(encoding='utf-8')!r}")

print("\n=== project reports itself dirty after injection ===")
proj = P.get_project("p6")
check("dirty flag set", bool(proj.get("dirty")) is True, f"got dirty={proj.get('dirty')!r}")

# =============================================================================
print("\n=== undo: restores the previous prompts ===")
fresh_project("u1", {"a.wav": "kick loop", "b.wav": "snare hit", "c.wav": ""})
inject("u1", text="120bpm")
check("undo depth surfaced on the project",
      P.get_project("u1").get("inject_undo_depth") == 1,
      f"got {P.get_project('u1').get('inject_undo_depth')!r}")
r = undo("u1")
check("200 OK", r.status_code == 200, f"status={r.status_code} body={r.get_json()}")
check("reports 3 restored", r.get_json().get("restored") == 3, f"got {r.get_json()}")
p = prompts_of("u1")
check("clip a back to original", p["a.wav"] == "kick loop", f"got {p['a.wav']!r}")
check("clip b back to original", p["b.wav"] == "snare hit", f"got {p['b.wav']!r}")
check("clip c back to empty", p["c.wav"] == "", f"got {p['c.wav']!r}")
check("depth back to 0", P.get_project("u1").get("inject_undo_depth") == 0,
      f"got {P.get_project('u1').get('inject_undo_depth')!r}")

print("\n=== undo unwinds repeat injections one pass at a time ===")
fresh_project("u2", {"a.wav": "kick"})
inject("u2", text="one")
inject("u2", text="two")
check("depth is 2", P.get_project("u2").get("inject_undo_depth") == 2)
check("both applied", prompts_of("u2")["a.wav"] == "kick, one, two",
      f"got {prompts_of('u2')['a.wav']!r}")
undo("u2")
check("first undo drops only the last pass",
      prompts_of("u2")["a.wav"] == "kick, one", f"got {prompts_of('u2')['a.wav']!r}")
undo("u2")
check("second undo returns to the original",
      prompts_of("u2")["a.wav"] == "kick", f"got {prompts_of('u2')['a.wav']!r}")

print("\n=== undo with nothing to undo → 409 ===")
r = undo("u2")
check("409 conflict", r.status_code == 409, f"status={r.status_code} body={r.get_json()}")

print("\n=== undo never eats a manual edit made since ===")
fresh_project("u3", {"a.wav": "one", "b.wav": "two"})
inject("u3", text="TAG")
set_prompt("u3", "a.wav", "hand-written since")
r = undo("u3")
body = r.get_json()
check("restores only the untouched clip", body.get("restored") == 1, f"got {body}")
check("reports the skip", body.get("skipped") == 1, f"got {body}")
p = prompts_of("u3")
check("manual edit survives", p["a.wav"] == "hand-written since", f"got {p['a.wav']!r}")
check("untouched clip restored", p["b.wav"] == "two", f"got {p['b.wav']!r}")

print("\n=== undo tolerates a clip deleted since ===")
fresh_project("u4", {"a.wav": "one", "b.wav": "two"})
inject("u4", text="TAG")
client.delete("/api/projects/u4/clip/a.wav")
r = undo("u4")
body = r.get_json()
check("200 OK", r.status_code == 200, f"status={r.status_code} body={body}")
check("survivor restored", body.get("restored") == 1, f"got {body}")
check("deleted clip counted as skipped", body.get("skipped") == 1, f"got {body}")
check("survivor is back to original", prompts_of("u4")["b.wav"] == "two",
      f"got {prompts_of('u4')['b.wav']!r}")

print("\n=== undo of a scoped inject only touches that scope ===")
fresh_project("u5", {"a.wav": "one", "b.wav": "two"})
inject("u5", text="TAG", files=["a.wav"])
undo("u5")
p = prompts_of("u5")
check("scoped clip restored", p["a.wav"] == "one", f"got {p['a.wav']!r}")
check("out-of-scope clip never moved", p["b.wav"] == "two", f"got {p['b.wav']!r}")

print("\n=== undo stack is bounded ===")
fresh_project("u6", {"a.wav": "x"})
for i in range(P.MAX_INJECT_UNDO + 5):
    inject("u6", text=f"t{i}")
check(f"depth capped at {P.MAX_INJECT_UNDO}",
      P.get_project("u6").get("inject_undo_depth") == P.MAX_INJECT_UNDO,
      f"got {P.get_project('u6').get('inject_undo_depth')!r}")

print("\n=== a no-op inject records nothing to undo ===")
fresh_project("u7", {"a.wav": "one"})
inject("u7", text="TAG", files=[])
check("empty target list leaves the stack empty",
      P.get_project("u7").get("inject_undo_depth") == 0,
      f"got {P.get_project('u7').get('inject_undo_depth')!r}")
r = undo("u7")
check("so undo reports nothing to undo", r.status_code == 409, f"status={r.status_code}")

# =============================================================================
print(f"\n{'=' * 60}\n  {_PASS} passed, {_FAIL} failed\n{'=' * 60}")
sys.exit(1 if _FAIL else 0)
