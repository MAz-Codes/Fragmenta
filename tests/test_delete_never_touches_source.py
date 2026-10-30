"""Proof that no in-app delete ever reaches the user's original audio.

Every destructive path in the dataset workbench is exercised against a real
symlink-ingested project and a real copy-ingested project, asserting after
each one that the *source* files are still on disk and still readable:

    delete clip · discard · delete project · slice · path traversal

The property under test is that project entries are removed with
Path.unlink(), which acts on the link itself and never follows it — and that
_safe_clip_path deliberately resolves only the parent so the leaf stays
un-followed.

Pure stdlib + the project venv; no pytest required. Run:
    venv/bin/python tests/test_delete_never_touches_source.py
"""
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="fragmenta-delsafe-"))
os.environ["FRAGMENTA_PROJECTS_DIR"] = str(_TMP / "projects")
os.environ["FRAGMENTA_UPLOADS_DIR"] = str(_TMP / "uploads")
# Symlink ingest is force-downgraded to copy under FRAGMENTA_DOCKER; this
# suite is specifically about the symlink path, so make sure we're "desktop".
os.environ.pop("FRAGMENTA_DOCKER", None)
# A Linux shell without a display (SSH, VS Code remote) also counts as
# headless, which disables symlink ingest.
os.environ.setdefault("DISPLAY", ":0")
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


def make_wav(path: Path, seconds=0.4, sr=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", (i % 300) * 80) for i in range(n)))


def make_source(tag, names=("one.wav", "two.wav", "three.wav")):
    """A pristine 'user's music folder' outside the projects tree."""
    src = _TMP / "sources" / tag
    src.mkdir(parents=True, exist_ok=True)
    for n in names:
        make_wav(src / n)
    return src


def sources_intact(src, names=("one.wav", "two.wav", "three.wav")):
    """Every source file still present, non-empty and readable as audio."""
    for n in names:
        p = src / n
        if not p.is_file() or p.is_symlink() or p.stat().st_size == 0:
            return False
        try:
            with wave.open(str(p), "rb") as w:
                if w.getnframes() == 0:
                    return False
        except Exception:
            return False
    return True


def ingest(project, src, mode):
    P.project_path(project).mkdir(parents=True, exist_ok=True)
    return client.post(f"/api/projects/{project}/ingest",
                       json={"folder_path": str(src), "mode": mode})


# =============================================================================
print("\n=== symlink ingest actually produces symlinks (else nothing below is a test) ===")
src_a = make_source("a")
r = ingest("sym_proj", src_a, "symlink")
check("ingest 200", r.status_code == 200, f"status={r.status_code} body={r.get_json()}")
proj_a = P.project_path("sym_proj")
links = [p for p in proj_a.glob("*.wav")]
check("3 clips staged", len(links) == 3, f"got {[p.name for p in links]}")
check("staged entries are symlinks", all(p.is_symlink() for p in links),
      f"got {[(p.name, p.is_symlink()) for p in links]}")
check("symlinks point back at the source",
      all(p.resolve().parent == src_a.resolve() for p in links),
      f"got {[str(p.resolve()) for p in links]}")

# =============================================================================
print("\n=== delete one symlinked clip ===")
r = client.delete("/api/projects/sym_proj/clip/one.wav")
check("delete 200", r.status_code == 200, f"status={r.status_code} body={r.get_json()}")
check("project entry is gone",
      not (proj_a / "one.wav").exists() and not (proj_a / "one.wav").is_symlink())
check("ORIGINAL still on disk", (src_a / "one.wav").is_file(),
      f"missing {src_a / 'one.wav'}")
check("all originals intact and readable", sources_intact(src_a))

# =============================================================================
print("\n=== discard (drops every uncommitted clip at once) ===")
r = client.post("/api/projects/sym_proj/discard")
check("discard 200", r.status_code == 200, f"status={r.status_code}")
remaining = list(proj_a.glob("*.wav"))
check("project emptied of clips", remaining == [], f"got {[p.name for p in remaining]}")
check("ORIGINALS all survive discard", sources_intact(src_a))

# =============================================================================
print("\n=== delete the whole project ===")
src_b = make_source("b")
ingest("doomed_proj", src_b, "symlink")
proj_b = P.project_path("doomed_proj")
check("staged as symlinks", all(p.is_symlink() for p in proj_b.glob("*.wav")))
r = client.delete("/api/projects/doomed_proj")
check("delete project 200", r.status_code == 200, f"status={r.status_code}")
check("project folder gone", not proj_b.exists())
check("ORIGINALS all survive rmtree", sources_intact(src_b))

# =============================================================================
print("\n=== slice replaces the parent entry, not the source ===")
src_c = make_source("c")
ingest("slice_proj", src_c, "symlink")
proj_c = P.project_path("slice_proj")
r = client.post("/api/projects/slice_proj/clip/one.wav/slice", json={"target_duration": 0.15, "overlap_sec": 0, "strategy": "hard"})
check("slice 200", r.status_code == 200, f"status={r.status_code} body={str(r.get_json())[:200]}")
check("parent entry removed from project",
      not (proj_c / "one.wav").exists() and not (proj_c / "one.wav").is_symlink())
children = [p for p in proj_c.glob("one_*.wav")]
check("children written into the project", len(children) > 1, f"got {[p.name for p in children]}")
check("children are real files, not links into the source",
      all(p.is_file() and not p.is_symlink() for p in children))
check("ORIGINAL survives slicing", (src_c / "one.wav").is_file())
check("all originals intact after slice", sources_intact(src_c))

# =============================================================================
print("\n=== copy ingest: deleting the copy leaves the source alone ===")
src_d = make_source("d")
ingest("copy_proj", src_d, "copy")
proj_d = P.project_path("copy_proj")
copies = list(proj_d.glob("*.wav"))
check("staged as real copies, not links",
      len(copies) == 3 and all(p.is_file() and not p.is_symlink() for p in copies))
client.delete("/api/projects/copy_proj/clip/two.wav")
check("copy removed from project", not (proj_d / "two.wav").exists())
check("ORIGINAL still on disk", (src_d / "two.wav").is_file())
check("all originals intact", sources_intact(src_d))

# =============================================================================
print("\n=== path traversal cannot aim a delete outside the project ===")
outside = _TMP / "sources" / "DO_NOT_DELETE.wav"
make_wav(outside)
src_e = make_source("e")
ingest("trav_proj", src_e, "symlink")
for attack in ("../DO_NOT_DELETE.wav",
               "../../sources/DO_NOT_DELETE.wav",
               "..%2f..%2fsources%2fDO_NOT_DELETE.wav"):
    r = client.delete(f"/api/projects/trav_proj/clip/{attack}")
    check(f"rejected: {attack}  (status {r.status_code})", r.status_code >= 400)
check("the file it was aiming at is untouched", outside.is_file(), f"missing {outside}")
check("originals still intact", sources_intact(src_e))

# =============================================================================
print("\n=== a dangling symlink (source moved away) is still cleaned up ===")
src_f = make_source("f", names=("gone.wav",))
ingest("dangle_proj", src_f, "symlink")
proj_f = P.project_path("dangle_proj")
(src_f / "gone.wav").unlink()          # user moved/deleted the source themselves
check("project entry is now a dangling link",
      (proj_f / "gone.wav").is_symlink() and not (proj_f / "gone.wav").exists())
r = client.delete("/api/projects/dangle_proj/clip/gone.wav")
check("delete 200", r.status_code == 200, f"status={r.status_code} body={r.get_json()}")
check("dangling link removed",
      not (proj_f / "gone.wav").is_symlink() and not (proj_f / "gone.wav").exists())

# =============================================================================
print(f"\n{'=' * 62}\n  {_PASS} passed, {_FAIL} failed\n{'=' * 62}")
sys.exit(1 if _FAIL else 0)
