"""Settings → Import: bringing data over from an older Fragmenta folder.

Pure stdlib + the project venv; no pytest, GPU or model needed. Run:
    venv/bin/python tests/test_data_import.py
"""
import errno
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.backend import data_import as di  # noqa: E402

MODEL = "models/pretrained/sa3/hub/models--stabilityai--stable-audio-3-small-music-base"
NO_DEPLOY = {"FRAGMENTA_DOCKER": "0", "FRAGMENTA_HF_SPACE": "0", "SPACE_ID": ""}
ALL = ["models", "clap", "adapters", "projects", "output", "vocabulary"]


def write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def make_old(root):
    """An older Fragmenta folder, with the real layouts."""
    blob = write(root / MODEL / "blobs" / "abc", "weights" * 100)
    snap = root / MODEL / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to("../../blobs/abc")
    write(root / MODEL / "refs" / "main", "rev")
    write(root / MODEL / "blobs" / "def.1234.incomplete", "half")
    write(root / "models/pretrained/sa3/xet/chunk", "cache")
    write(root / "models/pretrained/sa3/hub/CACHEDIR.TAG")
    write(root / "models/pretrained/clap/music.pt", "clap")
    write(root / "models/fine_tuned/my_lora/training_metadata.json", "{}")
    write(root / "models/fine_tuned/.gitkeep")
    write(root / "projects/drums/.project.json", "{}")
    write(root / "projects/drums/a.wav", "audio")
    (root / "projects/drums/linked.wav").symlink_to("/nowhere/source.wav")
    write(root / "output/one.wav", "generated")
    write(root / "config/annotator_labels.user.json", "{}")
    assert blob.exists()


def tree(root):
    out = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            out[rel] = "-> " + os.readlink(path)
        elif path.is_file():
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            out[rel] = "dir"
    return out


class Env:
    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.old, self.here = base / "old", base / "here"
        self.old.mkdir()
        self.here.mkdir()
        make_old(self.old)
        paths = {"models_pretrained": self.here / "models/pretrained",
                 "models_fine_tuned": self.here / "models/fine_tuned",
                 "output": self.here / "output"}
        config = SimpleNamespace(get_path=paths.__getitem__, user_data_dir=self.here)
        self.manager = SimpleNamespace(list_jobs=lambda: [])
        self.patches = [mock.patch.object(di, "get_config", return_value=config),
                        mock.patch.object(di, "get_projects_dir", return_value=self.here / "projects"),
                        mock.patch.dict(os.environ, NO_DEPLOY)]
        for p in self.patches:
            p.start()
        di._job.clear()
        di._job.update(state="idle")
        di._cancel.clear()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def start(self, items=None, keep_old=False, manager=None):
        return di.start(str(self.old), ALL if items is None else items, keep_old, manager or self.manager)

    def run(self, **kwargs):
        body, code = self.start(**kwargs)
        assert code == 200, body
        return wait()

    def items(self):
        body, code = di.scan(str(self.old))
        assert code == 200, body
        return {i["id"]: i for i in body["items"]}


def wait():
    deadline = time.monotonic() + 20
    while di.status()["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return di.status()


# --- looking inside ------------------------------------------------------------

def test_scan_finds_everything_with_sizes_and_changes_nothing():
    with Env() as e:
        before = tree(e.old)
        items = e.items()
        assert set(items) == set(ALL)
        assert items["projects"]["count"] == 1 and items["projects"]["bytes"] == len("{}") + len("audio")
        assert items["models"]["bytes"] == len("weights" * 100) + len("rev") + len("x"), items["models"]  # + CACHEDIR.TAG
        assert tree(e.old) == before and tree(e.here) == {}


def test_a_folder_with_nothing_of_ours_says_so():
    with Env() as e:
        empty = e.old.parent / "empty"
        write(empty / "holiday.jpg")
        body, code = di.scan(str(empty))
        assert code == 400 and "No Fragmenta" in body["error"]


def test_refuses_this_fragmentas_own_folder_and_anything_overlapping():
    with Env() as e:
        for path in (e.here, e.here / "models", e.here.parent):
            (e.here / "models").mkdir(exist_ok=True)
            body, code = di.scan(str(path))
            assert code == 400 and "own folder" in body["error"], path


def test_bad_paths_are_refused():
    with Env() as e:
        for path in (None, "", 5, str(e.old / "nope"), str(e.old / "output/one.wav")):
            assert di.scan(path)[1] == 400, path


def test_not_on_the_space_or_docker():
    for env in ({"FRAGMENTA_DOCKER": "1"}, {"FRAGMENTA_HF_SPACE": "1"}, {"SPACE_ID": "x/y"}):
        with Env() as e, mock.patch.dict(os.environ, env):
            assert di.scan(str(e.old))[1] == 403 and e.start()[1] == 403, env


def test_scan_counts_what_is_already_here():
    with Env() as e:
        write(e.here / "projects/drums/a.wav")
        items = e.items()
        assert items["projects"]["count"] == 0 and items["projects"]["already"] == 1


# --- copying -------------------------------------------------------------------

def test_copy_never_changes_the_old_folder():
    with Env() as e:
        before = tree(e.old)
        job = e.run(keep_old=True)
        assert job["state"] == "done" and job["mode"] == "copy" and not job["failed"]
        assert tree(e.old) == before


def test_copied_model_keeps_its_cache_links_working():
    with Env() as e:
        e.run(keep_old=True)
        link = e.here / MODEL / "snapshots/rev/model.safetensors"
        assert link.is_symlink() and os.readlink(link) == "../../blobs/abc"
        assert link.read_text() == "weights" * 100


def test_project_links_to_source_audio_stay_links():
    with Env() as e:
        e.run(keep_old=True)
        assert os.readlink(e.here / "projects/drums/linked.wav") == "/nowhere/source.wav"


def test_everything_lands_where_the_app_looks():
    with Env() as e:
        e.run(keep_old=True)
        for rel in ("models/pretrained/clap/music.pt", "models/fine_tuned/my_lora/training_metadata.json",
                    "projects/drums/a.wav", "output/one.wav", "config/annotator_labels.user.json"):
            assert (e.here / rel).exists(), rel


def test_junk_stays_behind():
    with Env() as e:
        e.run(keep_old=True)
        names = {p.name for p in e.here.rglob("*")}
        assert "def.1234.incomplete" not in names and "xet" not in names and ".gitkeep" not in names


def test_copy_keeps_file_dates():
    with Env() as e:
        os.utime(e.old / "output/one.wav", (1000000000, 1000000000))
        e.run(keep_old=True)
        assert int((e.here / "output/one.wav").stat().st_mtime) == 1000000000


def test_only_what_was_ticked_comes_over():
    with Env() as e:
        e.run(items=["projects"], keep_old=True)
        assert (e.here / "projects/drums").exists()
        assert not (e.here / "output").exists() and not (e.here / "models").exists()


# --- moving --------------------------------------------------------------------

def test_move_takes_it_out_of_the_old_folder():
    with Env() as e:
        job = e.run()
        assert job["state"] == "done" and job["mode"] == "move" and job["done_bytes"] == job["total_bytes"]
        assert (e.here / "projects/drums/a.wav").read_text() == "audio"
        assert not (e.old / "projects/drums").exists()
        assert (e.here / MODEL / "snapshots/rev/model.safetensors").read_text() == "weights" * 100


def test_another_drive_copies_even_when_moving_was_possible():
    with Env() as e:
        real = e.old.stat().st_dev
        fake_stat = mock.Mock(st_dev=real + 1)
        original = Path.stat

        def stat(self, **kw):
            return fake_stat if self == e.old.resolve() else original(self, **kw)

        with mock.patch.object(Path, "stat", stat):
            job = e.run()
        assert job["mode"] == "copy" and (e.old / "output/one.wav").exists()


def test_a_failed_rename_falls_back_to_copy_then_removes_the_original():
    with Env() as e, mock.patch.object(di.os, "rename", side_effect=OSError(errno.EXDEV, "cross-device")):
        job = e.run(items=["projects"])
        assert job["state"] == "done" and (e.here / "projects/drums/a.wav").exists()
        assert not (e.old / "projects/drums").exists()


# --- never overwrite -----------------------------------------------------------

def test_existing_project_with_the_same_name_is_left_alone_on_both_sides():
    for keep_old in (True, False):
        with Env() as e:
            write(e.here / "projects/drums/mine.wav", "mine")
            write(e.old / "projects/other/b.wav", "other")
            e.run(items=["projects"], keep_old=keep_old)
            assert (e.here / "projects/other/b.wav").exists()
            assert (e.here / "projects/drums/mine.wav").read_text() == "mine"
            assert not (e.here / "projects/drums/a.wav").exists()
            assert (e.old / "projects/drums/a.wav").exists(), "skipped, so it must stay in the old folder"


def test_a_partial_model_here_gets_its_gaps_filled():
    with Env() as e:
        write(e.here / MODEL / "refs" / "main", "newer-rev")
        write(e.here / MODEL / "blobs" / "zzz", "only here")
        e.run(items=["models"], keep_old=True)
        assert (e.here / MODEL / "refs" / "main").read_text() == "newer-rev"
        assert (e.here / MODEL / "blobs" / "zzz").read_text() == "only here"
        assert (e.here / MODEL / "snapshots/rev/model.safetensors").read_text() == "weights" * 100


def test_existing_vocabulary_is_kept():
    with Env() as e:
        write(e.here / "config/annotator_labels.user.json", "mine")
        e.run(keep_old=True)
        assert (e.here / "config/annotator_labels.user.json").read_text() == "mine"


def test_importing_twice_does_nothing_the_second_time():
    with Env() as e:
        e.run(keep_old=True)
        after_first = tree(e.here)
        assert all(i["count"] == 0 for i in e.items().values())
        body, code = e.start(keep_old=True)
        assert code == 400 and "Nothing" in body["error"] and tree(e.here) == after_first


def test_something_that_appears_after_the_scan_isnt_overwritten():
    with Env() as e:
        steps = [(e.old / "output/one.wav", e.here / "output/one.wav", 9)]
        write(e.here / "output/one.wav", "arrived first")
        di._job.update(state="running", done_bytes=0, total_bytes=9)
        di._run(steps, move=True)
        assert (e.here / "output/one.wav").read_text() == "arrived first"
        assert (e.old / "output/one.wav").exists()


# --- stopping and failing ------------------------------------------------------

def no_temp_left(root):
    return not [p for p in root.rglob("*") if p.name.startswith(".importing-")]


def test_cancel_leaves_no_half_copied_file():
    with Env() as e:
        write(e.old / "output/big.wav", "x" * (3 * di._CHUNK))
        real_write = di._copy_file

        def cancel_then_copy(src, dst):
            di._cancel.set()
            real_write(src, dst)

        with mock.patch.object(di, "_copy_file", cancel_then_copy):
            job = e.run(items=["output"], keep_old=True)
        assert job["state"] == "cancelled" and no_temp_left(e.here)
        assert not (e.here / "output/big.wav").exists() and (e.old / "output/big.wav").exists()


def test_a_full_disk_stops_with_a_plain_message_and_cleans_up():
    with Env() as e, mock.patch.object(di, "_copy_file", side_effect=OSError(errno.ENOSPC, "no space")):
        job = e.run(items=["output"], keep_old=True)
        assert job["state"] == "failed" and "disk is full" in job["error"] and no_temp_left(e.here)


def test_one_bad_file_doesnt_stop_the_rest():
    with Env() as e:
        write(e.old / "output/two.wav", "b")
        real = di._copy_file

        def fail_on_one(src, dst):
            if src.name == "one.wav":
                raise OSError(errno.EACCES, "denied")
            real(src, dst)

        with mock.patch.object(di, "_copy_file", fail_on_one):
            job = e.run(items=["output"], keep_old=True)
        assert job["state"] == "done" and [f["name"] for f in job["failed"]] == ["one.wav"]
        assert (e.here / "output/two.wav").exists() and not (e.here / "output/one.wav").exists()


def test_not_enough_space_for_a_copy_is_refused_up_front():
    with Env() as e, mock.patch.object(di.shutil, "disk_usage", return_value=SimpleNamespace(free=500_000_000)):
        body, code = e.start(keep_old=True)
        assert code == 409 and "Not enough space" in body["error"] and tree(e.here) == {}


def test_a_move_on_the_same_drive_needs_no_spare_room():
    with Env() as e, mock.patch.object(di.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
        assert e.run()["state"] == "done"


def test_refused_while_a_model_downloads():
    with Env() as e:
        busy = SimpleNamespace(list_jobs=lambda: [{"status": "running"}])
        body, code = e.start(manager=busy)
        assert code == 409 and "downloading" in body["error"]


def test_only_one_import_at_a_time():
    with Env() as e:
        di._job.update(state="running")
        body, code = e.start(keep_old=True)
        assert code == 409 and "already running" in body["error"]


# --- the routes ------------------------------------------------------------------

def client():
    import app.backend.app as backend
    return backend, backend.app.test_client()


def test_a_cross_site_form_post_is_refused():
    backend, c = client()
    with mock.patch.object(backend, "_ensure_components"):
        assert c.post("/api/import/start", data="path=/x", content_type="text/plain").status_code == 415
        assert c.post("/api/import/scan", data="path=/x", content_type="text/plain").status_code == 415


def test_imported_loras_list_when_the_data_folder_is_outside_the_code():
    """As in every installed app; this used to fail the whole list with a 500."""
    backend, c = client()
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        run = data / "models/fine_tuned/my_lora"
        (run / "checkpoints").mkdir(parents=True)
        write(run / "training_metadata.json", json.dumps(
            {"mode": "lora", "base_model": "sa3-small-music-base", "lora_config": {"rank": 16}}))
        ckpt = run / "checkpoints/epoch=1-step=10.safetensors"
        ckpt.write_bytes(b"x")
        config = SimpleNamespace(get_path=lambda name: data / "models/fine_tuned",
                                 project_root=Path(tmp) / "code")
        with mock.patch.object(backend, "get_config", return_value=config), \
                mock.patch.object(backend, "_ensure_components"):
            response = c.get("/api/loras")
    assert response.status_code == 200, response.data
    assert [l["path"] for l in response.get_json()["loras"]] == [str(ckpt)]


def test_edit_upload_works_when_the_data_folder_is_outside_the_code():
    """As in every installed app; this used to fail with a 500."""
    import io
    backend, c = client()
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        config = SimpleNamespace(get_path=lambda name: data / "output", project_root=Path(tmp) / "code")
        with mock.patch.object(backend, "get_config", return_value=config), \
                mock.patch.object(backend, "_ensure_components"):
            response = c.post("/api/audio/upload", content_type="multipart/form-data",
                              data={"file": (io.BytesIO(b"RIFF"), "a.wav")})
            assert response.status_code == 200, response.data
            assert Path(response.get_json()["path"]).read_bytes() == b"RIFF"


def test_api_paths_are_relative_inside_the_code_folder_and_absolute_outside():
    backend, _c = client()
    config = SimpleNamespace(project_root=Path("/code"))
    assert backend._api_path(Path("/code/models/a"), config) == "models/a"
    assert backend._api_path(Path("/data/models/a"), config) == "/data/models/a"


def test_full_fine_tunes_list_when_the_data_folder_is_outside_the_code():
    backend, c = client()
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        ckpt = write(data / "models/fine_tuned/old_run/epoch=1-step=5.ckpt")
        config = SimpleNamespace(get_path=lambda name: data / "models/fine_tuned", project_root=Path(tmp) / "code")
        with mock.patch.object(backend, "get_config", return_value=config), \
                mock.patch.object(backend, "_ensure_components"):
            response = c.get("/api/models")
    assert response.status_code == 200, response.data
    assert response.get_json()["models"][0]["ckpt_path"] == str(ckpt)


def test_training_lists_checkpoints_when_the_data_folder_is_outside_the_code():
    """As in every installed app; this used to raise once a checkpoint was saved."""
    from app.core.training import sa3_trainer as T
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = write(Path(tmp) / "data/run/checkpoints/epoch=1-step=10.ckpt")
        trainer = SimpleNamespace(run_dir=Path(tmp) / "data/run", status={})
        config = SimpleNamespace(project_root=Path(tmp) / "code")
        with mock.patch.object(T, "get_config", return_value=config):
            T.SA3Trainer._scan_checkpoints(trainer)
        assert trainer.status["checkpoints"] == [str(ckpt)]


def test_vocabulary_edits_go_to_the_data_folder():
    """In an installed app the code folder is read-only."""
    backend, _c = client()
    with mock.patch.object(backend, "get_config", return_value=SimpleNamespace(
            user_data_dir=Path("/data"), project_root=Path("/code"))):
        assert backend._annotator_labels_user_path() == Path("/data/config/annotator_labels.user.json")
        assert backend._annotator_labels_default_path() == Path("/code/config/annotator_labels.json")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
