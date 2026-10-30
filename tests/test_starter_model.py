"""The first-run starter card's state, and pausing a download.

Pure stdlib + the project venv; no pytest, network or model needed. Run:
    venv/bin/python tests/test_starter_model.py
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import constants as hf_constants  # noqa: E402

from app.backend import starter_model as starter  # noqa: E402
from app.core import model_manager as mm  # noqa: E402

STARTER = starter.STARTER_MODEL
NO_DEPLOY = {"FRAGMENTA_DOCKER": "0", "FRAGMENTA_HF_SPACE": "0", "SPACE_ID": ""}


class Env:
    """A real ModelManager on an empty folder, with state kept in a temp file."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        config = SimpleNamespace(get_path=lambda name: self.root / "pretrained")
        self.saved_env = {k: os.environ.get(k) for k in (
            "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_XET_CACHE", "HF_HUB_DISABLE_SHARED_BLOBS")}
        self.manager = mm.ModelManager(config)
        self.patches = [mock.patch.object(starter, "_state_path", return_value=self.root / "state.json"),
                        mock.patch.dict(os.environ, NO_DEPLOY)]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        for k, v in self.saved_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.tmp.cleanup()

    def install(self, model_id=STARTER):
        snap = self.manager._local_dir_for(model_id) / "snapshots" / "abc"
        (snap / "t5gemma-b-b-ul2").mkdir(parents=True, exist_ok=True)
        (snap / "model.safetensors").write_bytes(b"x" * 10)
        (snap / "t5gemma-b-b-ul2" / "config.json").write_text("{}")

    def status(self):
        return starter.status(self.manager)

    def accept(self):
        starter._save_state(accepted_terms_version=starter.TERMS_VERSION)


def job(status="running", model_id=STARTER, error=None, downloaded=0):
    return {"job_id": "j", "model_id": model_id, "status": status, "error": error,
            "downloaded_bytes": downloaded, "started_at": "2026-10-30T10:00:00"}


# --- when the card shows ------------------------------------------------------

def test_no_model_shows_the_card():
    with Env() as e:
        s = e.status()
    assert s["show"] and s["phase"] == "idle" and not s["terms_accepted"] and not s["returning"]


def test_any_generation_model_hides_it():
    for model_id in (STARTER, "sa3-small-music", "sa3-medium", "sa3-small-sfx-base"):
        with Env() as e:
            with mock.patch.object(e.manager, "is_model_downloaded", side_effect=lambda m: m == model_id):
                s = e.status()
        assert not s["show"] and s["installed"], model_id


def test_the_annotation_tool_isnt_a_model():
    with Env() as e:
        with mock.patch.object(e.manager, "is_model_downloaded", side_effect=lambda m: m == "clap-music"):
            assert e.status()["show"]


def test_closing_is_not_recorded_so_it_comes_back():
    """The card has no "closed" state to save: only a model or a download hides it."""
    with Env() as e:
        first = e.status()
        assert first["show"]
        assert not (e.root / "state.json").exists()
        assert e.status()["show"]


def test_downloading_another_model_hides_it():
    with Env() as e:
        other = mm._DownloadJob(job_id="o", model_id="sa3-small-music", status="running")
        e.manager._jobs["o"] = other
        assert not e.status()["show"]


def test_it_comes_back_if_that_other_download_fails():
    with Env() as e:
        e.manager._jobs["o"] = mm._DownloadJob(job_id="o", model_id="sa3-small-music", status="failed")
        assert e.status()["show"]


def test_not_on_the_space_or_docker():
    for env in ({"FRAGMENTA_DOCKER": "1"}, {"FRAGMENTA_HF_SPACE": "1"}, {"SPACE_ID": "x/y"}):
        with Env() as e, mock.patch.dict(os.environ, env):
            assert not e.status()["show"], env


def test_deleting_every_model_gives_the_short_version():
    with Env() as e:
        e.install()
        e.status()  # sees the model, remembers it had one
        e.manager.delete_model(STARTER)
        s = e.status()
    assert s["show"] and s["returning"]


def test_someone_who_never_used_the_card_gets_the_full_version():
    with Env() as e:
        assert not e.status()["returning"]


# --- the agreement ------------------------------------------------------------

def test_no_download_without_the_agreement():
    with Env() as e, mock.patch.object(e.manager, "start_download") as start:
        body, code = starter.start(e.manager)
        assert code == 403 and body["code"] == "terms" and not start.called
        body, code = starter.start(e.manager, accept_terms_version="1999-01-01")
        assert code == 403 and not start.called and not e.status()["terms_accepted"]


def test_agreement_is_recorded_with_its_version_and_time():
    with Env() as e, mock.patch.object(e.manager, "start_download", return_value=job()) as start:
        _body, code = starter.start(e.manager, accept_terms_version=starter.TERMS_VERSION)
        state = starter._load_state()
    assert code == 200 and state["accepted_terms_version"] == starter.TERMS_VERSION and state["accepted_at"]
    assert start.call_args.kwargs == {"keep_partial": True}


def test_resuming_needs_no_second_agreement():
    with Env() as e, mock.patch.object(e.manager, "start_download", return_value=job()):
        e.accept()
        assert starter.start(e.manager)[1] == 200


def test_new_terms_ask_again():
    with Env() as e:
        e.accept()
        with mock.patch.object(starter, "TERMS_VERSION", "2030-01-01"):
            assert not e.status()["terms_accepted"]
            assert starter.start(e.manager)[1] == 403


# --- disk ---------------------------------------------------------------------

def disk_free(free):
    return mock.patch.object(starter.shutil, "disk_usage", return_value=SimpleNamespace(free=free))


def test_not_enough_space_says_how_much_and_doesnt_start():
    with Env() as e, disk_free(1_000_000_000), mock.patch.object(e.manager, "start_download") as start:
        e.accept()
        body, code = starter.start(e.manager)
    assert code == 409 and body["code"] == "disk" and not start.called
    assert "need 4.5 GB, have 1.0 GB" in body["error"], body


def test_resuming_only_needs_room_for_whats_left():
    size = mm._SA3_CATALOG[STARTER]["size_bytes"]
    with Env() as e, mock.patch.object(e.manager, "start_download", return_value=job()), \
            mock.patch.object(starter, "_downloaded_so_far", return_value=size), disk_free(1_100_000_000):
        e.accept()
        assert starter.start(e.manager)[1] == 200


def test_busy_with_another_download_is_reported():
    with Env() as e, mock.patch.object(e.manager, "start_download", return_value={"error": "Another download"}):
        e.accept()
        body, code = starter.start(e.manager)
        assert code == 409 and body["code"] == "busy"
        assert starter._load_state().get("status") != "downloading"


# --- phases -------------------------------------------------------------------

def test_quitting_mid_download_comes_back_as_interrupted():
    with Env() as e:
        starter._save_state(status="downloading")
        assert e.status()["phase"] == "interrupted"


def test_a_paused_download_stays_paused_after_a_quit():
    with Env() as e:
        starter._save_state(status="paused")
        assert e.status()["phase"] == "paused"


def test_a_failed_download_says_why_in_plain_words():
    with Env() as e, mock.patch.object(e.manager, "list_jobs", return_value=[job("failed", error="ConnectError: boom")]):
        s = e.status()
    assert s["phase"] == "failed" and "internet" in s["error"]


def test_error_wording():
    cases = {
        # real text: hf-hub raises being offline as an OSError, which the job words as a disk problem
        "Couldn't write the download to disk (Got: ConnectError: [Errno 111] Connection refused\nAn error happened "
        "while trying to locate the files on the Hub ... Please check your internet connection and try again.).": "internet",
        "Not enough free disk space to download the model. Free up space": "disk space",
        "Permission denied writing to the install / data folder.": "Permission",
        "503 Server Error: Service Unavailable for url": "isn't responding",
        "something odd": "The download stopped: something odd",
        None: "The download stopped.",
    }
    for raw, expected in cases.items():
        assert expected in starter._plain_error(raw), raw


def test_progress_comes_from_the_running_job():
    with Env() as e, mock.patch.object(e.manager, "list_jobs", return_value=[job(downloaded=123)]):
        s = e.status()
    assert s["phase"] == "downloading" and s["downloaded_bytes"] == 123


def test_half_written_files_dont_count_as_progress():
    with Env() as e:
        blobs = e.manager._local_dir_for(STARTER) / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "done").write_bytes(b"x" * 100)
        (blobs / "abc.1234.incomplete").write_bytes(b"x" * 50)
        assert e.status()["downloaded_bytes"] == 100


# --- pause and cancel, through the real download job ---------------------------

def run_job(e, keep_partial, stop, seen=None):
    """A real job whose fetch writes a finished file and a partial one, then stops."""
    target = e.manager._local_dir_for(STARTER)

    def fake_snapshot(**_kwargs):
        if seen is not None:
            seen["xet_off"] = hf_constants.HF_HUB_DISABLE_XET
        blobs = target / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        (blobs / "finished").write_bytes(b"x" * 100)
        (blobs / "abc.incomplete").write_bytes(b"x" * 50)
        if stop:
            stop(the_job)
            raise mm._DownloadCancelled()
        snap = target / "snapshots" / "abc" / "t5gemma-b-b-ul2"
        snap.mkdir(parents=True, exist_ok=True)
        (snap.parent / "model.safetensors").write_bytes(b"x")
        (snap / "config.json").write_text("{}")
        return str(snap.parent)

    with mock.patch.object(mm, "snapshot_download", fake_snapshot):
        d = e.manager.start_download(STARTER, keep_partial=keep_partial)
        the_job = e.manager._jobs[d["job_id"]]
        the_job._thread.join(20)
    return the_job, target


def test_pause_keeps_what_was_fetched():
    with Env() as e:
        the_job, target = run_job(e, True, lambda j: e.manager.pause_job(j.job_id))
        assert the_job.status == "paused" and the_job.error is None
        assert (target / "blobs" / "finished").exists()


def test_cancel_removes_what_was_fetched():
    with Env() as e:
        the_job, target = run_job(e, True, lambda j: e.manager.cancel_job(j.job_id))
        assert the_job.status == "cancelled" and not target.exists()


def test_resume_clears_half_written_files_first():
    with Env() as e:
        target = e.manager._local_dir_for(STARTER)
        (target / "blobs").mkdir(parents=True)
        leftover = target / "blobs" / "old.1234.incomplete"
        leftover.write_bytes(b"x")
        run_job(e, True, None)
        assert not leftover.exists()


def test_only_starter_downloads_skip_xet():
    """Xet swallows the exception that stops a job, so these can't use it."""
    for keep_partial, expected in ((True, True), (False, hf_constants.HF_HUB_DISABLE_XET)):
        with Env() as e:
            seen = {}
            the_job, _target = run_job(e, keep_partial, None, seen)
            assert the_job.status == "complete" and seen["xet_off"] == expected, keep_partial


def test_files_stay_in_the_models_own_folder():
    """hf-hub 2.x's cache-wide blobs would survive deleting a model's folder."""
    with Env():
        assert hf_constants.HF_HUB_DISABLE_SHARED_BLOBS is True


def test_starter_pause_through_the_module():
    with Env() as e:
        running = mm._DownloadJob(job_id="r", model_id=STARTER, status="running")
        e.manager._jobs["r"] = running
        assert starter.pause(e.manager) == {"paused": ["r"]}
        assert running._cancel_flag.is_set() and running._pausing
        assert starter._load_state()["status"] == "paused"


def test_cancelling_a_paused_download_deletes_its_files():
    with Env() as e:
        target = e.manager._local_dir_for(STARTER)
        target.mkdir(parents=True)
        starter._save_state(status="paused")
        starter.cancel(e.manager)
        assert not target.exists() and e.status()["phase"] == "idle"


def test_cancelling_keeps_the_agreement():
    with Env() as e:
        e.accept()
        starter.cancel(e.manager)
        assert e.status()["terms_accepted"]


def test_cancel_cant_delete_a_finished_model():
    with Env() as e:
        e.install()
        starter.cancel(e.manager)
        assert e.manager.is_model_downloaded(STARTER)


# --- license texts --------------------------------------------------------------

def test_every_license_text_is_there():
    with Env():
        for license_id, info in starter.LICENSE_TEXTS.items():
            text = starter.license_text(license_id)
            assert text and len(text["text"]) > 500 and text["url"].startswith("https://"), license_id
            assert (ROOT / "licenses" / info["file"]).exists()


def test_only_listed_texts_can_be_read():
    for bad in ("../LICENSE", "../../etc/passwd", "community/../../x", ""):
        assert starter.license_text(bad) is None, bad


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
