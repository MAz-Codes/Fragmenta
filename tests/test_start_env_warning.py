"""`python start.py` skips the installer, so it warns about a stale environment.

Pure stdlib + the project venv; no pytest required. Run:
    venv/bin/python tests/test_start_env_warning.py
"""
import contextlib
import io
import os
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import start  # noqa: E402

RIGHT_PYTHON = sys.version_info[:2]


def warning(required=RIGHT_PYTHON, deps_ok=True, env=None):
    err = io.StringIO()
    with mock.patch.object(install, "REQUIRED_PY", required), \
            mock.patch.object(install, "REQUIRED_PY_STR", "%d.%d" % required), \
            mock.patch.object(install, "platform_lock", return_value=None), \
            mock.patch.object(install, "deps_up_to_date", return_value=deps_ok), \
            mock.patch.dict(os.environ, {"FRAGMENTA_PACKAGED": env or "0"}), \
            contextlib.redirect_stderr(err):
        start._warn_if_environment_stale()
    return err.getvalue()


def test_a_current_environment_says_nothing():
    assert warning() == ""


def test_wrong_python_names_both_versions_and_the_launcher():
    text = warning(required=(9, 9))
    assert "Python 9.9" in text and "Python %d.%d" % RIGHT_PYTHON in text and "fragmenta." in text, text


def test_stale_packages_warn():
    assert "packages" in warning(deps_ok=False)


def test_packaged_builds_stay_quiet():
    assert warning(required=(9, 9), env="1") == ""


def test_a_broken_check_does_not_stop_startup():
    err = io.StringIO()
    with mock.patch.object(install, "platform_lock", side_effect=RuntimeError("boom")), \
            mock.patch.dict(os.environ, {"FRAGMENTA_PACKAGED": "0"}), contextlib.redirect_stderr(err):
        start._warn_if_environment_stale()
    assert err.getvalue() == ""


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
