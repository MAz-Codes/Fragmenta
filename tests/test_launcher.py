"""First-run checks and the packaged launcher's failure handling.

Pure stdlib + the project venv; no pytest required. Run:
    venv/bin/python tests/test_launcher.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging"))

import install  # noqa: E402
import launcher  # noqa: E402

HOSTS_UP = dict.fromkeys(install.DOWNLOAD_HOSTS, True)
HOSTS_DOWN = dict.fromkeys(install.DOWNLOAD_HOSTS, False)

FAKE_INSTALL = textwrap.dedent("""\
    import json, os, sys
    mode = os.environ["FAKE_MODE"]
    if "--preflight" in sys.argv:
        if mode == "preflight-garbage":
            print("not json")
            sys.exit(0)
        if mode == "preflight-crash":
            raise SystemExit(3)
        full = "--full" in sys.argv
        failures = [{"code": "disk", "message": "No room."}] if mode == "preflight-fails" else []
        print(json.dumps({
            "ok": not failures, "install_needed": False, "full": full, "failures": failures,
            "warnings": ["Old driver."] if mode == "warns" else [], "notes": [],
            "system": {"gpu": "Test GPU" if full else None, "driver": "1.2" if full else None},
            "log_dir": os.environ["FAKE_LOG_DIR"],
        }))
        sys.exit(0 if not failures else 1)
    with open(os.environ["FAKE_CALLS"], "a") as f:
        f.write("launch\\n")
    print("[install] starting Fragmenta…", flush=True)
    if mode == "ready-then-crash":
        print("__FRAGMENTA_UI_READY__", flush=True)
        sys.exit(9)
    if mode == "no-space":
        print("failed to write: No space left on device (os error 28)", flush=True)
        sys.exit(1)
    if mode == "backend":
        print("Backend failed to start in time. é✓", flush=True)
        sys.exit(4)
    if mode == "slow":
        import time
        time.sleep(float(os.environ["FAKE_SLEEP"]))
    print("__FRAGMENTA_UI_READY__", flush=True)
""")


class Bundle:
    """A bundle whose Python is this one and whose install.py is the fake above."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        python = launcher._bundled_python(self.path)
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        (self.path / "payload").mkdir()
        (self.path / "payload" / "install.py").write_text(FAKE_INSTALL, encoding="utf-8")
        self.logs = self.path / "logs"
        self.calls = self.path / "calls"
        return self

    def __exit__(self, *exc):
        self.tmp.cleanup()

    def env(self, mode, **extra):
        return {"FAKE_MODE": mode, "FAKE_LOG_DIR": str(self.logs), "FAKE_CALLS": str(self.calls), **extra}

    def run(self, mode):
        with mock.patch.dict(os.environ, self.env(mode)), mock.patch.object(launcher, "_echo"):
            run = launcher._Run(self.path)
            run.thread.join(30)
        assert not run.thread.is_alive()
        return run

    def launches(self):
        return self.calls.read_text().count("launch") if self.calls.exists() else 0


def kinds(run):
    out = []
    while not run.q.empty():
        out.append(run.q.get_nowait()[0])
    return out


# --- preflight ---------------------------------------------------------------

def report(up_to_date=False, venv_exists=False, free=500.0, hosts=HOSTS_UP, gpu=None, lock=True,
           packaged=False, os_problem=None, full=False):
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        venv = data / "venv"
        if venv_exists:
            venv.mkdir()
        lock_file = data / "lock.txt"
        lock_file.write_text("x")
        reach = mock.Mock(return_value=hosts)
        nvidia = mock.Mock(return_value=gpu)
        with mock.patch.multiple(install, DATA_DIR=data, VENV_PATH=venv, LOG_DIR=data / "logs", PACKAGED=packaged), \
                mock.patch.object(install, "venv_python", return_value=Path(sys.executable)), \
                mock.patch.object(install, "is_required_py", return_value=True), \
                mock.patch.object(install, "deps_up_to_date", return_value=up_to_date), \
                mock.patch.object(install, "platform_lock", return_value=lock_file if lock else None), \
                mock.patch.object(install, "_free_gb", return_value=free), \
                mock.patch.object(install, "_os_problem", return_value=os_problem), \
                mock.patch.object(install, "_reachable_hosts", reach), \
                mock.patch.object(install, "_nvidia_gpu", nvidia):
            result = install.preflight(full=full)
        result["_slow_checks"] = reach.called or nvidia.called
        result["_saved"] = (data / "logs" / "preflight.json").exists()
        return result


def codes(result):
    return [f["code"] for f in result["failures"]]


def test_an_ordinary_launch_skips_the_slow_checks():
    r = report(up_to_date=True)
    assert r["ok"] and not r["install_needed"] and not r["_slow_checks"] and not r["_saved"]


def test_a_full_disk_doesnt_stop_an_installed_app():
    assert report(up_to_date=True, free=0.5)["ok"]


def test_first_install_needs_room():
    assert codes(report(free=install.DISK_FIRST_INSTALL_GB - 1)) == ["disk"]


def test_update_needs_less_room_than_a_first_install():
    assert report(venv_exists=True, free=install.DISK_FIRST_INSTALL_GB - 1)["ok"]
    assert codes(report(venv_exists=True, free=install.DISK_UPDATE_GB - 1)) == ["disk"]


def test_room_for_the_install_but_not_the_models_warns():
    r = report(free=install.DISK_FIRST_INSTALL_GB + 1)
    assert r["ok"] and "models" in r["warnings"][0], r


def test_no_internet_stops_an_install():
    assert codes(report(hosts=HOSTS_DOWN)) == ["network"]


def test_no_internet_doesnt_stop_an_installed_app():
    r = report(up_to_date=True, hosts=HOSTS_DOWN, full=True)
    assert r["ok"] and not r["warnings"] and r["system"]["hosts"] == HOSTS_DOWN


def test_one_host_down_warns_and_names_it():
    r = report(hosts={**HOSTS_UP, "github.com": False})
    assert r["ok"] and "github.com" in r["warnings"][0], r


def test_old_nvidia_driver_warns():
    r = report(gpu=("RTX", "535.104.05"))
    assert r["ok"] and "535.104.05" in r["warnings"][0] and r["system"]["driver"] == "535.104.05", r


def test_current_nvidia_driver_says_nothing():
    assert not report(gpu=("RTX", "570.86.16"))["warnings"]
    assert not report(gpu=("RTX", "595.99"))["warnings"]


def test_unsupported_os_fails():
    assert codes(report(os_problem="too old")) == ["os"]


def test_no_lock_fails_when_packaged_warns_in_source():
    assert codes(report(lock=False, packaged=True)) == ["os"]
    r = report(lock=False)
    assert r["ok"] and "requirements.txt" in r["warnings"][0], r


def test_report_is_saved_when_the_slow_checks_ran():
    assert report()["_saved"] and report(up_to_date=True, full=True)["_saved"]


def test_os_versions():
    with mock.patch.object(install.sys, "platform", "darwin"):
        with mock.patch.object(install.platform, "mac_ver", return_value=("13.6.1", ("", "", ""), "")):
            assert "macOS 14" in install._os_problem()
        with mock.patch.object(install.platform, "mac_ver", return_value=("14.0", ("", "", ""), "")):
            assert install._os_problem() is None
    with mock.patch.object(install.sys, "platform", "linux"):
        with mock.patch.object(install.platform, "libc_ver", return_value=("glibc", "2.27")):
            assert "glibc 2.28" in install._os_problem()
        with mock.patch.object(install.platform, "libc_ver", return_value=("glibc", "2.28")):
            assert install._os_problem() is None
        with mock.patch.object(install.platform, "libc_ver", return_value=("", "")):
            assert install._os_problem() is None


def test_unwritable_data_dir_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        blocker = Path(tmp) / "file"
        blocker.write_text("x")
        with mock.patch.object(install, "DATA_DIR", blocker / "data"):
            assert "can't write" in install._data_dir_problem()
            assert install._free_gb() > 0  # falls back to a folder that exists


def test_a_dead_proxy_means_no_host_answers_in_bounded_time():
    with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9"}):
        start = time.monotonic()
        assert install._reachable_hosts() == HOSTS_DOWN
        assert time.monotonic() - start <= install.HOST_TIMEOUT + 1.5


def test_preflight_prints_only_json_and_changes_no_venv():
    out = subprocess.run([sys.executable, str(ROOT / "install.py"), "--preflight"],
                         capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    parsed = json.loads(out.stdout)
    assert out.returncode == (0 if parsed["ok"] else 1) and "system" in parsed


# --- failure messages: the text is what uv and pip really printed ------------

def test_real_failure_text_is_classified():
    cases = {
        "disk": ["Caused by: failed to write to file `x/_elffile.py`: No space left on device (os error 28)",
                 "There is not enough space on the disk. (os error 112)"],
        "certificate": ["error: Failed to fetch: `https://pypi.org/simple/idna/`\n"
                        "  Caused by: invalid peer certificate: UnknownIssuer"],
        "download": ["  × Failed to download `idna==3.10`\n  ╰─▶ Hash mismatch for `idna==3.10`",
                     "ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE."],
        "network": ["error: Failed to fetch: `https://pypi.org/simple/idna/`\n  Caused by: Connection refused (os error 111)",
                    "  Caused by: dns error\n  Caused by: failed to lookup address information: Name or service not known",
                    "ERROR: Could not find a version that satisfies the requirement idna==3.10 (from versions: none)"],
        "unknown": ["Backend failed to start in time.", "ImportError: no module named x", ""],
    }
    for want, samples in cases.items():
        for sample in samples:
            assert launcher._classify_failure(sample.splitlines())[0] == want, sample


def test_old_output_doesnt_decide_the_reason():
    lines = ["warning: connection reset, retrying"] + ["ok"] * launcher.CLASSIFY_LINES + ["Backend failed."]
    assert launcher._classify_failure(lines)[0] == "unknown"


# --- one try at starting the app --------------------------------------------

def test_a_clean_start_has_no_failure():
    with Bundle() as b:
        run = b.run("ok")
        seen = kinds(run)
    assert run.failure is None and run.returncode == 0 and "ready" in seen and seen[-1] == "exit", seen


def test_preflight_failure_stops_before_launching():
    with Bundle() as b:
        run = b.run("preflight-fails")
        assert run.failure["code"] == "disk" and run.failure["message"] == "No room." and run.returncode == 1
        assert b.launches() == 0 and "failed" in kinds(run)
        assert "No room." in Path(run.failure["log"]).read_text(encoding="utf-8")


def test_a_broken_preflight_doesnt_stop_the_app():
    for mode in ("preflight-garbage", "preflight-crash"):
        with Bundle() as b:
            run = b.run(mode)
            assert run.failure is None and b.launches() == 1, mode


def test_warnings_reach_the_window_and_the_app_still_starts():
    with Bundle() as b:
        run = b.run("warns")
        seen = kinds(run)
    assert run.warnings == ["Old driver."] and "warning" in seen and "ready" in seen and run.failure is None


def test_child_failure_is_classified_and_logged():
    with Bundle() as b:
        run = b.run("no-space")
        assert run.failure["code"] == "disk" and run.returncode == 1
        assert "No space left" in Path(run.failure["log"]).read_text(encoding="utf-8")


def test_failure_at_app_start_keeps_its_exit_code_and_runs_the_slow_checks():
    with Bundle() as b:
        run = b.run("backend")
        assert run.failure["code"] == "unknown" and run.returncode == 4
        assert run.report["full"] and run.report["system"]["gpu"] == "Test GPU"
        assert "é✓" in Path(run.failure["log"]).read_text(encoding="utf-8")


def test_an_app_crash_after_it_opened_isnt_a_startup_failure():
    with Bundle() as b:
        run = b.run("ready-then-crash")
    assert run.failure is None and run.returncode == 9


def test_missing_files_say_reinstall_without_a_log():
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(launcher, "_echo"):
        run = launcher._Run(Path(tmp))
        run.thread.join(10)
    assert run.failure["code"] == "corrupt" and run.returncode == 2
    assert "reinstall" in run.failure["message"] and run.failure["log"] is None


def test_retry_starts_a_new_try_with_a_clean_slate():
    with Bundle() as b:
        first = b.run("no-space")
        second = b.run("ok")
        assert first.failure and second.failure is None and b.launches() == 2


def test_diagnostics_carry_the_facts_and_hide_the_home_folder():
    with Bundle() as b:
        (b.path / "payload" / "manifest.json").write_text(json.dumps(
            {"version": "2.0.0", "commit": "abc123", "target": "linux", "python": {"version": "3.14.10"}}))
        run = b.run("backend")
        run.tail.append(f"opening {Path.home()}/secret-name/file\n")
        text = launcher._diagnostics(run)
    for expected in ("Fragmenta 2.0.0", "abc123", "Test GPU", "driver 1.2", "Exit code: 4", "Backend failed"):
        assert expected in text, expected
    assert str(Path.home()) not in text and "~/secret-name/file" in text


def test_diagnostics_work_when_nothing_ran():
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(launcher, "_echo"):
        run = launcher._Run(Path(tmp))
        run.thread.join(10)
        assert "Problem: corrupt" in launcher._diagnostics(run)


# --- self-test ---------------------------------------------------------------

FAKE_BACKEND = textwrap.dedent("""\
    import http.server, json, os, sys
    mode = os.environ["FAKE_BACKEND"]
    if mode == "crash":
        print("ImportError: no module named x", flush=True)
        sys.exit(5)
    if mode == "hang":
        import time
        time.sleep(60)
    state = {"ready": False}
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            code, body = 200, ""
            if self.path == "/api/health":
                body = json.dumps({"version": "9.9" if mode == "old-version" else "1.0", "components_ready": state["ready"],
                                   "init_error": "no model dir" if mode == "bad-setup" else None})
            elif self.path == "/api/version":
                state["ready"] = mode != "bad-setup"
                code = 503 if mode == "bad-setup" else 200
                body = "{}"
            elif mode == "no-ui":
                code = 404
            else:
                body = "<!doctype html><html></html>"
            self.send_response(code)
            self.end_headers()
            self.wfile.write(body.encode())
    http.server.HTTPServer((os.environ["FLASK_HOST"], int(os.environ["FLASK_PORT"])), H).serve_forever()
""")


def backend_checks(mode, timeout=30.0):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "VERSION").write_text("1.0")
        (root / "app" / "backend").mkdir(parents=True)
        (root / "app" / "backend" / "app.py").write_text(FAKE_BACKEND)
        with mock.patch.multiple(install, PROJECT_ROOT=root, LOG_DIR=root / "logs"), \
                mock.patch.object(install, "log"), mock.patch.dict(os.environ, {"FAKE_BACKEND": mode}):
            return {c["name"]: c for c in install._backend_checks(Path(sys.executable), timeout=timeout)}


def test_a_working_backend_passes_every_check():
    checks = backend_checks("ok")
    assert len(checks) == 5 and all(c["ok"] for c in checks.values()), checks


def test_backend_that_crashes_fails_fast_and_says_why():
    start = time.monotonic()
    check = backend_checks("crash")["backend starts"]
    assert not check["ok"] and "exit 5" in check["detail"] and "no module named x" in check["detail"], check
    assert time.monotonic() - start < 15


def test_backend_that_never_answers_times_out():
    assert "no answer" in backend_checks("hang", timeout=2)["backend starts"]["detail"]


def test_setup_error_fails():
    checks = backend_checks("bad-setup")
    assert not checks["backend sets up"]["ok"] and "no model dir" in checks["no setup errors"]["detail"], checks


def test_wrong_version_fails():
    assert not backend_checks("old-version")["runs this version"]["ok"]


def test_missing_ui_fails():
    assert not backend_checks("no-ui")["serves the UI"]["ok"]


def test_a_proxy_setting_doesnt_catch_local_requests():
    with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9"}):
        assert all(c["ok"] for c in backend_checks("ok").values())


def self_test_result(preflight_ok=True, install_exit=None, imports_ok=True, backend_ok=True):
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        report = {"ok": preflight_ok, "failures": [] if preflight_ok else [{"code": "disk", "message": "No room."}]}
        install_step = mock.Mock(side_effect=SystemExit(install_exit) if install_exit else None)
        backend = mock.Mock(return_value=[{"name": "backend starts", "ok": backend_ok, "detail": ""}])
        with mock.patch.object(install, "LOG_DIR", logs), mock.patch.object(install, "log"), \
                mock.patch.object(install, "preflight", return_value=report), \
                mock.patch.object(install, "require_host_py"), mock.patch.object(install, "ensure_venv"), \
                mock.patch.object(install, "install_dependencies", install_step), \
                mock.patch.object(install, "platform_lock", return_value=None), \
                mock.patch.object(install, "_import_checks", return_value=[{"name": "import torch", "ok": imports_ok, "detail": ""}]), \
                mock.patch.object(install, "_backend_checks", backend):
            code = install.self_test()
        return code, json.loads((logs / "self-test.json").read_text()), install_step.called, backend.called


def test_self_test_passes_and_writes_its_result():
    code, result, _installed, _backend = self_test_result()
    assert code == 0 and result["ok"] and [c["name"] for c in result["checks"]][:2] == ["preflight", "install"]


def test_failed_preflight_stops_before_installing():
    code, result, installed, _backend = self_test_result(preflight_ok=False)
    assert code == 1 and not result["ok"] and not installed and "No room." in result["checks"][0]["detail"]


def test_failed_install_is_reported_not_raised():
    code, result, _installed, backend = self_test_result(install_exit=1)
    assert code == 1 and result["checks"][-1]["name"] == "install" and not backend


def test_any_failed_check_fails_the_self_test():
    assert self_test_result(imports_ok=False)[0] == 1
    assert self_test_result(backend_ok=False)[0] == 1


def launcher_self_test(bundle_dir, script=None):
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["FRAGMENTA_BUNDLE"] = str(bundle_dir)
    env["FAKE_MODE"] = "unused"
    return subprocess.run([sys.executable, str(ROOT / "packaging" / "launcher.py"), "--self-test"], env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=30)


def test_launcher_passes_on_install_py_self_test_and_its_exit_code():
    with Bundle() as b:
        (b.path / "payload" / "install.py").write_text(
            "import os, sys\n"
            "assert sys.argv[1:] == ['--self-test'] and os.environ['FRAGMENTA_PACKAGED'] == '1'\n"
            "print('[install] PASS  é✓', flush=True)\n"
            "sys.exit(7)\n", encoding="utf-8")
        out = launcher_self_test(b.path)
    assert out.returncode == 7 and "PASS  é✓" in out.stdout, out


def test_launcher_self_test_on_a_damaged_install_exits_2():
    with tempfile.TemporaryDirectory() as tmp:
        out = launcher_self_test(tmp)
    assert out.returncode == 2 and "damaged install" in out.stderr, out


# --- the window's hand-drawn shapes -------------------------------------------

def test_window_text_avoids_characters_the_linux_fonts_lack():
    lines = ["creating virtual environment", "ensurepip", "installing dependencies", "Downloading torch (1 GB)",
             "Installed 3 packages", "laion-clap", "verifying key packages", "starting fragmenta",
             "[preflight] checking"]
    statuses = [launcher._status_from_line(line) for line in lines]
    assert all(statuses), statuses
    for text in statuses + [launcher._GENERIC_FAILURE] + [p[2] for p in launcher._FAILURE_PATTERNS]:
        assert text.isascii(), text


def test_quit_from_a_real_window_exits_cleanly():
    """Clicks the real button under a virtual display; destroying the window from
    inside the click's own binding used to crash Tk."""
    xvfb, xdotool = shutil.which("Xvfb"), shutil.which("xdotool")
    if not (xvfb and xdotool):
        print("    (skipped: needs Xvfb and xdotool)")
        return
    display = ":91"
    screen = subprocess.Popen([xvfb, display, "-screen", "0", "1000x600x24"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1)
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "DISPLAY": display, "FRAGMENTA_BUNDLE": tmp, "XDG_RUNTIME_DIR": tmp}
            env.pop("WAYLAND_DISPLAY", None)
            app = subprocess.Popen([sys.executable, str(ROOT / "packaging" / "launcher.py")], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            time.sleep(1.5)
            geometry = subprocess.run([xdotool, "search", "--name", "^Fragmenta$", "getwindowgeometry", "--shell"],
                                      env=env, capture_output=True, text=True).stdout
            found = dict(line.split("=") for line in geometry.splitlines() if "=" in line)
            x, y = int(found["X"]) + 500, int(found["Y"]) + int(found["HEIGHT"]) - 65
            subprocess.run([xdotool, "mousemove", str(x), str(y), "click", "1"], env=env, check=True)
            assert app.wait(timeout=10) == 2, app.stderr.read()  # the "damaged install" exit code
    finally:
        screen.terminate()
        screen.wait()



def test_rounded_shape_has_smooth_corners_and_a_solid_middle():
    fill, bg = "#a0b2d5", "#f8f7f1"
    rows = [row.split() for row in launcher._shape_rows(60, 20, 10, fill, bg)]
    assert len(rows) == 20 and all(len(row) == 60 for row in rows)
    assert rows[0][0] == bg and rows[-1][-1] == bg and rows[10][30] == fill
    edge = {px for row in rows[:4] for px in row[:12]} - {fill, bg}
    assert len(edge) >= 3, edge  # blended pixels, not a jagged edge


def test_shape_drawn_row_by_row_matches_top_and_bottom():
    rows = launcher._shape_rows(40, 16, 8, "#a0b2d5", "#f8f7f1")
    assert rows == rows[::-1]


def test_shadow_darkens_only_outside_the_shape():
    fill, bg = "#f8f7f1", "#f2ede1"
    rows = [row.split() for row in launcher._shape_rows(80, 80, 24, fill, bg, shadow=12)]
    assert rows[40][40] == fill and rows[0][0] == bg
    below = rows[71][40]  # just under the card, where the shadow falls
    assert below not in (fill, bg) and sum(launcher._rgb(below)) < sum(launcher._rgb(bg))


# --- no display --------------------------------------------------------------

def test_no_display_means_no_window_not_a_crash():
    import tkinter
    with mock.patch.object(tkinter, "Tk", side_effect=tkinter.TclError("no display name")), \
            mock.patch.object(launcher, "_dbg"):
        assert launcher._run_window(mock.Mock(), "Starting") is None


def test_launcher_without_a_display_waits_for_a_slow_child():
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    with Bundle() as b:
        wait = launcher.SPLASH_AFTER_SECONDS + 1
        env.update(b.env("slow", FAKE_SLEEP=str(wait)), FRAGMENTA_BUNDLE=str(b.path))
        out = subprocess.run([sys.executable, str(ROOT / "packaging" / "launcher.py")], env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert out.returncode == 0, out.stderr
    assert "no window available" in out.stderr and "__FRAGMENTA_UI_READY__" in out.stdout, out


def test_launcher_without_a_display_reports_a_failure_and_its_exit_code():
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    with Bundle() as b:
        env.update(b.env("backend"), FRAGMENTA_BUNDLE=str(b.path))
        out = subprocess.run([sys.executable, str(ROOT / "packaging" / "launcher.py")], env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert out.returncode == 4 and "Something went wrong" in out.stderr, out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
