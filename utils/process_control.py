"""Cross-platform child-process stop + liveness helpers.

Shared by the trainer (sa3_trainer) and the dataset pre-encoder
(pre_encoder), both of which spawn long-running python children that must
be stoppable from the UI on every OS.
"""
import os
import signal
import subprocess
from typing import Optional


def graceful_stop(
    proc: subprocess.Popen,
    *,
    wait_timeout: float = 10.0,
    kill_timeout: float = 5.0,
) -> None:
    """Stop a child process, preferring a graceful interrupt.

    POSIX: SIGINT first — the child sees KeyboardInterrupt and can
    checkpoint/clean up — then terminate(), then kill().

    Windows: Popen.send_signal(SIGINT) raises ValueError (only CTRL_C/
    CTRL_BREAK events are deliverable, and those need the child spawned
    with CREATE_NEW_PROCESS_GROUP plus a handler on the child side), so
    go straight to terminate() (TerminateProcess). There is no graceful
    path without cooperation from the child script.

    Raises only if even the initial signal/terminate call fails (e.g. the
    process vanished and the OS API errors) — escalation timeouts are
    handled internally.
    """
    if os.name == "nt":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def pid_alive(pid: int, create_time: Optional[float] = None) -> bool:
    """True when `pid` refers to a live process.

    When `create_time` (psutil epoch seconds) is given, the live process
    must also match it — this guards against PID reuse: a recycled PID
    belonging to some unrelated process must not be reported (or killed)
    as ours.
    """
    if not pid or pid <= 0:
        return False
    try:
        import psutil
        p = psutil.Process(pid)
        if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
            return False
        if create_time is not None and abs(p.create_time() - create_time) > 1.0:
            return False
        return True
    except Exception:
        return False
