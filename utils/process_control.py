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
