"""The GPU memory readout takes device-wide numbers from nvidia-smi when it can.

Pure stdlib + the project venv; no pytest required. Run:
    venv/bin/python tests/test_gpu_memory_status.py
"""
import subprocess
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import app.backend.app as A  # noqa: E402


def smi_output(text):
    return mock.patch.object(subprocess, "run", return_value=mock.Mock(stdout=text))


def test_parses_used_and_total():
    with smi_output("4603, 16303\n"):
        used, total = A._nvidia_smi_memory_gb()
    assert abs(used - 4603 / 1024) < 1e-9 and abs(total - 16303 / 1024) < 1e-9


def test_unusable_output_means_no_reading():
    for text in ("", "[N/A], [N/A]\n", "1, 2, 3\n", "0, 0\n"):
        with smi_output(text):
            assert A._nvidia_smi_memory_gb() is None, text


def test_failures_mean_no_reading():
    for err in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 2)):
        with mock.patch.object(subprocess, "run", side_effect=err):
            assert A._nvidia_smi_memory_gb() is None, err


def _status():
    A._gpu_memory_cache_time = 0
    return A.app.test_client().get("/api/gpu-memory-status").get_json()["memory_info"]["cuda"]


def test_route_reports_what_nvidia_smi_says():
    if not torch.cuda.is_available():
        print("    (no CUDA: skipped)")
        return
    with mock.patch.object(A, "_nvidia_smi_memory_gb", return_value=(4.5, 15.9)):
        cuda = _status()
    assert cuda["free"] == 15.9 - 4.5 and cuda["total"] == 15.9 and cuda["memory_source"] == "nvidia-smi", cuda


def test_route_falls_back_to_the_driver():
    if not torch.cuda.is_available():
        print("    (no CUDA: skipped)")
        return
    with mock.patch.object(A, "_nvidia_smi_memory_gb", return_value=None):
        cuda = _status()
    assert cuda["memory_source"] == "cuda-driver" and cuda["free"] > 0, cuda


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
