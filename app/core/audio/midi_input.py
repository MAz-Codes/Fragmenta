"""Native MIDI input.

Reads hardware MIDI via python-rtmidi (CoreMIDI on macOS, WinMM on Windows,
ALSA on Linux) so MIDI works regardless of the web engine the OS gives us —
WKWebView has no Web MIDI, WebView2's is flaky. Same pattern as the native
Ableton Link binding in link_sync.py: wrap an optional native lib and no-op
gracefully if it isn't importable.

The backend owns the *transport* only: it enumerates input ports, opens one,
and broadcasts incoming messages to subscribers (drained by the SSE endpoint
in app.py). All mapping / learn / takeover logic stays in the frontend
MidiContext — it just consumes these events instead of Web MIDI.
"""
from __future__ import annotations

import ctypes
import glob
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional


def _preload_bundled_jack() -> None:
    """Work around a broken RPATH in python-rtmidi's manylinux wheel.

    The wheel bundles libjack as `python_rtmidi/libjack-<hash>.so.*`, but the
    `_rtmidi` extension's RPATH points at a directory that doesn't exist
    (`$ORIGIN/../python_rtmidi.` — note the stray trailing dot), so the loader
    can't find it and `import rtmidi` dies with
    `ImportError: libjack-<hash>.so...: cannot open shared object file`.

    The bundled lib's soname matches the extension's DT_NEEDED exactly, so
    dlopen'ing it with RTLD_GLOBAL first lets the loader satisfy the dependency
    from the already-loaded object. Doing it here (rather than patching the
    venv) survives a pip reinstall and needs no patchelf/root. Linux-only; a
    no-op everywhere the glob finds nothing.
    """
    if not sys.platform.startswith("linux"):
        return
    for base in sys.path:
        if not base or not os.path.isdir(base):
            continue
        for lib in glob.glob(os.path.join(base, "python_rtmidi*", "libjack-*.so*")):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


try:
    import rtmidi  # python-rtmidi
    _RTMIDI_OK = True
except Exception:  # pragma: no cover - import guard
    # Most likely the bundled-libjack RPATH bug — preload it and retry once.
    try:
        _preload_bundled_jack()
        import rtmidi
        _RTMIDI_OK = True
    except Exception:
        rtmidi = None
        _RTMIDI_OK = False

_lock = threading.Lock()
_inputs: Dict[str, Any] = {}         # open ports: name -> rtmidi.MidiIn
_subscribers: List["queue.Queue"] = []


_coremidi_lock = threading.Lock()
_coremidi_started = False


def _ensure_coremidi_runloop() -> None:
    """Let a long-running process see MIDI devices plugged in after it started.

    CoreMIDI updates its device list on the run loop of the thread that created
    the process's first client, and our Flask threads don't run one, so a new
    device stayed invisible until the app restarted. Keep one thread, with a
    client and a run loop, for the life of the process. macOS only.
    """
    global _coremidi_started
    if sys.platform != "darwin" or not _RTMIDI_OK:
        return
    with _coremidi_lock:
        if _coremidi_started:
            return
        _coremidi_started = True
    ready = threading.Event()

    def pump() -> None:
        try:
            cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            cf.CFRunLoopRunInMode.restype = ctypes.c_int32
            cf.CFRunLoopRunInMode.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_bool]
            mode = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")
            client = rtmidi.MidiIn()  # noqa: F841 - kept alive with the thread
        finally:
            ready.set()
        run_finished = 1  # kCFRunLoopRunFinished: no sources yet, so it returns at once
        while True:
            if cf.CFRunLoopRunInMode(mode, 1.0, False) == run_finished:
                time.sleep(0.25)

    threading.Thread(target=pump, name="coremidi-runloop", daemon=True).start()
    ready.wait(timeout=2)


def is_available() -> bool:
    """True if the native MIDI backend is importable."""
    return _RTMIDI_OK


def list_inputs() -> List[Dict[str, Any]]:
    """Enumerate input ports. `id` is the port name (stable across index
    shuffles); `index` is its current rtmidi index."""
    if not _RTMIDI_OK:
        return []
    _ensure_coremidi_runloop()
    mi = rtmidi.MidiIn()
    try:
        names = mi.get_ports()
    finally:
        mi.delete()
    return [{"id": name, "name": name, "index": i} for i, name in enumerate(names)]


def current_ports() -> List[str]:
    with _lock:
        return sorted(_inputs)


def _on_message(event, port=None) -> None:
    """rtmidi callback (runs on its own thread). `event` is (message, delta),
    `port` the name of the port it came in on. Broadcast the raw status/data
    bytes so the frontend can reuse its Web-MIDI-shaped dispatcher, plus the
    port, so mappings can tell two controllers apart."""
    message, _delta = event
    payload = {"data": list(message), "port": port}
    with _lock:
        subs = list(_subscribers)
    for q in subs:
        try:
            q.put_nowait(payload)
        except queue.Full:
            pass  # slow consumer — drop rather than block the MIDI thread


# Serializes whole select operations (close + enumerate + open + assign).
# Distinct from `_lock`, which only guards quick state reads/writes and is
# taken inside the close path (holding it across the whole select would
# deadlock). Without this, two concurrent /api/midi/select calls interleave
# and the losing MidiIn leaks open with a live callback — its events keep
# flowing alongside the winning port's until process exit.
_select_lock = threading.Lock()


def _close_locked(name: str) -> None:
    """Close one port. Caller must hold _select_lock."""
    with _lock:
        mi = _inputs.pop(name, None)
    if mi is None:
        return
    for step in (mi.cancel_callback, mi.close_port, mi.delete):
        try:
            step()
        except Exception:
            pass


def close_input() -> None:
    """Close every open port."""
    with _select_lock:
        for name in current_ports():
            _close_locked(name)


def open_inputs(port_ids: List[str]) -> Dict[str, List[str]]:
    """Listen to exactly these ports (by name): close the others, open the new
    ones. Returns the ports now open, the ones not plugged in, and the ones
    that are plugged in but wouldn't open (on Windows, another app holding
    the port)."""
    result: Dict[str, List[str]] = {"open": [], "missing": [], "failed": []}
    if not _RTMIDI_OK:
        return result
    _ensure_coremidi_runloop()
    wanted = [p for p in dict.fromkeys(port_ids) if p]
    with _select_lock:
        for name in current_ports():
            if name not in wanted:
                _close_locked(name)

        probe = rtmidi.MidiIn()
        try:
            present = probe.get_ports()
        finally:
            probe.delete()
        for name in wanted:
            if name in current_ports():
                continue
            if name not in present:
                result["missing"].append(name)
                continue
            mi = rtmidi.MidiIn()
            try:
                mi.open_port(present.index(name))
            except Exception:
                mi.delete()
                result["failed"].append(name)
                continue
            # Drop sysex / timing-clock / active-sensing so the stream stays to
            # the control messages the mapper cares about (CC + notes).
            mi.ignore_types(sysex=True, timing=True, active_sense=True)
            mi.set_callback(_on_message, name)
            with _lock:
                _inputs[name] = mi
    result["open"] = current_ports()
    return result


def subscribe() -> "queue.Queue":
    q: "queue.Queue" = queue.Queue(maxsize=512)
    with _lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: "queue.Queue") -> None:
    with _lock:
        if q in _subscribers:
            _subscribers.remove(q)
