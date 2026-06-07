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
_midi_in: Any = None                 # the open rtmidi.MidiIn, or None
_current_port: Optional[str] = None  # name of the open port, or None
_subscribers: List["queue.Queue"] = []


def is_available() -> bool:
    """True if the native MIDI backend is importable."""
    return _RTMIDI_OK


def list_inputs() -> List[Dict[str, Any]]:
    """Enumerate input ports. `id` is the port name (stable across index
    shuffles); `index` is its current rtmidi index."""
    if not _RTMIDI_OK:
        return []
    mi = rtmidi.MidiIn()
    try:
        names = mi.get_ports()
    finally:
        mi.delete()
    return [{"id": name, "name": name, "index": i} for i, name in enumerate(names)]


def current_port() -> Optional[str]:
    with _lock:
        return _current_port


def _on_message(event, _data=None) -> None:
    """rtmidi callback (runs on its own thread). `event` is (message, delta).
    Broadcast the raw status/data bytes so the frontend can reuse its existing
    Web-MIDI-shaped dispatcher unchanged."""
    message, _delta = event
    payload = {"data": list(message)}
    with _lock:
        subs = list(_subscribers)
    for q in subs:
        try:
            q.put_nowait(payload)
        except queue.Full:
            pass  # slow consumer — drop rather than block the MIDI thread


def close_input() -> None:
    global _midi_in, _current_port
    with _lock:
        mi = _midi_in
        _midi_in = None
        _current_port = None
    if mi is not None:
        try:
            mi.cancel_callback()
        except Exception:
            pass
        try:
            mi.close_port()
        except Exception:
            pass
        try:
            mi.delete()
        except Exception:
            pass


def open_input(port_id: Optional[str]) -> bool:
    """Open the input port whose name == port_id. A falsy port_id just closes
    the current port. Returns True on success (or on a pure close)."""
    if not _RTMIDI_OK:
        return False
    close_input()
    if not port_id:
        return True

    mi = rtmidi.MidiIn()
    idx = None
    for i, name in enumerate(mi.get_ports()):
        if name == port_id:
            idx = i
            break
    if idx is None:
        mi.delete()
        return False

    mi.open_port(idx)
    # Drop sysex / timing-clock / active-sensing so the stream stays to the
    # control messages the mapper cares about (CC + notes).
    mi.ignore_types(sysex=True, timing=True, active_sense=True)
    mi.set_callback(_on_message)

    global _midi_in, _current_port
    with _lock:
        _midi_in = mi
        _current_port = port_id
    return True


def subscribe() -> "queue.Queue":
    q: "queue.Queue" = queue.Queue(maxsize=512)
    with _lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: "queue.Queue") -> None:
    with _lock:
        if q in _subscribers:
            _subscribers.remove(q)
