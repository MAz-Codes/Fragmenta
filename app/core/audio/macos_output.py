"""Output device choice on macOS.

WKWebView can't send Web Audio to a chosen device (no AudioContext.setSinkId),
but it always plays on the Mac's default output, with every channel that
device has. So the backend picks the device the native way: it lists output
devices through Core Audio and makes the chosen one the default output. The
Main / Cue channel pairs then work as they do on Windows.

The choice is saved and applied again at startup, before the window opens.
The output the Mac had before is put back when the backend exits. Plain
ctypes against the system frameworks: no extra dependency, and a no-op on
other platforms.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


_SYSTEM_OBJECT = 1
_DEVICES = _fourcc("dev#")
_DEFAULT_OUTPUT = _fourcc("dOut")
_NAME = _fourcc("lnam")
_UID = _fourcc("uid ")
_STREAM_CONFIG = _fourcc("slay")
_HIDDEN = _fourcc("hidn")
_SCOPE_GLOBAL = _fourcc("glob")
_SCOPE_OUTPUT = _fourcc("outp")
_ELEMENT_MAIN = 0
_UTF8 = 0x08000100


class _Address(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32),
                ("scope", ctypes.c_uint32),
                ("element", ctypes.c_uint32)]


class _AudioBuffer(ctypes.Structure):
    _fields_ = [("channels", ctypes.c_uint32),
                ("byte_size", ctypes.c_uint32),
                ("data", ctypes.c_void_p)]


_ca = None
_cf = None
_lock = threading.Lock()
_original_uid: Optional[str] = None
_restore_installed = False


def _load() -> bool:
    global _ca, _cf
    if sys.platform != "darwin":
        return False
    if _ca is not None:
        return True
    try:
        ca = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    except OSError as exc:
        logger.warning(f"Core Audio unavailable: {exc}")
        return False
    addr_p = ctypes.POINTER(_Address)
    ca.AudioObjectGetPropertyDataSize.argtypes = [
        ctypes.c_uint32, addr_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    ca.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32, addr_p, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    ca.AudioObjectSetPropertyData.argtypes = [
        ctypes.c_uint32, addr_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    for fn in (ca.AudioObjectGetPropertyDataSize, ca.AudioObjectGetPropertyData,
               ca.AudioObjectSetPropertyData):
        fn.restype = ctypes.c_int32
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    _ca, _cf = ca, cf
    return True


def _get_bytes(obj: int, selector: int, scope: int = _SCOPE_GLOBAL) -> Optional[ctypes.Array]:
    addr = _Address(selector, scope, _ELEMENT_MAIN)
    size = ctypes.c_uint32(0)
    if _ca.AudioObjectGetPropertyDataSize(obj, ctypes.byref(addr), 0, None, ctypes.byref(size)) != 0:
        return None
    buf = (ctypes.c_char * max(size.value, 1))()
    if _ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None, ctypes.byref(size), buf) != 0:
        return None
    return buf


def _get_u32(obj: int, selector: int) -> Optional[int]:
    addr = _Address(selector, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    value = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(value))
    if _ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None, ctypes.byref(size),
                                      ctypes.byref(value)) != 0:
        return None
    return value.value


def _get_string(obj: int, selector: int) -> Optional[str]:
    addr = _Address(selector, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    ref = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(ref))
    if _ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None, ctypes.byref(size),
                                      ctypes.byref(ref)) != 0 or not ref.value:
        return None
    try:
        out = ctypes.create_string_buffer(1024)
        return out.value.decode("utf-8") if _cf.CFStringGetCString(ref, out, 1024, _UTF8) else None
    finally:
        _cf.CFRelease(ref)


def _output_channels(dev: int) -> int:
    buf = _get_bytes(dev, _STREAM_CONFIG, _SCOPE_OUTPUT)
    if buf is None or len(buf) < 4:
        return 0
    count = int.from_bytes(bytes(buf[:4]), sys.byteorder)
    # AudioBufferList: a UInt32 count, then the buffers at pointer alignment.
    offset = ctypes.alignment(_AudioBuffer)
    total = 0
    for i in range(count):
        start = offset + i * ctypes.sizeof(_AudioBuffer)
        if start + ctypes.sizeof(_AudioBuffer) > len(buf):
            break
        total += _AudioBuffer.from_buffer_copy(bytes(buf[start:start + ctypes.sizeof(_AudioBuffer)])).channels
    return total


def _device_ids() -> List[int]:
    buf = _get_bytes(_SYSTEM_OBJECT, _DEVICES)
    if buf is None:
        return []
    n = len(buf) // 4
    return list((ctypes.c_uint32 * n).from_buffer_copy(bytes(buf[:n * 4])))


def is_available() -> bool:
    return _load()


def list_outputs() -> List[Dict[str, Any]]:
    """Every visible device with output channels: {uid, name, channels}."""
    if not _load():
        return []
    out = []
    for dev in _device_ids():
        channels = _output_channels(dev)
        if channels <= 0 or _get_u32(dev, _HIDDEN):
            continue
        uid = _get_string(dev, _UID)
        if uid:
            out.append({"uid": uid, "name": _get_string(dev, _NAME) or uid, "channels": channels})
    return out


def current_output() -> Optional[str]:
    """UID of the Mac's default output device."""
    if not _load():
        return None
    dev = _get_u32(_SYSTEM_OBJECT, _DEFAULT_OUTPUT)
    return _get_string(dev, _UID) if dev else None


def _set_default(uid: str) -> bool:
    for dev in _device_ids():
        if _get_string(dev, _UID) == uid:
            addr = _Address(_DEFAULT_OUTPUT, _SCOPE_GLOBAL, _ELEMENT_MAIN)
            value = ctypes.c_uint32(dev)
            return _ca.AudioObjectSetPropertyData(
                _SYSTEM_OBJECT, ctypes.byref(addr), 0, None,
                ctypes.sizeof(value), ctypes.byref(value)) == 0
    return False


def select_output(uid: str, remember: bool = True) -> bool:
    """Make `uid` the Mac's default output. The first change remembers the
    output the Mac had, so it can be put back on exit."""
    global _original_uid
    if not _load() or not uid:
        return False
    with _lock:
        before = current_output()
        if before == uid:
            ok = True
        else:
            ok = _set_default(uid)
            if ok and _original_uid is None:
                _original_uid = before
        if ok and remember:
            _save_choice(uid)
        return ok


def restore_original() -> None:
    global _original_uid
    with _lock:
        uid, _original_uid = _original_uid, None
    if uid and _load() and current_output() != uid:
        if _set_default(uid):
            logger.info("Restored the Mac's previous audio output")


def _choice_path() -> Path:
    from app.core.config import get_config
    return get_config().user_data_dir / "config" / "audio_output.json"


def _save_choice(uid: str) -> None:
    try:
        path = _choice_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"uid": uid}), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning(f"Could not save the audio output choice: {exc}")


def saved_choice() -> Optional[str]:
    try:
        return json.loads(_choice_path().read_text(encoding="utf-8")).get("uid") or None
    except Exception:
        return None


def start() -> None:
    """Backend startup (main thread): put the saved device back before the
    window opens, and arrange for the previous output to return on exit."""
    global _restore_installed
    if not _load() or _restore_installed:
        return
    _restore_installed = True
    atexit.register(restore_original)
    # start.py stops the backend with SIGTERM; turn it into a normal exit so
    # atexit runs.
    previous = signal.getsignal(signal.SIGTERM)

    def _on_term(signum, frame):
        restore_original()
        if callable(previous):
            previous(signum, frame)
        raise SystemExit(143)

    try:
        signal.signal(signal.SIGTERM, _on_term)
    except (ValueError, OSError):
        pass
    uid = saved_choice()
    if uid and any(d["uid"] == uid for d in list_outputs()):
        select_output(uid, remember=False)
