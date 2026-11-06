"""OSC 1.0, in-house: int32, float32, string, blob, True, False, bundles.

Enough for TouchDesigner, Max and SuperCollider to talk to the engine, with
no dependency. `python-osc` can replace the codec behind `encode` / `decode`.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

BUNDLE_TAG = b"#bundle\0"


def _pad(b: bytes) -> bytes:
    return b + b"\0" * ((4 - len(b) % 4) % 4)


def _string(s: str) -> bytes:
    return _pad(s.encode("utf-8") + b"\0")


def _read_string(data: bytes, i: int) -> Tuple[str, int]:
    end = data.index(b"\0", i)
    s = data[i:end].decode("utf-8", errors="replace")
    i = end + 1
    i += (4 - i % 4) % 4
    return s, i


def encode(address: str, *args: Any) -> bytes:
    tags = ","
    body = b""
    for a in args:
        if isinstance(a, bool):
            tags += "T" if a else "F"
        elif isinstance(a, int):
            tags += "i"
            body += struct.pack(">i", int(max(-2**31, min(2**31 - 1, a))))
        elif isinstance(a, float):
            tags += "f"
            body += struct.pack(">f", a)
        elif isinstance(a, (bytes, bytearray)):
            tags += "b"
            body += struct.pack(">i", len(a)) + _pad(bytes(a))
        else:
            tags += "s"
            body += _string(str(a))
    return _string(address) + _string(tags) + body


def decode(data: bytes) -> List[Tuple[str, List[Any]]]:
    """Returns [(address, args), …]; a bundle yields its messages in order."""
    if data.startswith(BUNDLE_TAG):
        out: List[Tuple[str, List[Any]]] = []
        i = 16  # tag + timetag
        while i + 4 <= len(data):
            (size,) = struct.unpack(">i", data[i:i + 4])
            i += 4
            out.extend(decode(data[i:i + size]))
            i += size
        return out
    address, i = _read_string(data, 0)
    args: List[Any] = []
    if i >= len(data):
        return [(address, args)]
    tags, i = _read_string(data, i)
    for t in tags.lstrip(","):
        if t == "i":
            (v,) = struct.unpack(">i", data[i:i + 4]); i += 4; args.append(v)
        elif t == "f":
            (v,) = struct.unpack(">f", data[i:i + 4]); i += 4; args.append(v)
        elif t == "s":
            v, i = _read_string(data, i); args.append(v)
        elif t == "b":
            (n,) = struct.unpack(">i", data[i:i + 4]); i += 4
            args.append(data[i:i + n]); i += n + (4 - n % 4) % 4
        elif t == "T":
            args.append(True)
        elif t == "F":
            args.append(False)
        elif t == "N":
            args.append(None)
        elif t in "dh":
            (v,) = struct.unpack(">d" if t == "d" else ">q", data[i:i + 8]); i += 8; args.append(v)
        else:
            break
    return [(address, args)]


Handler = Callable[[str, List[Any]], None]


class OscServer:
    """UDP listener on its own thread; `handler(address, args)` per message."""

    def __init__(self, host: str, port: int, handler: Handler) -> None:
        self.host, self.port, self.handler = host, int(port), handler
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.received = 0
        self.last_error: Optional[str] = None

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.5)
        self.port = self._sock.getsockname()[1]
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="osc-in", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                for address, args in decode(data):
                    self.received += 1
                    self.handler(address, args)
            except Exception as exc:  # one bad packet must not stop the listener
                self.last_error = str(exc)


class OscClient:
    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, address: str, *args: Any) -> None:
        try:
            self._sock.sendto(encode(address, *args), (self.host, self.port))
            self.sent += 1
        except OSError:
            pass

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()
