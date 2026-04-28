"""Ableton Link integration.

The native Link library is wrapped by several PyPI packages (LinkPython-extern,
LinkPython, linkpy, link) — we try a few import names and gracefully no-op if
none are installed. That keeps the rest of the app working on machines without
the binding; users who want Link tempo sync install it themselves:

    pip install LinkPython-extern

The bridge is a module-level singleton because Link is intrinsically a
process-global resource (one peer session per process).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def _try_import_link():
    """Return the first available Link module, or None."""
    for name in ("link", "LinkPython", "linkpy"):
        try:
            return __import__(name)
        except ImportError:
            continue
    return None


class LinkBridge:
    """Thin, thread-safe wrapper around the native Link session.

    The various Python Link bindings use slightly different spellings
    (``Link`` vs ``LinkSession``, property vs setter for ``enabled``,
    ``captureAppSessionState`` vs ``captureSessionState``). We probe for
    whichever exists so this works across packages.
    """

    def __init__(self, initial_bpm: float = 120.0) -> None:
        self._mod = _try_import_link()
        self._link: Optional[Any] = None
        self._enabled = False
        self._last_bpm = float(initial_bpm)
        self._lock = threading.Lock()
        if self._mod is not None:
            logger.info(f"Ableton Link binding found: {self._mod.__name__}")

    @property
    def available(self) -> bool:
        return self._mod is not None

    def _ensure_session(self) -> None:
        if self._link is not None or self._mod is None:
            return
        for ctor_name in ("Link", "LinkSession"):
            ctor = getattr(self._mod, ctor_name, None)
            if ctor is not None:
                try:
                    self._link = ctor(self._last_bpm)
                    # Tempo sync works without this; transport (play/stop) sync
                    # is opt-in per peer. Live exposes the same toggle as
                    # "Start Stop Sync" in its Link/Tempo/MIDI prefs.
                    self._set_start_stop_sync(True)
                    return
                except Exception as exc:
                    logger.warning(f"Link ctor {ctor_name} failed: {exc}")
        logger.warning("Loaded Link module but no known constructor was found")

    def _set_start_stop_sync(self, value: bool) -> None:
        if self._link is None:
            return
        try:
            self._link.startStopSyncEnabled = value
            return
        except AttributeError:
            pass
        for setter_name in ("setStartStopSyncEnabled", "set_start_stop_sync_enabled"):
            setter = getattr(self._link, setter_name, None)
            if setter:
                try:
                    setter(value)
                    return
                except Exception as exc:
                    logger.warning(f"Link {setter_name} failed: {exc}")

    def _set_enabled_on_link(self, value: bool) -> None:
        if self._link is None:
            return
        try:
            self._link.enabled = value
        except AttributeError:
            setter = getattr(self._link, "set_enabled", None) or getattr(self._link, "setEnabled", None)
            if setter:
                setter(value)

    def _capture_session(self):
        for name in ("captureAppSessionState", "captureSessionState"):
            fn = getattr(self._link, name, None)
            if fn is not None:
                return fn()
        return None

    def _commit_session(self, session) -> None:
        for name in ("commitAppSessionState", "commitSessionState"):
            fn = getattr(self._link, name, None)
            if fn is not None:
                fn(session)
                return

    def enable(self) -> bool:
        if not self.available:
            return False
        with self._lock:
            self._ensure_session()
            if self._link is None:
                return False
            self._set_enabled_on_link(True)
            self._enabled = True
        return True

    def disable(self) -> None:
        with self._lock:
            if self._link is not None:
                self._set_enabled_on_link(False)
            self._enabled = False

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            state: Dict[str, Any] = {
                "available": self.available,
                "enabled": self._enabled,
                "bpm": self._last_bpm,
                "num_peers": 0,
                "is_playing": False,
                "beat": 0.0,
                "time_micros": 0,
            }
            if not self._enabled or self._link is None:
                return state
            try:
                session = self._capture_session()
                if session is None:
                    return state
                if hasattr(session, "tempo"):
                    state["bpm"] = float(session.tempo())
                    self._last_bpm = state["bpm"]
                num_peers = getattr(self._link, "numPeers", None)
                if callable(num_peers):
                    state["num_peers"] = int(num_peers())

                for name in ("isPlaying", "is_playing"):
                    fn = getattr(session, name, None)
                    if callable(fn):
                        state["is_playing"] = bool(fn())
                        break

                # Sample beat + host time together so the client can extrapolate
                # forward by (now - capturedAt) when scheduling launches.
                clock_fn = getattr(self._link, "clock", None)
                if callable(clock_fn):
                    micros = int(clock_fn().micros())
                    state["time_micros"] = micros
                    # Quantum value here only affects phase wrapping for peers
                    # that just joined; for an existing session the absolute
                    # beat is stable. 4 = one bar in 4/4, a sane default.
                    for name in ("beatAtTime", "beat_at_time"):
                        fn = getattr(session, name, None)
                        if callable(fn):
                            state["beat"] = float(fn(micros, 4.0))
                            break
            except Exception as exc:
                logger.warning(f"Link state read failed: {exc}")
        return state

    def set_bpm(self, bpm: float) -> bool:
        with self._lock:
            self._last_bpm = float(bpm)
            if not self._enabled or self._link is None:
                return False
            try:
                session = self._capture_session()
                if session is None:
                    return False
                clock = getattr(self._link, "clock", None)
                micros = clock().micros() if callable(clock) else 0
                if hasattr(session, "setTempo"):
                    session.setTempo(float(bpm), micros)
                elif hasattr(session, "set_tempo"):
                    session.set_tempo(float(bpm), micros)
                else:
                    return False
                self._commit_session(session)
                return True
            except Exception as exc:
                logger.warning(f"Link set_bpm failed: {exc}")
                return False


_bridge: Optional[LinkBridge] = None
_bridge_lock = threading.Lock()


def get_link_bridge() -> LinkBridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = LinkBridge()
        return _bridge
