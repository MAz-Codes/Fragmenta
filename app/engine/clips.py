"""Rendered clips on disk, their analysis, and a size-capped LRU store.

A clip is a WAV in `<engine>/clips/` plus a row in `index.json`. The audio
engine loads clips by id; the conductor reads the analysis (loudness, band)
to avoid stacking two events in the same band.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

ROLES = ("bed", "event", "soundmark", "oneshot")
BANDS = ("low", "lowmid", "mid", "high")
_BAND_EDGES = (0.0, 250.0, 1000.0, 4000.0, 1e9)


@dataclass
class Clip:
    id: str
    role: str
    path: str
    duration: float
    sample_rate: int
    channels: int
    bytes: int
    created: float
    last_used: float
    event_type: str = ""
    scene: str = ""
    blend: Dict[str, float] = field(default_factory=dict)
    seed: int = 0
    prompt: str = ""
    analysis: Dict[str, Any] = field(default_factory=dict)
    render_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyse(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Loudness (RMS dBFS after a 60 Hz high-pass), peak, dominant band, onset."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    mono = a.mean(axis=0)
    n = mono.shape[0]
    if n == 0:
        return {"loudness_db": -120.0, "peak_db": -120.0, "band": "mid", "centroid_hz": 0.0, "onset_s": 0.0, "bands": {}}
    # one-pole high-pass at 60 Hz: the phone room tone shouldn't set the level
    from scipy.signal import lfilter
    rc = 1.0 / (2 * np.pi * 60.0)
    alpha = rc / (rc + 1.0 / sr)
    hp = lfilter([alpha, -alpha], [1.0, -alpha], mono).astype(np.float32)
    rms = float(np.sqrt(np.mean(hp * hp) + 1e-12))
    peak = float(np.max(np.abs(a)) + 1e-12)
    # spectrum of up to the middle 8 s
    seg = mono[: min(n, sr * 8)]
    win = np.hanning(seg.shape[0]).astype(np.float32)
    spec = np.abs(np.fft.rfft(seg * win)) ** 2
    freqs = np.fft.rfftfreq(seg.shape[0], 1.0 / sr)
    total = float(spec.sum() + 1e-12)
    centroid = float((freqs * spec).sum() / total)
    bands = {}
    for name, lo, hi in zip(BANDS, _BAND_EDGES[:-1], _BAND_EDGES[1:]):
        m = (freqs >= lo) & (freqs < hi)
        bands[name] = float(spec[m].sum() / total)
    band = max(bands, key=bands.get)
    thresh = 10 ** (-40 / 20)
    above = np.nonzero(np.abs(mono) > thresh)[0]
    onset = float(above[0] / sr) if above.size else 0.0
    return {
        "loudness_db": round(20 * np.log10(rms), 2),
        "peak_db": round(20 * np.log10(peak), 2),
        "band": band, "bands": {k: round(v, 4) for k, v in bands.items()},
        "centroid_hz": round(centroid, 1), "onset_s": round(onset, 3),
    }


class ClipStore:
    def __init__(self, root: Path, cap_bytes: int = 4 * 1024 ** 3) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cap_bytes = int(cap_bytes)
        self._index_path = self.root / "index.json"
        self._clips: Dict[str, Clip] = {}
        self._pinned: set = set()
        self._lock = threading.RLock()
        self._load_index()

    # --- index ---------------------------------------------------------------
    def _load_index(self) -> None:
        try:
            rows = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows = []
        for r in rows:
            try:
                c = Clip(**r)
            except TypeError:
                continue
            if Path(c.path).exists():
                self._clips[c.id] = c

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([c.to_dict() for c in self._clips.values()]), encoding="utf-8")
        os.replace(tmp, self._index_path)

    # --- api -----------------------------------------------------------------
    def add(self, audio: np.ndarray, sr: int, role: str, **meta: Any) -> Clip:
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        cid = uuid.uuid4().hex[:12]
        path = self.root / f"{role}_{cid}.wav"
        sf.write(str(path), a.T, sr, subtype="FLOAT")
        now = time.time()
        clip = Clip(
            id=cid, role=role, path=str(path), duration=round(a.shape[1] / sr, 3),
            sample_rate=int(sr), channels=int(a.shape[0]), bytes=path.stat().st_size,
            created=now, last_used=now, analysis=meta.pop("analysis", None) or analyse(a, sr),
            **{k: v for k, v in meta.items() if k in Clip.__dataclass_fields__},
        )
        with self._lock:
            self._clips[cid] = clip
            self._evict_locked()
            self._save_index()
        return clip

    def get(self, cid: str) -> Optional[Clip]:
        with self._lock:
            return self._clips.get(cid)

    def load(self, cid: str) -> Optional[np.ndarray]:
        clip = self.get(cid)
        if clip is None:
            return None
        try:
            data, _sr = sf.read(clip.path, dtype="float32", always_2d=True)
        except (OSError, RuntimeError):
            with self._lock:
                self._clips.pop(cid, None)
            return None
        with self._lock:
            clip.last_used = time.time()
        return np.ascontiguousarray(data.T)

    def touch(self, cid: str) -> None:
        with self._lock:
            c = self._clips.get(cid)
            if c:
                c.last_used = time.time()

    def pin(self, cid: str) -> None:
        with self._lock:
            self._pinned.add(cid)

    def unpin(self, cid: str) -> None:
        with self._lock:
            self._pinned.discard(cid)

    def remove(self, cid: str) -> bool:
        with self._lock:
            c = self._clips.pop(cid, None)
            self._pinned.discard(cid)
            if c is None:
                return False
            try:
                Path(c.path).unlink()
            except OSError:
                pass
            self._save_index()
            return True

    def list(self, role: Optional[str] = None, event_type: Optional[str] = None,
             scene: Optional[str] = None) -> List[Clip]:
        with self._lock:
            out = [c for c in self._clips.values()
                   if (role is None or c.role == role)
                   and (event_type is None or c.event_type == event_type)
                   and (scene is None or c.scene == scene)]
        return sorted(out, key=lambda c: c.created)

    def total_bytes(self) -> int:
        with self._lock:
            return sum(c.bytes for c in self._clips.values())

    def _evict_locked(self) -> None:
        total = sum(c.bytes for c in self._clips.values())
        if total <= self.cap_bytes:
            return
        for c in sorted(self._clips.values(), key=lambda c: c.last_used):
            if total <= self.cap_bytes:
                break
            if c.id in self._pinned:
                continue
            self._clips.pop(c.id, None)
            total -= c.bytes
            try:
                Path(c.path).unlink()
            except OSError:
                pass

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            by_role: Dict[str, int] = {}
            for c in self._clips.values():
                by_role[c.role] = by_role.get(c.role, 0) + 1
            return {"count": len(self._clips), "bytes": sum(c.bytes for c in self._clips.values()),
                    "cap_bytes": self.cap_bytes, "by_role": by_role}
