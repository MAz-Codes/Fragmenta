"""The real-time layer: bed streams, event voices, a mixer, sinks.

Everything is block-based: a sink calls `Mixer.pull(n)` and gets float32
[channels, n]. Nothing here touches the model. Stereo for now, behind the
`Spatializer` interface the HOA encoder will implement (decisions/05).

Azimuth: degrees, 0 = front, positive = left. Elevation: positive = up.
Distance: metres, reference 1 m.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

SR = 44100
BLOCK = 1024


# --- spatial ----------------------------------------------------------------
class Spatializer:
    """azimuth, elevation, distance → per-output-channel gains, plus a low-pass
    cutoff for distance. Stereo implementation; `channels` is what it decodes to."""

    channels = 2
    name = "stereo"

    def gains(self, az_deg: float, el_deg: float, dist_m: float) -> Tuple[np.ndarray, float]:
        az = math.radians(az_deg)
        x = max(-1.0, min(1.0, math.sin(az)))           # +1 = hard left
        left = math.sqrt((1.0 + x) / 2.0)
        right = math.sqrt((1.0 - x) / 2.0)
        rear = max(0.0, -math.cos(az))                   # 0 front … 1 behind
        g = 1.0 / max(1.0, float(dist_m))
        g *= 1.0 - 0.25 * rear                           # behind: a little quieter
        el = max(0.0, min(1.0, abs(el_deg) / 90.0))
        g *= 1.0 - 0.15 * el
        cutoff = 18000.0 / (1.0 + 0.35 * max(0.0, dist_m - 1.0)) * (1.0 - 0.4 * rear)
        return np.array([left * g, right * g], dtype=np.float32), max(800.0, cutoff)


class OnePoleLP:
    def __init__(self, channels: int = 1) -> None:
        self.state = np.zeros(channels, dtype=np.float32)

    def process(self, block: np.ndarray, cutoff_hz: float, sr: int = SR) -> np.ndarray:
        if cutoff_hz >= sr * 0.45:
            self.state[:] = block[:, -1]
            return block
        a = 1.0 - math.exp(-2 * math.pi * cutoff_hz / sr)
        out = np.empty_like(block)
        s = self.state.copy()
        # per-sample one-pole; blocks are 1024 so a python loop is too slow →
        # use scipy's lfilter per channel
        from scipy.signal import lfilter
        for c in range(block.shape[0]):
            y, zf = lfilter([a], [1.0, -(1.0 - a)], block[c], zi=[s[c] * (1.0 - a)])
            out[c] = y
            s[c] = y[-1]
        self.state = s
        return out.astype(np.float32)


def _equal_power(n: int, up: bool) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.sin((t if up else 1.0 - t) * math.pi / 2)


# --- bed stream ---------------------------------------------------------------
class BedStream:
    """One decorrelated take of the bed: a chain of clips crossfaded end to end.

    If the next clip is not there when the current one runs out, the current
    one loops from a random offset (counted as a starvation) so audio never stops.
    """

    def __init__(self, key: str, sr: int = SR, crossfade_s: float = 8.0, seed: int = 0) -> None:
        self.key = key
        self.sr = sr
        self.crossfade_s = crossfade_s
        self._queue: List[Tuple[str, np.ndarray]] = []
        self._cur: Optional[np.ndarray] = None
        self._cur_id = ""
        self._pos = 0
        self._next: Optional[np.ndarray] = None
        self._next_id = ""
        self._xf_pos = -1
        self.gain = 1.0
        self._gain_cur = 0.0
        self._gain_target = 1.0
        self._gain_rate = 0.0            # per second
        self.azimuth = 0.0
        self._az_rate = 0.0
        self.spread = 1.0
        self.state = "entering"
        self.starvations = 0
        self.played_ids: List[str] = []
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self.spatializer = Spatializer()

    # --- feeding -------------------------------------------------------------
    def push(self, clip_id: str, audio: np.ndarray) -> None:
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 1:
            a = np.stack([a, a])
        elif a.shape[0] == 1:
            a = np.concatenate([a, a], axis=0)
        with self._lock:
            self._queue.append((clip_id, a))

    def queued(self) -> int:
        with self._lock:
            return len(self._queue) + (1 if self._next is not None else 0)

    def ahead_seconds(self) -> float:
        with self._lock:
            s = 0.0
            if self._cur is not None:
                s += max(0, self._cur.shape[1] - self._pos) / self.sr
            if self._next is not None:
                s += self._next.shape[1] / self.sr
            s += sum(a.shape[1] for _, a in self._queue) / self.sr
            return s

    def enter_from(self, azimuth: float, seconds: float) -> None:
        self.azimuth = float(azimuth)
        self._az_rate = -self.azimuth / max(0.1, seconds)
        self._gain_cur = 0.0
        self._gain_target = self.gain
        self._gain_rate = self.gain / max(0.1, seconds)
        self.state = "entering"

    def fade_in(self, seconds: float) -> None:
        self.enter_from(0.0, seconds)

    def recede(self, seconds: float) -> None:
        self._gain_target = 0.0
        self._gain_rate = -max(self._gain_cur, 1e-3) / max(0.1, seconds)
        self.state = "receding"

    @property
    def done(self) -> bool:
        return self.state == "receding" and self._gain_cur <= 1e-4

    # --- pulling -------------------------------------------------------------
    def _advance_clip(self) -> bool:
        """Make the next queued clip current. False if nothing was there."""
        if self._next is not None:
            self._cur, self._cur_id = self._next, self._next_id
            self._next, self._next_id = None, ""
            self._pos = 0
            self._xf_pos = -1
            self.played_ids.append(self._cur_id)
            return True
        if self._queue:
            self._cur_id, self._cur = self._queue.pop(0)
            self._pos = 0
            self._xf_pos = -1
            self.played_ids.append(self._cur_id)
            return True
        return False

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros((2, n), dtype=np.float32)
        with self._lock:
            if self._cur is None and not self._advance_clip():
                return out
            xf = int(self.crossfade_s * self.sr)
            written = 0
            while written < n:
                remaining = self._cur.shape[1] - self._pos
                if remaining <= 0:
                    if not self._advance_clip():
                        # starvation: loop the same clip from a random offset
                        self.starvations += 1
                        self._pos = int(self._rng.integers(0, max(1, self._cur.shape[1] // 2)))
                        remaining = self._cur.shape[1] - self._pos
                    else:
                        remaining = self._cur.shape[1] - self._pos
                take = min(n - written, remaining)
                seg = self._cur[:, self._pos:self._pos + take]
                # crossfade into the next clip over the tail of this one
                into_tail = self._pos + take - (self._cur.shape[1] - xf)
                if xf > 0 and into_tail > 0 and (self._next is not None or self._queue):
                    if self._next is None:
                        self._next_id, self._next = self._queue.pop(0)
                        self._xf_pos = 0
                    start = max(0, self._pos - (self._cur.shape[1] - xf))
                    k = min(take, xf - start) if start < xf else 0
                    if k > 0:
                        t = np.arange(start, start + k, dtype=np.float32) / xf
                        fade_out = np.cos(t * math.pi / 2)
                        fade_in = np.sin(t * math.pi / 2)
                        nxt = self._next[:, self._xf_pos:self._xf_pos + k]
                        if nxt.shape[1] < k:
                            nxt = np.pad(nxt, ((0, 0), (0, k - nxt.shape[1])))
                        mixed = seg[:, :k] * fade_out + nxt * fade_in
                        seg = np.concatenate([mixed, seg[:, k:]], axis=1)
                        self._xf_pos += k
                out[:, written:written + take] = seg
                self._pos += take
                written += take
                if self._next is not None and self._pos >= self._cur.shape[1]:
                    # the tail is done; the next clip continues from where the crossfade left it
                    self._cur, self._cur_id = self._next, self._next_id
                    self._pos = self._xf_pos
                    self._next, self._next_id = None, ""
                    self._xf_pos = -1
                    self.played_ids.append(self._cur_id)
        # width
        if self.spread < 1.0:
            mid = (out[0] + out[1]) * 0.5
            side = (out[0] - out[1]) * 0.5 * self.spread
            out = np.stack([mid + side, mid - side])
        # entering from an azimuth: collapse toward mono and pan while az → 0
        dt = n / self.sr
        if self._az_rate != 0.0:
            self.azimuth += self._az_rate * dt
            if (self._az_rate < 0 and self.azimuth <= 0) or (self._az_rate > 0 and self.azimuth >= 0):
                self.azimuth, self._az_rate = 0.0, 0.0
        if abs(self.azimuth) > 0.5:
            a = min(1.0, abs(self.azimuth) / 90.0)
            g, _ = self.spatializer.gains(self.azimuth, 0.0, 1.0)
            mono = (out[0] + out[1]) * 0.5
            panned = np.stack([mono * g[0], mono * g[1]]) * math.sqrt(2)
            out = out * (1 - a) + panned * a
        # gain ramp
        if self._gain_rate != 0.0 and self._gain_cur != self._gain_target:
            g0 = self._gain_cur
            g1 = g0 + self._gain_rate * dt
            if (self._gain_rate > 0 and g1 >= self._gain_target) or (self._gain_rate < 0 and g1 <= self._gain_target):
                g1 = self._gain_target
                self._gain_rate = 0.0
                if self.state == "entering":
                    self.state = "playing"
            ramp = np.linspace(g0, g1, n, dtype=np.float32)
            out *= ramp
            self._gain_cur = g1
        else:
            out *= self._gain_cur
            if self.state == "entering" and self._gain_cur >= self._gain_target:
                self.state = "playing"
        return out


# --- event voice ------------------------------------------------------------
@dataclass
class Trajectory:
    type: str = "static"          # static | passby | orbit | drift
    speed: float = 1.0            # passby: m/s · orbit: deg/s · drift: deg/s std
    direction: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "speed": self.speed, "direction": self.direction}


class EventVoice:
    def __init__(self, clip_id: str, audio: np.ndarray, event_type: str, azimuth: float,
                 elevation: float = 0.0, distance: float = 1.0, gain: float = 1.0,
                 trajectory: Optional[Trajectory] = None, band: str = "mid", sr: int = SR,
                 seed: int = 0) -> None:
        a = np.asarray(audio, dtype=np.float32)
        self.mono = a.mean(axis=0) if a.ndim == 2 else a
        self.clip_id, self.event_type, self.band = clip_id, event_type, band
        self.azimuth, self.elevation, self.distance = float(azimuth), float(elevation), float(distance)
        self.gain = float(gain)
        self.trajectory = trajectory or Trajectory()
        self.sr = sr
        self._pos = 0
        self._lp = OnePoleLP(2)
        self._rng = np.random.default_rng(seed)
        self.spatializer = Spatializer()
        self.started = time.time()
        # pass-by: a straight line in front of the listener through the start
        # azimuth at the given distance; x runs from -L to +L over the clip.
        dur = self.mono.shape[0] / sr
        self._passby_len = max(2.0, self.trajectory.speed * dur / 2.0)
        self._t = 0.0
        self.duration = dur
        fade = int(0.01 * sr)
        if self.mono.shape[0] > 2 * fade:
            self.mono[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            self.mono[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

    @property
    def done(self) -> bool:
        return self._pos >= self.mono.shape[0]

    def position(self) -> Tuple[float, float, float]:
        return self.azimuth, self.elevation, self.distance

    def _move(self, dt: float) -> None:
        tr = self.trajectory
        self._t += dt
        if tr.type == "orbit":
            self.azimuth = ((self.azimuth + tr.direction * tr.speed * dt + 180) % 360) - 180
        elif tr.type == "drift":
            self.azimuth = ((self.azimuth + float(self._rng.normal(0, tr.speed * dt)) + 180) % 360) - 180
        elif tr.type == "passby":
            # progress -1 … +1 along the line
            p = -1.0 + 2.0 * min(1.0, self._t / max(1e-3, self.duration))
            x = tr.direction * p * self._passby_len
            y = max(0.5, self.distance)
            self.azimuth = math.degrees(math.atan2(x, y))
            self._dist_now = math.hypot(x, y)
            return
        self._dist_now = self.distance

    def pull(self, n: int) -> np.ndarray:
        seg = self.mono[self._pos:self._pos + n]
        self._pos += n
        if seg.shape[0] < n:
            seg = np.pad(seg, (0, n - seg.shape[0]))
        self._move(n / self.sr)
        g, cutoff = self.spatializer.gains(self.azimuth, self.elevation, getattr(self, "_dist_now", self.distance))
        out = np.stack([seg * g[0], seg * g[1]]) * self.gain
        return self._lp.process(out, cutoff, self.sr)

    def to_dict(self) -> Dict[str, Any]:
        az, el, d = self.position()
        return {"clip_id": self.clip_id, "type": self.event_type, "azimuth": round(az, 1),
                "elevation": round(el, 1), "distance": round(getattr(self, "_dist_now", d), 2),
                "band": self.band, "progress": round(min(1.0, self._pos / max(1, self.mono.shape[0])), 3),
                "trajectory": self.trajectory.type, "duration": round(self.duration, 2)}


# --- limiter and mixer ------------------------------------------------------
class Limiter:
    def __init__(self, ceiling_db: float = -1.0, release_s: float = 0.15, sr: int = SR) -> None:
        self.ceiling = 10 ** (ceiling_db / 20)
        self.release = release_s
        self.sr = sr
        self._gr = 1.0
        self.reduction_db = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(block))) if block.size else 0.0
        need = self.ceiling / peak if peak > self.ceiling else 1.0
        n = block.shape[1]
        if need < self._gr:
            # instant attack over the block
            ramp = np.linspace(self._gr, need, n, dtype=np.float32)
            self._gr = need
        else:
            rel = 1.0 - math.exp(-(n / self.sr) / self.release)
            g1 = self._gr + (1.0 - self._gr) * rel
            ramp = np.linspace(self._gr, g1, n, dtype=np.float32)
            self._gr = g1
        self.reduction_db = round(20 * math.log10(max(1e-6, self._gr)), 2)
        out = block * ramp
        return np.clip(out, -self.ceiling, self.ceiling)


LAYERS = ("bed", "events", "soundmarks")
DEFAULT_ROUTING: Dict[str, Dict[str, Any]] = {k: {"mode": "spatial"} for k in LAYERS}


def _route_ok(route: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(route.get("mode", "spatial"))
    if mode not in ("spatial", "direct"):
        raise ValueError(f"routing mode must be spatial or direct, not {mode!r}")
    chans = [int(c) for c in (route.get("channels") or [])]
    if mode == "direct" and not chans:
        raise ValueError("a direct route needs channels, e.g. [3, 4]")
    if any(c < 1 for c in chans):
        raise ValueError("channels are 1-based")
    return {"mode": mode, "channels": chans[:2]}


class Mixer:
    """Layers → outputs. Every layer (each bed take, events, soundmarks) is a
    stereo block; a route sends it through the spatial decoder (stereo for
    now, HOA later) or straight to fixed output channels. The stereo monitor
    mix (Listen, meters, taps) is made from the layers before routing, so it
    always hears everything."""

    def __init__(self, sr: int = SR, channels: int = 2) -> None:
        self.sr, self.channels = sr, max(1, int(channels))
        self.beds: Dict[str, BedStream] = {}
        self.voices: List[EventVoice] = []
        self.master_gain = 1.0
        self.limiter = Limiter(sr=sr)
        self.monitor_limiter = Limiter(sr=sr)
        self.limiter_on = True
        self._lock = threading.RLock()
        self.peak = np.zeros(self.channels, dtype=np.float32)
        self.rms = np.zeros(self.channels, dtype=np.float32)
        self.monitor_peak = np.zeros(2, dtype=np.float32)
        self.frames = 0
        self._taps: List[queue.Queue] = []
        self._recorder = None
        self._record_path: Optional[Path] = None
        self.record_frames = 0
        self.on_voice_done: Optional[Callable[[EventVoice], None]] = None
        self.underruns = 0
        self.routing: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in DEFAULT_ROUTING.items()}
        self.spatial_channels: List[int] = [1, 2]

    # --- outputs -------------------------------------------------------------
    def set_channels(self, n: int) -> None:
        n = max(1, int(n))
        with self._lock:
            if n == self.channels:
                return
            self.stop_record()
            self.channels = n
            self.peak = np.zeros(n, dtype=np.float32)
            self.rms = np.zeros(n, dtype=np.float32)

    def set_routing(self, routing: Optional[Dict[str, Any]] = None, spatial_channels: Optional[List[int]] = None) -> None:
        new = {k: dict(v) for k, v in DEFAULT_ROUTING.items()}
        for name, route in (routing or {}).items():
            if name not in LAYERS and not (name.startswith("bed:") and name[4:].isdigit()):
                raise ValueError(f"unknown layer {name!r}")
            new[name] = _route_ok(route or {})
        sc = [int(c) for c in (spatial_channels or [1, 2])][:2] or [1, 2]
        with self._lock:
            self.routing = new
            self.spatial_channels = sc

    def route_for(self, layer: str) -> Dict[str, Any]:
        r = self.routing.get(layer)
        if r is None and layer.startswith("bed:"):
            r = self.routing.get("bed")
        return r or {"mode": "spatial"}

    def _place(self, out: np.ndarray, stereo: np.ndarray, route: Dict[str, Any]) -> None:
        chans = self.spatial_channels if route.get("mode", "spatial") == "spatial" else (route.get("channels") or [1, 2])
        chans = [c - 1 for c in chans if 1 <= c <= self.channels]
        if not chans:
            return
        if len(chans) == 1:
            out[chans[0]] += (stereo[0] + stereo[1]) * 0.5
        else:
            out[chans[0]] += stereo[0]
            out[chans[1]] += stereo[1]

    def output_config(self) -> Dict[str, Any]:
        with self._lock:
            return {"channels": self.channels, "routing": {k: dict(v) for k, v in self.routing.items()},
                    "spatial_channels": list(self.spatial_channels)}

    # --- content -------------------------------------------------------------
    def bed(self, key: str, create: bool = True, **kw: Any) -> Optional[BedStream]:
        with self._lock:
            b = self.beds.get(key)
            if b is None and create:
                b = BedStream(key, sr=self.sr, **kw)
                self.beds[key] = b
            return b

    def remove_bed(self, key: str) -> None:
        with self._lock:
            self.beds.pop(key, None)

    def add_voice(self, v: EventVoice) -> None:
        with self._lock:
            self.voices.append(v)

    def active_voices(self) -> List[EventVoice]:
        with self._lock:
            return list(self.voices)

    # --- taps and recording --------------------------------------------------
    def tap(self, maxsize: int = 96) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._taps.append(q)
        return q

    def untap(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._taps:
                self._taps.remove(q)

    def start_record(self, path: Path) -> Path:
        import soundfile as sf
        with self._lock:
            self.stop_record()
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._recorder = sf.SoundFile(str(path), mode="w", samplerate=self.sr, channels=self.channels, subtype="PCM_24")
            self._record_path = path
            self.record_frames = 0
            return path

    def stop_record(self) -> Optional[Path]:
        with self._lock:
            if self._recorder is None:
                return None
            try:
                self._recorder.close()
            except Exception:
                pass
            p = self._record_path
            self._recorder, self._record_path = None, None
            return p

    @property
    def recording(self) -> bool:
        return self._recorder is not None

    # --- the block -----------------------------------------------------------
    def pull(self, n: int = BLOCK) -> np.ndarray:
        with self._lock:
            layers: Dict[str, np.ndarray] = {}
            done_beds = []
            for key, b in self.beds.items():
                take = key.split(":")[-1]
                blk = b.pull(n)
                name = f"bed:{take}"
                layers[name] = layers[name] + blk if name in layers else blk
                if b.done:
                    done_beds.append(key)
            for key in done_beds:
                self.beds.pop(key, None)
            events = np.zeros((2, n), dtype=np.float32)
            marks = np.zeros((2, n), dtype=np.float32)
            finished = []
            for v in self.voices:
                blk = v.pull(n)
                if v.event_type.startswith("soundmark:"):
                    marks += blk
                else:
                    events += blk
                if v.done:
                    finished.append(v)
            for v in finished:
                self.voices.remove(v)
            layers["events"] = events
            layers["soundmarks"] = marks
            # the stereo monitor: every layer, before routing
            monitor = np.zeros((2, n), dtype=np.float32)
            out = np.zeros((self.channels, n), dtype=np.float32)
            for name, blk in layers.items():
                monitor += blk
                self._place(out, blk, self.route_for(name))
            out *= self.master_gain
            monitor *= self.master_gain
            if self.limiter_on:
                out = self.limiter.process(out)
                monitor = self.monitor_limiter.process(monitor)
            if n:
                self.peak = np.max(np.abs(out), axis=1)
                self.monitor_peak = np.max(np.abs(monitor), axis=1)
            self.rms = np.sqrt(np.mean(out * out, axis=1) + 1e-12)
            self.frames += n
            if self._recorder is not None:
                try:
                    self._recorder.write(out.T)
                    self.record_frames += n
                except Exception:
                    self.stop_record()
            if self._taps:
                pcm = (np.clip(monitor.T, -1, 1) * 32767).astype("<i2").tobytes()
                for q in self._taps:
                    try:
                        q.put_nowait(pcm)
                    except queue.Full:
                        try:
                            q.get_nowait()
                            q.put_nowait(pcm)
                        except (queue.Empty, queue.Full):
                            pass
        if finished and self.on_voice_done:
            for v in finished:
                try:
                    self.on_voice_done(v)
                except Exception:
                    pass
        return out

    def meters(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "peak_db": [round(20 * math.log10(max(1e-6, float(p))), 1) for p in self.peak],
                "rms_db": [round(20 * math.log10(max(1e-6, float(r))), 1) for r in self.rms],
                "monitor_peak_db": [round(20 * math.log10(max(1e-6, float(p))), 1) for p in self.monitor_peak],
                "limiter_db": self.limiter.reduction_db, "channels": self.channels,
                "voices": len(self.voices),
                "beds": {k: {"ahead_s": round(b.ahead_seconds(), 1), "queued": b.queued(), "state": b.state,
                             "gain": round(b._gain_cur, 3), "azimuth": round(b.azimuth, 1),
                             "starvations": b.starvations} for k, b in self.beds.items()},
                "frames": self.frames, "recording": self._recorder is not None,
                "record_s": round(self.record_frames / self.sr, 1),
            }


# --- sinks -----------------------------------------------------------------
class NullSink:
    """Pulls blocks at wall-clock pace on a thread. The file sink and the
    monitor-only mode use it; it is also what tests use."""

    name = "null"

    def __init__(self, mixer: Mixer, block: int = BLOCK, realtime: bool = True) -> None:
        self.mixer, self.block, self.realtime = mixer, block, realtime
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.blocks = 0
        self.late = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="audio-sink", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _on_block(self, block: np.ndarray) -> None:
        pass

    def _loop(self) -> None:
        period = self.block / self.mixer.sr
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                self._on_block(self.mixer.pull(self.block))
            except Exception:
                self.mixer.underruns += 1
            self.blocks += 1
            if not self.realtime:
                continue
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period * 4:
                self.late += 1
                nxt = time.perf_counter()

    def status(self) -> Dict[str, Any]:
        return {"sink": self.name, "blocks": self.blocks, "late": self.late}


class FileSink(NullSink):
    name = "file"

    def __init__(self, mixer: Mixer, path: Path, block: int = BLOCK, realtime: bool = True) -> None:
        super().__init__(mixer, block, realtime)
        import soundfile as sf
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = sf.SoundFile(str(self.path), mode="w", samplerate=mixer.sr, channels=mixer.channels, subtype="PCM_24")

    def _on_block(self, block: np.ndarray) -> None:
        self._file.write(block.T)

    def stop(self) -> None:
        super().stop()
        try:
            self._file.close()
        except Exception:
            pass


def list_output_devices() -> List[Dict[str, Any]]:
    """Output devices PortAudio sees (empty, with no error, when sounddevice is absent)."""
    try:
        import sounddevice as sd
    except Exception:
        return []
    out = []
    try:
        apis = sd.query_hostapis()
        default = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_output_channels", 0) <= 0:
                continue
            out.append({"index": i, "name": d["name"], "channels": int(d["max_output_channels"]),
                        "samplerate": int(d.get("default_samplerate", 0)),
                        "hostapi": apis[d["hostapi"]]["name"] if 0 <= d.get("hostapi", -1) < len(apis) else "",
                        "default": i == default})
    except Exception:
        pass
    return out


class SoundDeviceSink:
    """PortAudio through `sounddevice`, when it is installed (an extra)."""

    name = "sounddevice"

    def __init__(self, mixer: Mixer, device: Optional[Any] = None, block: int = BLOCK) -> None:
        import sounddevice as sd  # noqa: F401  (ImportError propagates to the factory)
        self._sd = sd
        self.mixer, self.block = mixer, block
        self.device = device if device not in ("", None, "default") else None
        if isinstance(self.device, str) and self.device.isdigit():
            self.device = int(self.device)
        self._stream = None
        self.blocks = 0
        self.late = 0
        self.error = ""
        # fail now, not in the callback, if the device can't do this many channels
        sd.check_output_settings(device=self.device, channels=mixer.channels, samplerate=mixer.sr, dtype="float32")
        self.device_name = str(sd.query_devices(self.device, "output")["name"]) if self.device is not None else "default"

    def start(self) -> None:
        sd = self._sd

        def callback(outdata, frames, _time, status):
            if status:
                self.late += 1
            try:
                outdata[:] = self.mixer.pull(frames).T
            except Exception as exc:
                outdata.fill(0)
                self.error = str(exc)
                self.mixer.underruns += 1
            self.blocks += 1

        self._stream = sd.OutputStream(samplerate=self.mixer.sr, channels=self.mixer.channels,
                                       blocksize=self.block, device=self.device, dtype="float32",
                                       latency="high", callback=callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def status(self) -> Dict[str, Any]:
        return {"sink": self.name, "blocks": self.blocks, "late": self.late, "device": self.device_name,
                "channels": self.mixer.channels, "error": self.error}


def make_sink(kind: str, mixer: Mixer, path: Optional[Path] = None, device: Optional[Any] = None,
              realtime: bool = True, targets: Optional[List[Any]] = None) -> Tuple[Any, str]:
    """`auto` = sounddevice if importable, else null. Returns (sink, note)."""
    kind = (kind or "auto").lower()
    if kind == "radio":
        if not targets:
            return NullSink(mixer, realtime=realtime), "radio sink needs at least one stream target; null sink instead"
        return RadioSink(mixer, targets, realtime=realtime), ""
    if kind in ("auto", "sounddevice"):
        try:
            return SoundDeviceSink(mixer, device=device), ""
        except Exception as exc:  # ImportError, PortAudio errors
            if kind == "sounddevice":
                raise
            return NullSink(mixer, realtime=realtime), f"sounddevice unavailable ({exc.__class__.__name__}); null sink, monitor stream only"
    if kind == "file":
        return FileSink(mixer, path or Path("engine_output.wav"), realtime=realtime), ""
    return NullSink(mixer, realtime=realtime), ""


def wav_header(sr: int = SR, channels: int = 2, bits: int = 16) -> bytes:
    """A WAV header with an open-ended data size, for a live stream."""
    import struct
    byte_rate = sr * channels * bits // 8
    block_align = channels * bits // 8
    size = 0xFFFFFFFF
    return (b"RIFF" + struct.pack("<I", size) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, block_align, bits)
            + b"data" + struct.pack("<I", size - 44))


# --- radio -------------------------------------------------------------------
def _mask(url: str) -> str:
    import re
    return re.sub(r"://([^:/@]+):[^@/]+@", r"://\1:***@", url)


def radio_target(url: str, name: str = "", description: str = "") -> Dict[str, Any]:
    """A stream target from its URL; the codec follows the mount's extension."""
    low = url.lower()
    if low.endswith((".opus", ".ogg", ".oga")):
        codec, fmt, bitrate, ctype = "libopus", "ogg", "96k", "audio/ogg"
    elif low.endswith(".aac"):
        codec, fmt, bitrate, ctype = "aac", "adts", "128k", "audio/aac"
    else:
        codec, fmt, bitrate, ctype = "libmp3lame", "mp3", "128k", "audio/mpeg"
    return {"url": url, "codec": codec, "format": fmt, "bitrate": bitrate, "content_type": ctype,
            "name": name, "description": description}


class RadioSink(NullSink):
    """The 24/7 stream: PCM into ffmpeg, ffmpeg into Icecast (or any URL/file
    ffmpeg can write). Reconnects with backoff; audio keeps running meanwhile.
    """

    name = "radio"

    def __init__(self, mixer: Mixer, targets: List[Dict[str, Any]], block: int = BLOCK, realtime: bool = True,
                 ffmpeg: str = "ffmpeg") -> None:
        super().__init__(mixer, block, realtime)
        self.targets = [t if isinstance(t, dict) else radio_target(str(t)) for t in targets]
        self.ffmpeg = ffmpeg
        self._proc = None
        self._stderr_tail: List[str] = []
        self.connected = False
        self.restarts = 0
        self.bytes_out = 0
        self.last_error = ""
        self._retry_at = 0.0
        self._backoff = 1.0
        self._plock = threading.Lock()
        self.title = ""

    def _args(self) -> List[str]:
        # -y: a file target is overwritten on reconnect instead of ffmpeg asking on stdin
        args = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
                "-f", "s16le", "-ar", str(self.mixer.sr), "-ac", "2", "-i", "pipe:0"]
        for t in self.targets:
            args += ["-c:a", t["codec"], "-b:a", t["bitrate"], "-f", t["format"]]
            url = t["url"]
            if url.startswith(("icecast://", "http://", "https://")):
                args += ["-content_type", t["content_type"]]
                if t.get("name"):
                    args += ["-ice_name", t["name"]]
                if t.get("description"):
                    args += ["-ice_description", t["description"]]
                if self.title:
                    args += ["-metadata", f"title={self.title}"]
            args.append(url)
        return args

    def _spawn(self) -> None:
        import subprocess
        with self._plock:
            try:
                self._proc = subprocess.Popen(self._args(), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.PIPE)
            except OSError as exc:
                self.last_error = f"ffmpeg: {exc}"
                self._proc = None
                self._retry_at = time.monotonic() + self._backoff
                self._backoff = min(30.0, self._backoff * 2)
                return
            self.connected = True
            threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()

    def _read_stderr(self, proc) -> None:
        try:
            for line in proc.stderr:
                s = line.decode("utf-8", errors="replace").strip()
                if s:
                    self._stderr_tail = (self._stderr_tail + [s])[-5:]
                    self.last_error = s
        except Exception:
            pass

    def start(self) -> None:
        self._spawn()
        super().start()

    def _on_block(self, block: np.ndarray) -> None:
        proc = self._proc
        if self._stop.is_set():
            return
        if proc is None or proc.poll() is not None:
            if proc is not None:
                try:
                    proc.stdin.close()      # otherwise the dead pipe complains at garbage collection
                except Exception:
                    pass
            if self.connected:
                self.connected = False
                self._retry_at = time.monotonic() + self._backoff
                self._backoff = min(30.0, self._backoff * 2)
            if time.monotonic() >= self._retry_at:
                self.restarts += 1
                self._spawn()
            return
        pcm = (np.clip(block[:2].T, -1, 1) * 32767).astype("<i2").tobytes()
        try:
            proc.stdin.write(pcm)
            self.bytes_out += len(pcm)
            self._backoff = 1.0
        except (BrokenPipeError, OSError) as exc:
            self.last_error = f"stream dropped: {exc}"
            self.connected = False
            try:
                proc.kill()
            except Exception:
                pass
            self._retry_at = time.monotonic() + self._backoff
            self._backoff = min(30.0, self._backoff * 2)

    def stop(self) -> None:
        super().stop()
        with self._plock:
            proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.connected = False

    def set_title(self, title: str) -> None:
        """Now-playing text for the next (re)connect; Icecast's live metadata
        update needs the admin password and is left to the relay (plan ER.6)."""
        self.title = title

    def status(self) -> Dict[str, Any]:
        return {"sink": self.name, "blocks": self.blocks, "late": self.late, "connected": self.connected,
                "restarts": self.restarts, "bytes_out": self.bytes_out, "last_error": self.last_error,
                "targets": [_mask(t["url"]) for t in self.targets], "title": self.title}
