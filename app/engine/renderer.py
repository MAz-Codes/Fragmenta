"""Render workers: a priority job queue in front of one loaded model.

`Renderer` owns a `TorchEngine` (load, LoRA stack) and renders through SA3's
low-level path with a blended anchor as the text conditioning
(perpetual/decisions/02). `FakeRenderer` has the same surface and makes
shaped noise, so everything above it runs without a GPU.

Priorities: bed refills first, then a one-shot someone is waiting for, then
the event pool, then soundmarks.
"""
from __future__ import annotations

import hashlib
import math
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.engine.anchors import AnchorBank, AnchorDef, Encoder, FakeEncoder, render_prompt
from app.engine.clips import Clip, ClipStore, analyse

PRIORITY = {"bed": 0, "oneshot": 1, "event": 2, "soundmark": 3}
SR = 44100
# Loudness targets per role (RMS dBFS after post-processing); E4.1 tunes them.
TARGETS_DB = {"bed": -22.0, "event": -16.0, "soundmark": -16.0, "oneshot": -14.0}


@dataclass
class RenderJob:
    role: str = "oneshot"
    blend: Dict[str, float] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)   # template fields for derived anchors
    prompt: str = ""                                           # used when there is no blend
    negative_prompt: str = ""
    seed: int = -1
    duration_s: float = 10.0
    cfg: float = 7.0
    steps: int = 50
    batch_size: int = 1
    init_clip: Optional[str] = None
    init_strength: float = 0.0
    event_type: str = ""
    scene: str = ""
    take: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created: float = field(default_factory=time.time)

    @property
    def priority(self) -> int:
        return PRIORITY.get(self.role, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "role": self.role, "blend": self.blend, "overrides": self.overrides,
                "prompt": self.prompt, "seed": self.seed, "duration_s": self.duration_s, "cfg": self.cfg,
                "steps": self.steps, "batch_size": self.batch_size, "event_type": self.event_type,
                "scene": self.scene, "take": self.take, "created": self.created, "init_clip": self.init_clip}


JobDone = Callable[[RenderJob, List[Clip], Optional[str]], None]


# --- post-processing ---------------------------------------------------------
def _rms_db(a: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(a * a) + 1e-12)))


def _normalise(a: np.ndarray, target_db: float, peak_db: float = -1.0) -> np.ndarray:
    g = 10 ** ((target_db - _rms_db(a)) / 20)
    peak = float(np.max(np.abs(a)) + 1e-9)
    g = min(g, 10 ** (peak_db / 20) / peak)
    return (a * g).astype(np.float32)


def post_process(audio: np.ndarray, sr: int, role: str, targets: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Per-role shaping (spec §3.6). Events: mono, high-pass, trimmed, faded, normalised."""
    targets = {**TARGETS_DB, **(targets or {})}
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if role == "bed":
        return _normalise(_trim_bed(a, sr), targets["bed"])
    if role == "oneshot":
        peak = float(np.max(np.abs(a)) + 1e-9)
        return (a * min(1.0, 10 ** (-1 / 20) / peak)).astype(np.float32)
    # event / soundmark
    from scipy.signal import butter, sosfilt
    mono = a.mean(axis=0)
    sos = butter(2, 80.0, btype="highpass", fs=sr, output="sos")
    mono = sosfilt(sos, mono).astype(np.float32)
    # trim to the region above -45 dB relative to the peak, with a short pre-roll
    peak = float(np.max(np.abs(mono)) + 1e-9)
    thr = peak * 10 ** (-45 / 20)
    idx = np.nonzero(np.abs(mono) > thr)[0]
    if idx.size:
        start = max(0, int(idx[0]) - int(0.05 * sr))
        end = min(mono.shape[0], int(idx[-1]) + int(0.15 * sr))
        mono = mono[start:end]
    fi, fo = int(0.01 * sr), int(0.06 * sr)
    if mono.shape[0] > fi + fo:
        mono[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)
        mono[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    return _normalise(mono[None, :], targets.get(role, targets["event"]))


def _trim_bed(a: np.ndarray, sr: int, head_s: float = 0.5, frame_s: float = 0.5, drop_db: float = 6.0, keep_min: float = 0.7) -> np.ndarray:
    """SA3 winds a clip down toward its stated length, so the last seconds of a
    bed decay toward silence; crossfading over that decay is the audible seam.
    Cut the tail back to the last frame still within `drop_db` of the clip's
    median level (never below `keep_min` of the clip), and skip a soft head."""
    n = a.shape[1]
    hop = max(1, int(frame_s * sr))
    frames = n // hop
    if frames < 8:
        return a
    mono = a.mean(axis=0)[: frames * hop].reshape(frames, hop)
    rms_db = 20 * np.log10(np.sqrt(np.mean(mono ** 2, axis=1)) + 1e-9)
    mid = rms_db[frames // 4: max(frames // 4 + 1, 3 * frames // 4)]
    ref = float(np.median(mid)) if mid.size else float(np.median(rms_db))
    last = frames - 1
    while last > int(frames * keep_min) and rms_db[last] < ref - drop_db:
        last -= 1
    end = min(n, (last + 1) * hop)
    start = min(int(head_s * sr), max(0, end - int(sr * 2)))
    return np.ascontiguousarray(a[:, start:end])


# --- the queue and the base worker -----------------------------------------
def _is_fatal_device_error(exc: BaseException) -> bool:
    """A CUDA error that poisons the context (launch timeout, illegal address, device lost).
    Out-of-memory is not fatal: `_render` already retries smaller."""
    name = exc.__class__.__name__
    msg = str(exc)
    if "OutOfMemoryError" in name or "out of memory" in msg.lower():
        return False
    return name in ("AcceleratorError",) or "CUDA error" in msg or "device-side assert" in msg


class BaseRenderer:
    name = "base"

    def __init__(self, clips: ClipStore, bank: Optional[AnchorBank] = None) -> None:
        self.clips = clips
        self.bank = bank
        self._q: "queue.PriorityQueue[Tuple[int, float, str]]" = queue.PriorityQueue()
        self._jobs: Dict[str, RenderJob] = {}
        self._cancelled: set = set()
        self._done_cbs: List[JobDone] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.fatal: Optional[str] = None       # set once by a dead-context error; the engine restarts the process
        self._lock = threading.RLock()
        self.current: Optional[RenderJob] = None
        self.progress = 0.0
        self.renders = 0
        self.errors = 0
        self.last_error = ""
        self.last_render_s = 0.0
        self.render_times: List[float] = []
        self.model_id = ""
        self.loras: List[Dict[str, Any]] = []
        self.loaded = False
        self.targets_db: Dict[str, float] = {}
        self.alive_at = time.time()
        self.encoder: Optional[Encoder] = None

    # --- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="renderer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def on_done(self, cb: JobDone) -> None:
        self._done_cbs.append(cb)

    # --- jobs ----------------------------------------------------------------
    def submit(self, job: RenderJob) -> str:
        with self._lock:
            self._jobs[job.id] = job
        self._q.put((job.priority, job.created, job.id))
        return job.id

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs:
                self._cancelled.add(job_id)
                return True
        return False

    def pending(self, role: Optional[str] = None, event_type: Optional[str] = None,
                take: Optional[str] = None, scene: Optional[str] = None) -> int:
        with self._lock:
            n = 0
            for j in self._jobs.values():
                if j.id in self._cancelled:
                    continue
                if role and j.role != role:
                    continue
                if event_type and j.event_type != event_type:
                    continue
                if take and j.take != take:
                    continue
                if scene and j.scene != scene:
                    continue
                n += 1
            return n

    def queue_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: (j.priority, j.created))
                    if j.id not in self._cancelled]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n = len([j for j in self._jobs if j not in self._cancelled])
            avg = (sum(self.render_times[-10:]) / len(self.render_times[-10:])) if self.render_times else 0.0
            return {
                "renderer": self.name, "running": self.running, "queued": n,
                "current": self.current.to_dict() if self.current else None, "progress": round(self.progress, 3),
                "renders": self.renders, "errors": self.errors, "last_error": self.last_error,
                "last_render_s": round(self.last_render_s, 2), "avg_render_s": round(avg, 2),
                "model_id": self.model_id, "loras": self.loras, "loaded": self.loaded,
                "alive_s": round(time.time() - self.alive_at, 1), **self.device_stats(),
            }

    def device_stats(self) -> Dict[str, Any]:
        return {}

    # --- model (overridden) --------------------------------------------------
    def load(self, model_id: str, loras: Optional[List[Dict[str, Any]]] = None) -> None:
        self.model_id = model_id
        self.loras = list(loras or [])
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def set_lora_strength(self, index: int, strength: float) -> bool:
        if 0 <= index < len(self.loras):
            self.loras[index]["strength"] = float(strength)
            return True
        return False

    def encoder_key(self) -> str:
        lk = ";".join(f"{Path(l['path']).name}:{l.get('strength', 1.0)}" for l in self.loras)
        return f"{self.name}|{self.model_id}|{lk}"

    def ensure_anchors(self, force: bool = False) -> Dict[str, bool]:
        """Encode the bank's anchors with this renderer's encoder (on the caller's thread)."""
        if self.bank is None or self.encoder is None:
            return {}
        return self.bank.encode_all(self.encoder, force=force)

    # --- the loop ------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            self.alive_at = time.time()
            try:
                _prio, _t, jid = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            with self._lock:
                job = self._jobs.get(jid)
                if job is None or jid in self._cancelled:
                    self._jobs.pop(jid, None)
                    self._cancelled.discard(jid)
                    continue
                self.current = job
                self.progress = 0.0
            clips: List[Clip] = []
            err: Optional[str] = None
            t0 = time.perf_counter()
            try:
                outs, sr, prompt = self._render(job)
                for i, audio in enumerate(outs):
                    shaped = post_process(audio, sr, job.role, self.targets_db)
                    clips.append(self.clips.add(
                        shaped, sr, job.role, event_type=job.event_type, scene=job.scene,
                        blend=dict(job.blend), seed=int(job.seed) + i, prompt=prompt,
                        render_s=round(time.perf_counter() - t0, 2)))
                self.renders += 1
            except Exception as exc:  # the loop must survive any job …
                if self._stop.is_set():
                    break
                self.errors += 1
                err = f"{exc.__class__.__name__}: {exc}"
                self.last_error = err
                traceback.print_exc()
                if _is_fatal_device_error(exc):
                    # … except a dead CUDA context: nothing rendered in this process will ever
                    # succeed again, so stop taking jobs and let the engine restart the process
                    self.fatal = err
                    with self._lock:
                        self.current = None
                        self._jobs.pop(jid, None)
                    for cb in list(self._done_cbs):
                        try:
                            cb(job, [], err)
                        except Exception:
                            traceback.print_exc()
                    break
            self.last_render_s = time.perf_counter() - t0
            self.render_times.append(self.last_render_s)
            self.render_times = self.render_times[-100:]
            with self._lock:
                self.current = None
                self.progress = 0.0
                self._jobs.pop(jid, None)
            for cb in list(self._done_cbs):
                try:
                    cb(job, clips, err)
                except Exception:
                    traceback.print_exc()

    def _render(self, job: RenderJob) -> Tuple[List[np.ndarray], int, str]:
        raise NotImplementedError

    # --- anchors for a job ---------------------------------------------------
    def _blend_for(self, job: RenderJob, rescale: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Resolve the job's blend (+ overrides → derived anchors) to tensors."""
        if self.bank is None or not job.blend or self.encoder is None:
            return None, None, job.prompt
        weights = AnchorBank.normalise_weights(job.blend)
        if not weights:
            return None, None, job.prompt
        if job.overrides:
            ids = {}
            for aid in weights:
                if aid not in self.bank:
                    continue
                base = self.bank.get(aid).definition
                d = AnchorDef(id=f"{aid}|{_digest(job.overrides)}", label=base.label,
                              fields={**base.fields, **job.overrides},
                              variants=[{**v, **job.overrides} for v in base.variants])
                if d.id not in self.bank:
                    self.bank.add(d)
                ids[d.id] = weights[aid]
            missing = [i for i in ids if not self.bank.get(i).encoded]
            if missing:
                self.bank.encode_all(self.encoder)
            weights = ids
        emb, mask, _ = self.bank.blend(weights, rescale=rescale)
        top = max(weights, key=weights.get)
        return emb, mask, render_prompt(self.bank.get(top).definition.fields)


def _digest(d: Dict[str, Any]) -> str:
    import json
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


# --- fake ---------------------------------------------------------------------
class FakeRenderer(BaseRenderer):
    """Shaped noise, instantly or after `render_time` seconds. Tests and dry runs."""

    name = "fake"

    def __init__(self, clips: ClipStore, bank: Optional[AnchorBank] = None, render_time: float = 0.0,
                 sr: int = SR) -> None:
        super().__init__(clips, bank)
        self.render_time = render_time
        self.sr = sr
        self.encoder = FakeEncoder(key="fake")

    def load(self, model_id: str, loras: Optional[List[Dict[str, Any]]] = None) -> None:
        super().load(model_id, loras)
        self.encoder = FakeEncoder(key=self.encoder_key())
        self.ensure_anchors()

    def _render(self, job: RenderJob) -> Tuple[List[np.ndarray], int, str]:
        if self.render_time > 0:
            steps = max(1, int(self.render_time / 0.05))
            for i in range(steps):
                if self._stop.is_set():
                    raise RuntimeError("stopped")
                time.sleep(self.render_time / steps)
                self.progress = (i + 1) / steps
        _emb, _mask, prompt = self._blend_for(job)
        seed = int(job.seed) if job.seed >= 0 else int(time.time() * 1000) % 2**31
        n = int(job.duration_s * self.sr)
        outs = []
        for b in range(job.batch_size):
            rng = np.random.default_rng(seed + b)
            noise = rng.normal(0, 0.1, (2, n)).astype(np.float32)
            # colour by the dominant anchor name so different blends are tellable
            key = max(job.blend, key=job.blend.get) if job.blend else (job.prompt or "x")
            h = int(hashlib.sha256(key.encode()).hexdigest()[:4], 16)
            from scipy.signal import butter, sosfilt
            lo = 100 + (h % 7) * 300
            sos = butter(2, [lo, lo * 4], btype="bandpass", fs=self.sr, output="sos")
            out = sosfilt(sos, noise, axis=1).astype(np.float32)
            if job.role in ("event", "soundmark"):
                env = np.exp(-np.linspace(0, 6, n, dtype=np.float32))
                out *= env
            outs.append(out)
        return outs, self.sr, prompt


# --- torch ---------------------------------------------------------------------
class TorchEncoder:
    """The loaded model's text conditioner as an `Encoder`."""

    def __init__(self, model: Any, device: str, key: str) -> None:
        self.model, self.device, self.key = model, device, key

    def encode(self, prompts):
        import torch
        cond = self.model.model.conditioner.conditioners["prompt"]
        embs, masks = [], []
        with torch.no_grad():
            for i in range(0, len(prompts), 8):
                e, m = cond(list(prompts[i:i + 8]), self.device)
                embs.append(e.detach().float().cpu().numpy())
                masks.append(m.detach().bool().cpu().numpy())
        return np.concatenate(embs, axis=0).astype(np.float32), np.concatenate(masks, axis=0)


class Renderer(BaseRenderer):
    name = "torch"

    def __init__(self, clips: ClipStore, bank: Optional[AnchorBank] = None, config: Any = None,
                 half: bool = True) -> None:
        super().__init__(clips, bank)
        self.config = config
        self.half = half
        self._engine = None
        self.blend_rescale = True

    def _ensure_engine(self):
        if self._engine is None:
            from app.core.config import get_config
            from app.core.generation.torch_engine import TorchEngine
            self._engine = TorchEngine(self.config or get_config())
        return self._engine

    def load(self, model_id: str, loras: Optional[List[Dict[str, Any]]] = None) -> None:
        eng = self._ensure_engine()
        eng.load(model_id, half=self.half)
        eng.apply_loras(list(loras or []))
        self.model_id, self.loras, self.loaded = model_id, list(loras or []), True
        self.encoder = TorchEncoder(eng.model, eng.device, self.encoder_key())
        self.ensure_anchors()

    def unload(self) -> None:
        if self._engine is not None:
            self._engine.unload()
        self.loaded = False
        self.encoder = None

    def set_lora_strength(self, index: int, strength: float) -> bool:
        ok = self._engine.set_lora_strength(index, strength) if self._engine else False
        if ok:
            self.loras[index]["strength"] = float(strength)
            self.encoder = TorchEncoder(self._engine.model, self._engine.device, self.encoder_key())
        return ok

    def device_stats(self) -> Dict[str, Any]:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return {"device": "cuda", "gpu_used_gb": round((total - free) / 1e9, 2), "gpu_total_gb": round(total / 1e9, 2)}
        except Exception:
            pass
        return {"device": getattr(self._engine, "device", "") or ""}

    def _render(self, job: RenderJob) -> Tuple[List[np.ndarray], int, str]:
        import torch
        eng = self._ensure_engine()
        if not self.loaded:
            raise RuntimeError("no model loaded")
        model = eng.model
        emb, mask, prompt = self._blend_for(job, rescale=self.blend_rescale)
        seed = int(job.seed) if job.seed >= 0 else int(np.random.randint(0, 2**31 - 1))
        steps, batch = int(job.steps), int(job.batch_size)
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._render_once(model, eng.device, job, emb, mask, prompt or job.prompt, seed, steps, batch), SR, prompt or job.prompt
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempts >= 3:
                    raise
                if batch > 1:
                    batch = max(1, batch // 2)
                else:
                    steps = max(8, int(steps * 0.8))

    def _render_once(self, model, device, job, emb, mask, prompt, seed, steps, batch) -> List[np.ndarray]:
        import torch
        dur = float(job.duration_s)
        conditioning = [{"prompt": prompt, "seconds_total": dur} for _ in range(batch)]
        negative = ([{"prompt": job.negative_prompt, "seconds_total": dur} for _ in range(batch)]
                    if job.negative_prompt else None)
        tensors = None
        if emb is not None:
            # every other conditioner (seconds_total) as the model computes it;
            # only the text embedding is replaced by the blend
            tensors = model.model.conditioner(conditioning, device)
            e = torch.from_numpy(emb).to(device).unsqueeze(0).expand(batch, -1, -1).contiguous()
            m = torch.from_numpy(mask).to(device).unsqueeze(0).expand(batch, -1).contiguous()
            tensors["prompt"] = [e, m]
        try:
            native = int(model.model_config.get("sample_size"))
        except Exception:
            native = 0
        ceiling = max(native, int(round(dur * SR)))
        kwargs: Dict[str, Any] = dict(
            conditioning=conditioning, conditioning_tensors=tensors, negative_conditioning=negative,
            duration=dur, sample_size=ceiling, steps=steps, cfg_scale=float(job.cfg), seed=seed,
            batch_size=batch, callback=self._on_step(steps),
        )
        if job.init_clip:
            audio = self.clips.load(job.init_clip)
            if audio is not None:
                a = audio if audio.shape[0] == 2 else np.concatenate([audio, audio], axis=0)
                kwargs["init_audio"] = (SR, torch.from_numpy(a))
                kwargs["init_noise_level"] = float(max(0.0, min(1.0, 1.0 - job.init_strength)))
        out = model.generate(**kwargs)
        if out.ndim != 3:
            raise RuntimeError(f"unexpected output shape {tuple(out.shape)}")
        return [out[i].detach().float().cpu().numpy() for i in range(out.shape[0])]

    def _on_step(self, steps: int):
        def cb(info: Dict[str, Any]) -> None:
            if self._stop.is_set():
                raise RuntimeError("renderer stopping")
            i = info.get("i", 0)
            self.progress = min(1.0, (i + 1) / max(1, steps))
        return cb


def make_renderer(kind: str, clips: ClipStore, bank: Optional[AnchorBank], config: Any = None,
                  render_time: float = 0.0) -> BaseRenderer:
    if kind == "fake":
        return FakeRenderer(clips, bank, render_time=render_time)
    return Renderer(clips, bank, config=config)
