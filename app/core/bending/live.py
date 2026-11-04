"""Live bending — pseudo-continuous playback of a bend while it is changed.

SA3 is latent diffusion: every render is a whole clip, so nothing streams
sample by sample. What does work is re-rendering fast and swapping smartly.

The material is a chain of chunks. Each chunk is rendered by continuing the
previous one through the inpaint path: the last CONTEXT seconds are the
source, the model continues them for CHUNK seconds (rendered PAST seconds
further so the chunk sits well before the model's wind-down, then trimmed).
The context is always taken from a CLEAN render of the chunk, so the bend is
a lens over a stable stream and never feeds back into itself: a heavy bend
cannot spiral the stream into noise, and taking it off brings the material
back.

The player either loops the current chunk or flows on:

  loop on   the chunk repeats. A change to the bend (or prompt, or dry/wet)
            re-renders the SAME chunk — same context, same noise — with the
            new bend, and the player crossfades to it at the same position.
            Delay = one render. `advance()` (the Next button) moves on to
            the following chunk once, which then loops.
  loop off  every request for a chunk advances: the sound evolves instead
            of repeating. Changes land in the next chunk.

Nothing here writes a fragment, signs anything or appends to the Bending
Log: this is a listening surface. One worker thread, one render at a time,
behind the generator's own lock, so a normal generation simply queues.
"""
from __future__ import annotations

import io
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("LiveBend")

SAMPLE_RATE = 44100
XFADE_SEC = 0.05            # seam between chunks, equal power
DEFAULT_CHUNK_SEC = 4.0
DEFAULT_CONTEXT_SEC = 4.0
DEFAULT_PAST_SEC = 8.0      # render past the chunk; the model winds down to the stated end
MIN_RENDER_GAP = 0.02
# Params whose change makes different material (the stream starts over).
_RESET_KEYS = ("model_id", "seed", "chunk_sec", "context_sec", "past_sec")


def _wav_bytes(audio: np.ndarray) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    pcm = np.clip(np.nan_to_num(audio, nan=0.0), -1.0, 1.0).T  # [N, C]
    sf.write(buf, pcm, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _xfade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(a.shape[-1], b.shape[-1])
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    return a[..., :n] * np.cos(t) + b[..., :n] * np.sin(t)


class LiveBend:
    """One live session per backend. `generator` is the AudioGenerator."""

    def __init__(self, generator: Any) -> None:
        self._gen = generator
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._dirty = False          # re-render the current chunk
        self._advance = False        # move on to the next chunk
        self._want = False           # the player asked for a chunk
        self._params: Dict[str, Any] = {}
        self._version = 0
        self._seq = 0
        self._chunk: Optional[bytes] = None
        self._chunk_meta: Dict[str, Any] = {}
        self._rendering = False
        self._last_render_ms: Optional[float] = None
        self._error: Optional[str] = None
        # The chain. `_context`/`_held` belong to the current chunk (what it
        # continues from, and the tail it crossfades onto); `_next_*` are what
        # the following chunk will use, taken from the current chunk's clean
        # render. Chunk ids are "<stream>.<index>" so the player can tell a
        # re-render (same id) from a new chunk.
        self._stream_id = 0
        self._index = 0
        self._context: Optional[np.ndarray] = None
        self._held: Optional[np.ndarray] = None
        self._next_context: Optional[np.ndarray] = None
        self._next_held: Optional[np.ndarray] = None

    # --- control -------------------------------------------------------------
    def start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._cv:
            self._params = dict(params)
            self._params.setdefault("loop", True)
            self._version += 1
            self._error = None
            self._seq = 0
            self._chunk = None
            self._reset_chain_locked()
            self._dirty = True
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._run, name="LiveBend", daemon=True)
                self._thread.start()
            self._cv.notify_all()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._cv:
            self._running = False
            self._want = self._dirty = self._advance = False
            self._cv.notify_all()
        # A render in flight is interrupted like any other generation.
        try:
            self._gen.request_stop()
        except Exception:
            pass
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=30)
        with self._cv:
            self._thread = None
            self._chunk = None
            self._reset_chain_locked()
        return self.status()

    def update(self, partial: Dict[str, Any]) -> Dict[str, Any]:
        """Merge a change in. Looping: the current chunk is re-rendered.
        Flowing: the next chunk picks it up. New material (seed, model,
        lengths) starts the chain over."""
        with self._cv:
            changed = {k for k, v in partial.items() if self._params.get(k) != v}
            self._params.update(partial)
            self._version += 1
            if changed & set(_RESET_KEYS):
                self._reset_chain_locked()
                self._dirty = True
            elif changed - {"loop"}:
                self._dirty = True
            self._cv.notify_all()
        return self.status()

    def advance(self) -> Dict[str, Any]:
        """Next: continue once into the following chunk."""
        with self._cv:
            self._advance = True
            self._cv.notify_all()
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "loop": bool(self._params.get("loop", True)),
                "seq": self._seq,
                "chunk": f"{self._stream_id}.{self._index}",
                "version": self._version,
                "rendering": self._rendering,
                "last_render_ms": self._last_render_ms,
                "error": self._error,
                "chunk_sec": float(self._params.get("chunk_sec") or DEFAULT_CHUNK_SEC),
                "context_sec": float(self._params.get("context_sec") or DEFAULT_CONTEXT_SEC),
            }

    def next_chunk(self, after: int, timeout: float = 25.0):
        """Block until a chunk newer than `after` exists (or timeout → None).
        When not looping, the request itself asks the worker to move on."""
        deadline = time.time() + timeout
        with self._cv:
            if self._seq <= after:
                self._want = True
                self._cv.notify_all()
            while (self._running and (self._seq <= after or self._chunk is None)
                   and self._error is None):
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            if self._seq > after and self._chunk is not None:
                return self._seq, self._chunk, dict(self._chunk_meta)
            return None

    # --- worker --------------------------------------------------------------
    def _reset_chain_locked(self) -> None:
        self._stream_id += 1
        self._index = 0
        self._context = self._held = None
        self._next_context = self._next_held = None

    def _run(self) -> None:
        logger.info("Live bend worker started")
        try:
            while True:
                with self._cv:
                    while self._running and not self._due_locked():
                        self._cv.wait(1.0)
                    if not self._running:
                        break
                    advance = self._advance or (
                        self._want and not self._params.get("loop", True)
                        and self._next_context is not None)
                    self._dirty = self._advance = self._want = False
                    if advance:
                        self._index += 1
                        self._context, self._held = self._next_context, self._next_held
                        self._next_context = self._next_held = None
                    params = dict(self._params)
                    version = self._version
                    chunk_id = f"{self._stream_id}.{self._index}"
                    self._rendering = True
                try:
                    t0 = time.time()
                    audio, warnings, nxt = self._render_chunk(params)
                    ms = (time.time() - t0) * 1000.0
                    wav = _wav_bytes(audio)
                    with self._cv:
                        if chunk_id == f"{self._stream_id}.{self._index}":
                            self._next_context, self._next_held = nxt
                        self._seq += 1
                        self._chunk = wav
                        self._chunk_meta = {
                            "chunk": chunk_id, "version": version, "render_ms": round(ms),
                            "seconds": audio.shape[-1] / SAMPLE_RATE,
                            "loop": bool(params.get("loop", True)),
                            "warnings": warnings or [],
                        }
                        self._last_render_ms = ms
                        self._rendering = False
                        self._cv.notify_all()
                    logger.info("Live render #%d (chunk %s): %.0f ms for %.1fs",
                                self._seq, chunk_id, ms, audio.shape[-1] / SAMPLE_RATE)
                except Exception as exc:  # noqa: BLE001
                    from app.core.generation.audio_generator import GenerationStopped
                    with self._cv:
                        self._rendering = False
                        if isinstance(exc, GenerationStopped):
                            # Stop pressed on a normal generation, or shutting
                            # down: render this chunk again when still live.
                            if self._running:
                                self._dirty = True
                        else:
                            logger.exception("Live render failed")
                            self._error = str(exc)
                            self._running = False
                        self._cv.notify_all()
                time.sleep(MIN_RENDER_GAP)
        finally:
            with self._cv:
                self._running = False
                self._rendering = False
                self._cv.notify_all()
            logger.info("Live bend worker stopped")

    def _due_locked(self) -> bool:
        if self._dirty or self._advance:
            return True
        # Flowing: a request for a chunk is a request to move on. Looping: a
        # request only waits for a re-render or Next.
        return self._want and not self._params.get("loop", True) and self._next_context is not None

    def _common_kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kw = dict(
            model_id=params["model_id"],
            batch_size=1,
            loras=params.get("loras") or [],
        )
        if params.get("steps"):
            kw["steps"] = int(params["steps"])
        # Every chunk gets its own noise, otherwise the continuation repeats
        # itself; the seed still makes the whole chain reproducible.
        kw["seed"] = (int(params.get("seed") or 0) + self._index * 7919) & 0x7fffffff
        return kw

    def _render_chunk(self, params: Dict[str, Any]):
        """Render the current chunk with the bend. Returns (audio to emit,
        warnings, (context, held) for the following chunk)."""
        chunk = float(params.get("chunk_sec") or DEFAULT_CHUNK_SEC)
        context = float(params.get("context_sec") or DEFAULT_CONTEXT_SEC)
        past = float(params.get("past_sec") or DEFAULT_PAST_SEC)
        sr = SAMPLE_RATE
        n_chunk, n_ctx, n_xf = int(chunk * sr), int(context * sr), int(XFADE_SEC * sr)
        patch = params.get("bend_patch") or None
        prompt = params.get("prompt") or ""
        kw = self._common_kwargs(params)

        def render(bend):
            if self._context is None:
                audio, report = self._gen.generate_array(
                    prompt, duration=chunk + XFADE_SEC + past, bend_patch=bend, **kw)
                start = 0
            else:
                duration = context + chunk + past
                audio, report = self._gen.generate_array(
                    prompt, duration=duration, bend_patch=bend,
                    inpaint_audio_data=(sr, self._context),
                    inpaint_starts=[context], inpaint_ends=[duration - 0.01],
                    extend_render_past=False, **kw)
                start = n_ctx
            return audio, report, start

        bent, report, start = render(patch)
        if self._held is None:
            # First chunk of the chain: nothing to crossfade onto.
            emit = bent[..., start:start + n_chunk]
            end = start + n_chunk + n_xf
        else:
            new = bent[..., start - n_xf:start + n_chunk]
            head = _xfade(self._held, new[..., :n_xf])
            emit = np.concatenate([head, new[..., n_xf:n_chunk]], axis=-1)
            end = start + n_chunk
        # The held tail: the XFADE seconds right after what is emitted, from
        # the render that is heard. The next chunk's head crossfades onto it.
        held = bent[..., end - n_xf:end]
        if held.shape[-1] < n_xf:
            held = np.pad(held, ((0, 0), (0, n_xf - held.shape[-1])))
        held = np.ascontiguousarray(held, dtype=np.float32)

        # What the next chunk continues from: the clean render's tail, so the
        # chain never feeds a bend back into itself. Without a bend the render
        # above is already clean. Decided once per chunk; a re-render of the
        # same chunk only refreshes the held tail.
        if self._next_context is None:
            if patch:
                clean, _r, _s = render(None)
            else:
                clean = bent
            ctx = clean[..., max(0, end - n_ctx):end]
            if ctx.shape[-1] < n_ctx:
                ctx = np.pad(ctx, ((0, 0), (n_ctx - ctx.shape[-1], 0)))
            nxt = (np.ascontiguousarray(ctx, dtype=np.float32), held)
        else:
            nxt = (self._next_context, held)
        return np.ascontiguousarray(emit, dtype=np.float32), report, nxt
