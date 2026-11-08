"""The conductor: what plays when, and what is rendered next.

Scenes are a blend plus parameter settings. All bed takes share one blend and
differ by seed (spec §3.2). Events are short clips from a per-type pool,
placed as point sources under an event budget, a minimum interval, a
sparseness roll and a spectral-occupancy check (§3.4). Transitions move the
new bed in from an azimuth while the old one recedes (§3.3).

Runs at control rate from the engine loop (`tick`). Never touches the GPU:
it submits jobs and receives clips.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.engine.anchors import AnchorBank
from app.engine.audio import EventVoice, Mixer, Trajectory
from app.engine.clips import Clip, ClipStore
from app.engine.params import ParameterStore
from app.engine.program import Program, ProgramStep
from app.engine.renderer import BaseRenderer, RenderJob

Emit = Callable[[str, Dict[str, Any]], None]


@dataclass
class Scene:
    id: str
    label: str = ""
    blend: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    modulators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # event type → {probability, fields (template overrides), trajectory, distance}
    events: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    soundmarks: List[Dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0      # for sequences; 0 = until changed

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label or self.id, "blend": self.blend, "params": self.params,
                "modulators": self.modulators, "events": self.events, "soundmarks": self.soundmarks,
                "duration_s": self.duration_s}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scene":
        ev = d.get("events") or {}
        if isinstance(ev, list):
            ev = {str(e["type"]): {k: v for k, v in e.items() if k != "type"} for e in ev}
        return cls(id=str(d["id"]), label=d.get("label", ""), blend={k: float(v) for k, v in (d.get("blend") or {}).items()},
                   params=dict(d.get("params") or {}), modulators=dict(d.get("modulators") or {}),
                   events={str(k): dict(v or {}) for k, v in ev.items()},
                   soundmarks=[dict(s) for s in (d.get("soundmarks") or [])], duration_s=float(d.get("duration_s", 0) or 0))


class Conductor:
    def __init__(self, params: ParameterStore, bank: AnchorBank, renderer: BaseRenderer, clips: ClipStore,
                 mixer: Mixer, emit: Emit, seed: int = 0) -> None:
        self.params, self.bank, self.renderer, self.clips, self.mixer, self.emit = params, bank, renderer, clips, mixer, emit
        self.scenes: Dict[str, Scene] = {}
        self.sequence: List[str] = []
        # the program: steps with lengths; replaces `sequence` when set (decision 11)
        self.program: Optional[Program] = None
        self.step_index: int = -1
        self._step_started: float = 0.0
        self.program_ended = False
        self.scene: Optional[Scene] = None
        self.playing = False
        self._rng = np.random.default_rng(seed)
        self._base_seed = int(seed) or int(time.time()) % 100000
        self._counter = 0
        self._lock = threading.RLock()
        # transition
        self._blend_from: Dict[str, float] = {}
        self._blend_to: Dict[str, float] = {}
        self._trans_t0 = 0.0
        self._trans_len = 0.0
        self._scene_started = 0.0
        self._generation = 0            # bumps per scene change; bed keys carry it
        self._next_onset = 0.0
        self._play_on_done: Dict[str, Dict[str, Any]] = {}
        self._event_counts: Dict[str, int] = {}
        self.events_fired = 0
        self.warnings: List[str] = []
        # clock mode: the scene follows a place's local time (the radio)
        self.clock_tz: str = ""
        self.clock_program: List[Dict[str, Any]] = []      # [{"at": "05:30", "scene": id}], sorted
        self._clock_scene: Optional[str] = None
        self.now_fn: Callable[[], float] = time.time
        renderer.on_done(self._job_done)
        mixer.on_voice_done = self._voice_done

    # --- scenes ---------------------------------------------------------------
    def load_scenes(self, scenes: List[Scene], sequence: Optional[List[str]] = None) -> None:
        with self._lock:
            self.scenes = {s.id: s for s in scenes}
            self.sequence = [s for s in (sequence or []) if s in self.scenes]

    def add_scene(self, scene: Scene) -> None:
        with self._lock:
            self.scenes[scene.id] = scene
            if self.scene is not None and self.scene.id == scene.id:   # an edit of the playing place
                self.scene = scene
                self.params.set_many(scene.params)
                self._sync_event_type_params(scene)
                self._blend_to = AnchorBank.normalise_weights(scene.blend)
                for aid in self.bank.ids():
                    p = f"/scene/blend/{aid}"
                    if self.params.exists(p):
                        self.params.set(p, self._blend_to.get(aid, 0.0), notify=False)

    def remove_scene(self, sid: str) -> bool:
        with self._lock:
            return self.scenes.pop(sid, None) is not None

    # --- program ----------------------------------------------------------------
    def set_program(self, program: Optional[Program], start_step: Optional[int] = 0, transition_s: Optional[float] = None) -> None:
        """Load a program; with `start_step` set, jump to that step now (its scene is activated)."""
        with self._lock:
            self.program = program
            self.program_ended = False
            self.step_index = -1
            if program and start_step is not None and program.steps:
                self.goto_step(start_step, transition_s=transition_s)

    def goto_step(self, index: int, transition_s: Optional[float] = None) -> Optional[ProgramStep]:
        with self._lock:
            if not self.program or not self.program.steps:
                return None
            index = index % len(self.program.steps)
            step = self.program.steps[index]
            if step.scene not in self.scenes:
                self.warnings.append(f"program step {index + 1}: scene '{step.scene}' missing")
                return None
            self.step_index = index
            self._step_started = self.now_fn()
            self.program_ended = False
            self.activate(step.scene, transition_s=step.transition_s if transition_s is None else transition_s,
                          from_azimuth=step.from_azimuth)
            if self.params.exists("/bed/source"):
                self.params.set("/bed/source", step.beds)
            if step.params:
                self.params.set_many(step.params)
            self.emit("program", {"id": self.program.id, "step": index, "scene": step.scene, "label": step.label,
                                  "seconds": step.seconds, "loop": self.program.loop})
            return step

    def next_step(self) -> Optional[ProgramStep]:
        with self._lock:
            if not self.program or not self.program.steps:
                return None
            nxt = self.step_index + 1
            if nxt >= len(self.program.steps):
                if not self.program.loop:
                    self.program_ended = True
                    self.emit("program", {"id": self.program.id, "ended": True})
                    return None
                nxt = 0
            return self.goto_step(nxt)

    def _tick_program(self) -> None:
        if not self.program or self.step_index < 0 or self.program_ended:
            return
        step = self.program.steps[self.step_index]
        if self.now_fn() - self._step_started >= step.seconds:
            self.next_step()

    def program_status(self) -> Optional[Dict[str, Any]]:
        if not self.program:
            return None
        step = self.program.steps[self.step_index] if 0 <= self.step_index < len(self.program.steps) else None
        elapsed = (self.now_fn() - self._step_started) if step else 0.0
        return {"id": self.program.id, "label": self.program.label or self.program.id, "loop": self.program.loop,
                "steps": len(self.program.steps), "step": self.step_index, "step_label": step.label if step else "",
                "step_scene": step.scene if step else None, "step_elapsed_s": round(elapsed, 1),
                "step_total_s": step.seconds if step else 0.0, "beds": step.beds if step else None,
                "ended": self.program_ended, "total_s": self.program.total_seconds}

    def activate(self, sid: str, transition_s: Optional[float] = None, from_azimuth: Optional[float] = None) -> Scene:
        with self._lock:
            if sid not in self.scenes:
                raise KeyError(sid)
            scene = self.scenes[sid]
            first = self.scene is None
            if transition_s is None:
                transition_s = float(self.params.get("/scene/transition_s"))
            if from_azimuth is None:
                from_azimuth = float(self.params.get("/scene/from_azimuth"))
            self._blend_from = dict(self.effective_blend())
            self._blend_to = AnchorBank.normalise_weights(scene.blend)
            self._trans_t0 = time.time()
            self._trans_len = 0.0 if first else max(0.0, float(transition_s))
            self.scene = scene
            self._scene_started = time.time()
            self._generation += 1
            self.params.set_many(scene.params)
            for path, mod in scene.modulators.items():
                if self.params.exists(path):
                    self.params.set_modulator(path, mod)
            for aid in self.bank.ids():
                p = f"/scene/blend/{aid}"
                if self.params.exists(p):
                    self.params.set(p, self._blend_to.get(aid, 0.0), notify=False)
            self._sync_event_type_params(scene)
            # the old beds recede; the new ones (created in tick) enter from the azimuth
            for key, b in list(self.mixer.beds.items()):
                if not key.startswith(f"g{self._generation}:"):
                    b.recede(max(0.5, self._trans_len or 1.0))
            self._enter_azimuth = float(from_azimuth)
            self._next_onset = time.time() + 1.0
        self.emit("scene", {"id": sid, "transition_s": self._trans_len, "from_azimuth": from_azimuth})
        return scene

    def _sync_event_type_params(self, scene: Scene) -> None:
        for path in self.params.paths("/events/types"):
            self.params.undefine(path)
        for t, spec in scene.events.items():
            self.params.define(f"/events/types/{t}/probability", float(spec.get("probability", 0.2)), (0.0, 1.0),
                               label=f"{t} probability")

    def set_clock(self, program: List[Dict[str, Any]], timezone: str = "UTC") -> None:
        rows = []
        for r in program or []:
            at = str(r.get("at", "00:00"))
            h, _, m = at.partition(":")
            rows.append({"at": at, "minute": int(h) * 60 + int(m or 0), "scene": str(r.get("scene", ""))})
        with self._lock:
            self.clock_program = sorted((r for r in rows if r["scene"] in self.scenes), key=lambda r: r["minute"])
            self.clock_tz = timezone

    def clock_now(self) -> Optional[Dict[str, Any]]:
        """Local time in the clock's zone and the scene the program names for it."""
        if not self.clock_program:
            return None
        from datetime import datetime, timezone as _tz
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(self.clock_tz or "UTC")
        except Exception:
            tz = _tz.utc
        local = datetime.fromtimestamp(self.now_fn(), tz)
        minute = local.hour * 60 + local.minute
        row = self.clock_program[-1]
        for r in self.clock_program:
            if r["minute"] <= minute:
                row = r
        return {"timezone": self.clock_tz, "local_time": local.strftime("%H:%M"), "scene": row["scene"], "since": row["at"]}

    def _tick_clock(self) -> None:
        if not self.params.exists("/scene/clock/on") or not self.params.get("/scene/clock/on"):
            return
        c = self.clock_now()
        if c is None or c["scene"] == self._clock_scene:
            return
        self._clock_scene = c["scene"]
        if self.scene is None or self.scene.id != c["scene"]:
            self.activate(c["scene"])

    def next_in_sequence(self) -> Optional[str]:
        if not self.sequence:
            return None
        if self.scene is None or self.scene.id not in self.sequence:
            return self.sequence[0]
        i = self.sequence.index(self.scene.id)
        return self.sequence[(i + 1) % len(self.sequence)]

    # --- blend ---------------------------------------------------------------
    def transition_progress(self) -> float:
        if self._trans_len <= 0:
            return 1.0
        return min(1.0, (time.time() - self._trans_t0) / self._trans_len)

    def effective_blend(self) -> Dict[str, float]:
        """The blend the next render uses: the live blend parameters, interpolated
        from the previous scene during a transition."""
        live = {aid: float(self.params.get(f"/scene/blend/{aid}")) for aid in self.bank.ids()
                if self.params.exists(f"/scene/blend/{aid}")}
        live = AnchorBank.normalise_weights(live) or dict(self._blend_to)
        p = self.transition_progress()
        if p >= 1.0 or not self._blend_from:
            return live
        keys = set(live) | set(self._blend_from)
        return AnchorBank.normalise_weights({k: self._blend_from.get(k, 0.0) * (1 - p) + live.get(k, 0.0) * p for k in keys})

    def _wander(self, dt: float) -> None:
        if not self.params.get("/scene/wander/on"):
            return
        rate = float(self.params.get("/scene/wander/rate"))
        depth = float(self.params.get("/scene/wander/depth"))
        ids = self.bank.ids()
        if len(ids) < 2:
            return
        cur = {aid: float(self.params.base(f"/scene/blend/{aid}")) for aid in ids if self.params.exists(f"/scene/blend/{aid}")}
        for aid in cur:
            cur[aid] = max(0.0, cur[aid] + float(self._rng.normal(0, depth * rate * dt)))
        w = AnchorBank.normalise_weights(cur)
        if not w:
            return
        for aid in cur:
            self.params.set(f"/scene/blend/{aid}", w.get(aid, 0.0), notify=False)

    # --- render requests -------------------------------------------------------
    def _seed(self) -> int:
        self._counter += 1
        spread = int(self.params.get("/bed/seed_spread"))
        return (self._base_seed + self._counter * max(1, spread)) % (2**31 - 1)

    def _render_params(self) -> Dict[str, Any]:
        mode = str(self.params.get("/render/seed_mode"))
        return {"cfg": float(self.params.get("/render/cfg")), "steps": int(self.params.get("/render/steps")),
                "seed": (int(self.params.get("/render/seed")) if mode == "fixed" else self._seed())}

    def _bed_key(self, take: int) -> str:
        return f"g{self._generation}:{take}"

    def _tick_beds(self) -> None:
        if self.scene is None:
            return
        takes = int(self.params.get("/bed/takes"))
        clip_len = float(self.params.get("/bed/clip_length"))
        xf = float(self.params.get("/bed/crossfade_s"))
        low_water = clip_len + xf
        for i in range(takes):
            key = self._bed_key(i)
            b = self.mixer.bed(key, create=False)
            if b is None:
                b = self.mixer.bed(key, create=True, crossfade_s=xf, seed=self._base_seed + i)
                b.spread = float(self.params.get("/spatial/bed/spread"))
                b.enter_from(getattr(self, "_enter_azimuth", 0.0) if self._generation > 1 else 0.0,
                             max(0.5, self._trans_len or 2.0))
            b.crossfade_s = xf
            b.spread = float(self.params.get("/spatial/bed/spread"))
            if b.ahead_seconds() < low_water and self.renderer.pending(role="bed", take=key) == 0:
                rp = self._render_params()
                # render past the stated length: SA3 winds down toward it
                self.renderer.submit(RenderJob(role="bed", blend=self.effective_blend(), overrides={"type": "ambience"},
                                               duration_s=min(118.0, clip_len + 4.0), take=key, scene=self.scene.id,
                                               prompt=self.bank.prompt_for(self.effective_blend()), **rp))
        # drop takes above the count
        for key in list(self.mixer.beds):
            if key.startswith(f"g{self._generation}:") and int(key.split(":")[1]) >= takes:
                self.mixer.beds[key].recede(2.0)

    def _event_pool(self, etype: str) -> List[Clip]:
        return self.clips.list(role="event", event_type=etype, scene=self.scene.id if self.scene else None)

    def _tick_event_pool(self) -> None:
        if self.scene is None:
            return
        pool_size = int(self.params.get("/events/pool_size"))
        batch = int(self.params.get("/events/render_batch"))
        for etype, spec in self.scene.events.items():
            prob = float(self.params.get(f"/events/types/{etype}/probability")) if self.params.exists(f"/events/types/{etype}/probability") else float(spec.get("probability", 0.2))
            if prob <= 0:
                continue
            have = len(self._event_pool(etype))
            pending = self.renderer.pending(role="event", event_type=etype, scene=self.scene.id)
            if have + pending * batch < pool_size:
                lo, hi = self._event_duration_range(spec)
                rp = self._render_params()
                overrides = {"type": "event", "perspective": "close", "sources": [etype], **(spec.get("fields") or {})}
                self.renderer.submit(RenderJob(role="event", blend=self.effective_blend(), overrides=overrides,
                                               duration_s=float(self._rng.uniform(lo, hi)), batch_size=batch,
                                               event_type=etype, scene=self.scene.id,
                                               prompt=self.bank.prompt_for(self.effective_blend()), **rp))

    def _event_duration_range(self, spec: Dict[str, Any]):
        r = spec.get("duration") or [float(self.params.get("/events/duration_min")), float(self.params.get("/events/duration_max"))]
        return float(r[0]), float(max(r[0], r[1]))

    def _tick_soundmarks(self, dt: float) -> None:
        if self.scene is None or not self.scene.soundmarks:
            return
        per_min = float(self.params.get("/soundmarks/probability"))
        if per_min <= 0:
            return
        ready = self.clips.list(role="soundmark", scene=self.scene.id)
        if not ready and self.renderer.pending(role="soundmark", scene=self.scene.id) == 0:
            sm = self.scene.soundmarks[int(self._rng.integers(0, len(self.scene.soundmarks)))]
            rp = self._render_params()
            self.renderer.submit(RenderJob(role="soundmark", blend=self.effective_blend(),
                                           overrides={"type": "event", **(sm.get("fields") or {})},
                                           duration_s=float(sm.get("duration_s", 6.0)), event_type=str(sm.get("id", "soundmark")),
                                           scene=self.scene.id, prompt=self.bank.prompt_for(self.effective_blend()), **rp))
            return
        active = [v for v in self.mixer.active_voices() if v.event_type.startswith("soundmark:")]
        if ready and not active and self._rng.random() < per_min * dt / 60.0:
            clip = ready[int(self._rng.integers(0, len(ready)))]
            self._play(clip, f"soundmark:{clip.event_type}", {"distance": [2.0, 6.0]})

    # --- scheduling -------------------------------------------------------------
    def _choose_type(self) -> Optional[str]:
        if self.scene is None:
            return None
        weights = {}
        for t in self.scene.events:
            p = f"/events/types/{t}/probability"
            w = float(self.params.get(p)) if self.params.exists(p) else float(self.scene.events[t].get("probability", 0.2))
            if w > 0 and self._event_pool(t):
                weights[t] = w
        if not weights:
            return None
        ts = list(weights)
        ws = np.array([weights[t] for t in ts], dtype=float)
        return ts[int(self._rng.choice(len(ts), p=ws / ws.sum()))]

    def _pick_clip(self, etype: str) -> Optional[Clip]:
        pool = self._event_pool(etype)
        if not pool:
            return None
        busy_bands = {v.band for v in self.mixer.active_voices()}
        free = [c for c in pool if c.analysis.get("band") not in busy_bands]
        if not free and busy_bands:
            return None          # spectral occupancy: wait rather than stack
        candidates = free or pool
        candidates.sort(key=lambda c: c.last_used)
        return candidates[0]

    def _play(self, clip: Clip, etype: str, spec: Dict[str, Any]) -> Optional[EventVoice]:
        audio = self.clips.load(clip.id)
        if audio is None:
            return None
        spread = float(self.params.get("/spatial/events/spread"))
        az = float(self._rng.uniform(-spread, spread))
        dr = spec.get("distance") or [float(self.params.get("/spatial/events/distance_min")), float(self.params.get("/spatial/events/distance_max"))]
        dist = float(self._rng.uniform(dr[0], max(dr[0], dr[1])))
        el = float(self._rng.uniform(-10, 25)) if spec.get("elevation", True) else 0.0
        traj_types = spec.get("trajectory") or self.params.get("/spatial/events/trajectories")
        if isinstance(traj_types, str):
            traj_types = [traj_types]
        traj_types = [t for t in (traj_types or ["static"]) if t in ("static", "passby", "orbit", "drift")] or ["static"]
        ttype = traj_types[int(self._rng.integers(0, len(traj_types)))]
        speed = {"static": 0.0, "passby": float(self._rng.uniform(3, 9)), "orbit": float(self._rng.uniform(8, 30)), "drift": 6.0}[ttype]
        traj = Trajectory(type=ttype, speed=speed, direction=int(self._rng.choice([-1, 1])))
        gain = 10 ** (float(self.params.get("/events/gain_db")) / 20)
        v = EventVoice(clip.id, audio, etype, azimuth=az, elevation=el, distance=dist, gain=gain,
                       trajectory=traj, band=str(clip.analysis.get("band", "mid")), seed=int(self._rng.integers(0, 2**31)))
        self.mixer.add_voice(v)
        self.clips.touch(clip.id)
        self.events_fired += 1
        self._event_counts[etype] = self._event_counts.get(etype, 0) + 1
        self.emit("event", {**v.to_dict(), "scene": self.scene.id if self.scene else ""})
        return v

    def _tick_events(self, now: float) -> None:
        if self.scene is None or now < self._next_onset:
            return
        density = float(self.params.get("/events/density"))
        min_int = float(self.params.get("/events/min_interval"))
        jitter = float(self.params.get("/events/jitter"))
        interval = (min_int + (1.0 - density) * float(self.params.get("/events/density_span"))) * float(self._rng.uniform(1 - jitter, 1 + jitter))
        self._next_onset = now + max(0.05, interval)
        if self._rng.random() < float(self.params.get("/events/sparseness")):
            return
        if len([v for v in self.mixer.active_voices() if not v.event_type.startswith("soundmark:")]) >= int(self.params.get("/events/max_concurrent")):
            return
        etype = self._choose_type()
        if etype is None:
            return
        clip = self._pick_clip(etype)
        if clip is None:
            return
        self._play(clip, etype, self.scene.events.get(etype, {}))

    def trigger(self, etype: str) -> Dict[str, Any]:
        """Fire one event now (OSC /fragmenta/event/trigger). Renders first if the pool is empty."""
        with self._lock:
            clip = self._pick_clip(etype) if self.scene else None
            if clip is None:
                pool = self._event_pool(etype) if self.scene else []
                clip = pool[0] if pool else None
            if clip is not None:
                v = self._play(clip, etype, (self.scene.events.get(etype, {}) if self.scene else {}))
                return {"played": v is not None, "clip_id": clip.id}
            if self.scene is None:
                return {"played": False, "reason": "no scene"}
            rp = self._render_params()
            spec = self.scene.events.get(etype, {})
            lo, hi = self._event_duration_range(spec)
            job = RenderJob(role="event", blend=self.effective_blend(),
                            overrides={"type": "event", "perspective": "close", "sources": [etype], **(spec.get("fields") or {})},
                            duration_s=float(self._rng.uniform(lo, hi)), event_type=etype, scene=self.scene.id,
                            prompt=self.bank.prompt_for(self.effective_blend()), **rp)
            self._play_on_done[job.id] = {"type": etype}
            self.renderer.submit(job)
            return {"played": False, "queued": job.id}

    # --- callbacks ------------------------------------------------------------------
    def _job_done(self, job: RenderJob, clips: List[Clip], err: Optional[str]) -> None:
        if err:
            self.warnings.append(f"{job.role} {job.id}: {err}")
            self.warnings = self.warnings[-20:]
            self.emit("warning", {"job": job.to_dict(), "error": err})
            return
        self.emit("clip", {"job": job.to_dict(), "clips": [c.id for c in clips]})
        if job.role == "bed":
            b = self.mixer.bed(job.take, create=False)
            if b is not None and clips:
                audio = self.clips.load(clips[0].id)
                if audio is not None:
                    b.push(clips[0].id, audio)
            return
        if job.id in self._play_on_done and clips:
            spec = self._play_on_done.pop(job.id)
            with self._lock:
                self._play(clips[0], spec["type"], self.scene.events.get(spec["type"], {}) if self.scene else {})

    def _voice_done(self, v: EventVoice) -> None:
        self.emit("event_end", {"clip_id": v.clip_id, "type": v.event_type})

    # --- the loop ---------------------------------------------------------------------
    def tick(self, dt: float) -> None:
        now = time.time()
        with self._lock:
            if self.scene is None:
                return
            self._wander(dt)
            if self.playing:
                self._tick_clock()
                self._tick_beds()
                self._tick_event_pool()
                self._tick_events(now)
                self._tick_soundmarks(dt)
                if self.program is not None:
                    self._tick_program()
                elif self.scene.duration_s > 0 and now - self._scene_started >= self.scene.duration_s:
                    nxt = self.next_in_sequence()
                    if nxt and nxt != self.scene.id:
                        self.activate(nxt)

    def start(self) -> None:
        with self._lock:
            self.playing = True
            self._next_onset = time.time() + 2.0
            if self.program and self.program.steps and self.step_index < 0:
                self.goto_step(0, transition_s=0.0)
            elif self.scene is None and self.scenes:
                c = self.clock_now() if (self.params.exists("/scene/clock/on") and self.params.get("/scene/clock/on")) else None
                self.activate(c["scene"] if c else (self.sequence[0] if self.sequence else next(iter(self.scenes))))

    def stop(self) -> None:
        with self._lock:
            self.playing = False
            for b in self.mixer.beds.values():
                b.recede(1.5)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            pools = {}
            if self.scene:
                for t in self.scene.events:
                    pools[t] = {"ready": len(self._event_pool(t)), "pending": self.renderer.pending(role="event", event_type=t, scene=self.scene.id)}
            return {
                "playing": self.playing, "scene": self.scene.id if self.scene else None,
                "blend": self.effective_blend() if self.bank.ids() else {}, "transition": round(self.transition_progress(), 3),
                "next_onset_in": round(max(0.0, self._next_onset - time.time()), 1), "pools": pools,
                "events_fired": self.events_fired, "event_counts": self._event_counts,
                "active_events": [v.to_dict() for v in self.mixer.active_voices()],
                "scene_elapsed_s": round(time.time() - self._scene_started, 1) if self.scene else 0.0,
                "sequence": self.sequence, "warnings": self.warnings[-5:],
                "clock": self.clock_now(), "program": self.program_status(),
            }
