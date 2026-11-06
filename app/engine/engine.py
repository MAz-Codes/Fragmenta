"""The engine: wires parameters, anchors, renderer, clips, audio and conductor,
runs the control loop, snapshots state, speaks OSC, and feeds the event hub
the API streams from. One instance per process.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from app.engine import __version__
from app.engine.anchors import AnchorBank, AnchorDef
from app.engine.audio import Mixer, make_sink
from app.engine.clips import ClipStore
from app.engine.conductor import Conductor, Scene
from app.engine.osc import OscClient, OscServer
from app.engine.params import ParameterStore
from app.engine.renderer import RenderJob, make_renderer

PRESETS_DIR = Path(__file__).resolve().parent / "presets"
CONTROL_HZ = 25.0
SNAPSHOT_EVERY_S = 5.0
METERS_EVERY_S = 0.5
OSC_PREFIX = "/fragmenta"


class EventHub:
    """Fan-out of engine events to SSE subscribers (bounded queues, drop-oldest)."""

    def __init__(self) -> None:
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()
        self.count = 0

    def subscribe(self, maxsize: int = 256) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, kind: str, data: Dict[str, Any]) -> None:
        item = (kind, data, time.time())
        self.count += 1
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(item)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(item)
                except (queue.Empty, queue.Full):
                    pass

    @property
    def subscribers(self) -> int:
        return len(self._subs)


def _ensure_presets(user_dir: Path) -> None:
    """Shipped scenes / layouts / anchors are copied on first run, never overwritten."""
    for sub in ("scenes", "layouts", "anchors"):
        src, dst = PRESETS_DIR / sub, user_dir / sub
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            for f in src.glob("*.yaml"):
                if not (dst / f.name).exists():
                    shutil.copy2(f, dst / f.name)


def _resolve(user_dir: Path, sub: str, name: Optional[str]) -> Optional[Path]:
    """`places` → <user>/engine/<sub>/places.yaml, or a path as given."""
    if not name:
        return None
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    cand = user_dir / sub / f"{p.stem}.yaml"
    if cand.exists():
        return cand
    cand = PRESETS_DIR / sub / f"{p.stem}.yaml"
    return cand if cand.exists() else None


class Engine:
    def __init__(self, data_dir: Path, scene: Optional[str] = "places", layout: Optional[str] = "stereo",
                 anchors: Optional[str] = None, renderer: str = "torch", sink: str = "auto",
                 model_id: str = "sa3-small-sfx-base", loras: Optional[List[Dict[str, Any]]] = None,
                 osc_port: int = 9000, osc_out: Optional[str] = "127.0.0.1:9001", seed: int = 0,
                 fake_render_time: float = 0.0, clip_cap_gb: float = 4.0, config: Any = None,
                 autoload: bool = True, resume: bool = True, radio: Optional[List[Any]] = None,
                 clock: Optional[bool] = None, device: Optional[Any] = None, channels: Optional[int] = None,
                 routing: Optional[Dict[str, Any]] = None) -> None:
        self.root = Path(data_dir) / "engine"
        for sub in ("clips", "anchors_cache", "presets", "recordings"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        _ensure_presets(self.root)
        self.config = config
        self.started_at = time.time()
        self.events = EventHub()
        self.params = ParameterStore()
        self.notes: List[str] = []
        self._define_params()

        # anchors
        anchors_path = _resolve(self.root, "anchors", anchors) or _resolve(self.root, "anchors", scene) or _resolve(self.root, "anchors", "places")
        self.anchors_path = anchors_path
        self.bank = AnchorBank.from_yaml(anchors_path, self.root / "anchors_cache") if anchors_path else AnchorBank([], self.root / "anchors_cache")
        for aid in self.bank.ids():
            self._define_blend_param(aid)

        self.clips = ClipStore(self.root / "clips", cap_bytes=int(clip_cap_gb * 1024 ** 3))
        self.renderer_kind = renderer
        self.renderer = make_renderer(renderer, self.clips, self.bank, config=config, render_time=fake_render_time)
        self.state_path = self.root / "state.json"
        self.layout: Dict[str, Any] = self._load_layout(layout)
        # output: what was saved last time, then what the command line says
        self.output: Dict[str, Any] = {"sink": sink, "device": None, "channels": int(self.layout.get("channels") or 2),
                                       "layout": self.layout.get("id", "stereo"), "routing": {}, "spatial_channels": [1, 2]}
        saved = self._read_state() if resume else {}
        if isinstance(saved.get("output"), dict):
            self.output.update({k: v for k, v in saved["output"].items() if k in self.output})
            if self.output.get("layout") != self.layout.get("id"):
                self.layout = self._load_layout(self.output.get("layout"))
        if sink != "auto":
            self.output["sink"] = sink
        if device is not None:
            self.output["device"] = device
        if channels:
            self.output["channels"] = int(channels)
        if routing is not None:
            self.output["routing"] = dict(routing)
        self.mixer = Mixer(channels=self.output["channels"])
        try:
            self.mixer.set_routing(self.output["routing"], self.output.get("spatial_channels"))
        except ValueError as exc:
            self.notes.append(f"routing reset: {exc}")
            self.output["routing"] = {}
            self.mixer.set_routing({})
        self.sink_kind = self.output["sink"]
        self.radio_targets = list(radio or [])
        self.sink, note = make_sink(self.output["sink"], self.mixer, path=self.root / "recordings" / "engine_output.wav",
                                    device=self.output.get("device"), targets=self.radio_targets)
        if note:
            self.notes.append(note)
        self.conductor = Conductor(self.params, self.bank, self.renderer, self.clips, self.mixer, self.emit, seed=seed)
        self.scene_path = _resolve(self.root, "scenes", scene)
        if self.scene_path:
            self._load_scenes(self.scene_path)
        if clock is not None:
            self.params.set("/scene/clock/on", bool(clock), notify=False)

        # model
        self.model_id = model_id
        self.loras = list(loras or [])
        self.model_state = "idle"
        self.model_error = ""
        self._loader: Optional[threading.Thread] = None

        # control loop
        self._stop = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None
        self._renderer_restarts = 0
        self.oneshots: Dict[str, Dict[str, Any]] = {}
        self.renderer.on_done(self._oneshot_done)

        # OSC
        self.osc_server: Optional[OscServer] = None
        self.osc_client: Optional[OscClient] = None
        if osc_port and int(osc_port) > 0:
            self.osc_server = OscServer("0.0.0.0", int(osc_port), self._osc_in)
        if osc_out:
            host, _, port = str(osc_out).rpartition(":")
            self.osc_client = OscClient(host or "127.0.0.1", int(port or 9001))

        self.params.subscribe(self._param_changed)
        if resume:
            self.restore()
        if autoload and model_id:
            self.load_model(model_id, self.loras)

    # --- parameters ------------------------------------------------------------
    def _define_params(self) -> None:
        P = self.params.define
        P("/scene/wander/on", False, label="Wander")
        P("/scene/wander/rate", 0.05, (0.0, 1.0), label="Wander rate")
        P("/scene/wander/depth", 0.2, (0.0, 1.0), label="Wander depth")
        P("/scene/clock/on", False, label="Follow the clock")
        P("/scene/transition_s", 20.0, (0.0, 120.0), label="Transition", unit="s")
        P("/scene/from_azimuth", 0.0, (-180.0, 180.0), label="Arrive from", unit="°")
        P("/bed/takes", 2, (1, 4), label="Bed takes")
        P("/bed/clip_length", 60.0, (10.0, 114.0), label="Bed clip length", unit="s")
        P("/bed/crossfade_s", 8.0, (1.0, 30.0), label="Bed crossfade", unit="s")
        P("/bed/seed_spread", 7, (1, 1000), label="Seed spread")
        P("/events/density", 0.4, (0.0, 1.0), curve="exp", smoothing_s=2.0, label="Density")
        P("/events/sparseness", 0.3, (0.0, 1.0), smoothing_s=2.0, label="Silence")
        P("/events/max_concurrent", 3, (1, 8), label="Max concurrent events")
        P("/events/min_interval", 2.0, (0.2, 30.0), label="Min interval", unit="s")
        P("/events/density_span", 20.0, (0.0, 90.0), label="Interval at density 0", unit="s")
        P("/events/jitter", 0.3, (0.0, 1.0), label="Interval jitter")
        P("/events/pool_size", 3, (1, 12), label="Ready clips per type")
        P("/events/render_batch", 2, (1, 4), label="Event render batch")
        P("/events/duration_min", 1.5, (0.5, 8.0), label="Event min length", unit="s")
        P("/events/duration_max", 6.0, (0.5, 10.0), label="Event max length", unit="s")
        P("/events/gain_db", 0.0, (-24.0, 12.0), smoothing_s=0.5, label="Events gain", unit="dB")
        P("/soundmarks/probability", 0.5, (0.0, 10.0), label="Soundmarks per minute")
        P("/render/cfg", 7.0, (1.0, 15.0), label="CFG")
        P("/render/steps", 50, (4, 100), label="Steps")
        P("/render/lora_strength", 0.85, (0.0, 1.5), label="LoRA strength")
        P("/render/init_strength", 0.0, (0.0, 1.0), label="Init strength")
        P("/render/seed_mode", "spread", kind="str", label="Seed mode")   # spread | fixed
        P("/render/seed", 0, (0, 2**31 - 1), label="Seed")
        P("/render/blend_rescale", True, label="Rescale blends")
        P("/spatial/bed/spread", 1.0, (0.0, 1.0), smoothing_s=1.0, label="Bed width")
        P("/spatial/events/spread", 180.0, (0.0, 180.0), label="Events spread", unit="°")
        P("/spatial/events/distance_min", 1.0, (0.5, 20.0), label="Nearest", unit="m")
        P("/spatial/events/distance_max", 6.0, (0.5, 30.0), label="Farthest", unit="m")
        P("/spatial/events/trajectories", ["static", "passby"], kind="list", label="Trajectories")
        P("/master/gain", 0.8, (0.0, 1.5), smoothing_s=0.2, label="Master")
        P("/master/limiter", True, label="Limiter")

    def _define_blend_param(self, aid: str) -> None:
        self.params.define(f"/scene/blend/{aid}", 0.0, (0.0, 1.0), label=f"Blend {aid}")

    def _param_changed(self, path: str, value: Any) -> None:
        if path == "/render/lora_strength" and self.renderer.loras:
            self.renderer.set_lora_strength(0, float(value))
        elif path == "/render/blend_rescale":
            setattr(self.renderer, "blend_rescale", bool(value))
        self.events.publish("param", {"path": path, "value": value})

    # --- scenes / layout -------------------------------------------------------
    def _load_scenes(self, path: Path, _depth: int = 0) -> None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        scenes = []
        # `include: places` pulls another file's scenes in first (the radio reuses the installation's)
        inc = data.get("include")
        if inc and _depth < 3:
            p = _resolve(self.root, "scenes", inc)
            if p and p != Path(path):
                idata = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                scenes += [Scene.from_dict(s) for s in idata.get("scenes", [])]
        scenes += [Scene.from_dict(s) for s in data.get("scenes", [])]
        self.conductor.load_scenes(scenes, data.get("sequence"))
        clock = data.get("clock") or {}
        if clock.get("program"):
            self.conductor.set_clock(clock["program"], clock.get("timezone", "UTC"))
            if clock.get("enabled", clock.get(True)) and self.params.base("/scene/clock/on") is False:
                self.params.set("/scene/clock/on", True, notify=False)
        # scene-file anchors (optional) add to the bank
        for d in data.get("anchors", []) or []:
            a = AnchorDef.from_dict(d)
            if a.id not in self.bank:
                self.bank.add(a)
                self._define_blend_param(a.id)

    def _load_layout(self, name: Optional[str]) -> Dict[str, Any]:
        p = _resolve(self.root, "layouts", name)
        layout = {"id": "stereo", "name": "stereo", "decoder": "stereo", "channels": 2, "device": None}
        if p:
            try:
                layout.update(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
                layout["file"] = str(p)
                layout["id"] = p.stem          # what `--layout` and /v1/output take; the name is for people
            except (OSError, yaml.YAMLError) as exc:
                self.notes.append(f"layout {p.name}: {exc}")
        if layout.get("decoder") not in ("stereo",):
            self.notes.append(f"decoder '{layout.get('decoder')}' is not built yet; decoding to stereo (plan E5)")
        return layout

    # --- output ---------------------------------------------------------------------
    def apply_output(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Change device, channel count, layout, routing or sink while running.
        Returns the config plus a note when the device could not be opened."""
        from app.engine.audio import make_sink as _make_sink
        new = dict(self.output)
        if "layout" in cfg:
            self.layout = self._load_layout(cfg["layout"])
            new["layout"] = self.layout.get("id", str(cfg["layout"]))
            if "channels" not in cfg and self.layout.get("channels"):
                new["channels"] = int(self.layout["channels"])
        for k in ("sink", "device", "channels", "routing", "spatial_channels"):
            if k in cfg:
                new[k] = cfg[k]
        new["channels"] = max(1, int(new["channels"] or 2))
        if "radio" in cfg:
            self.radio_targets = list(cfg["radio"] or [])
        # routing first: a bad table is a 400 before anything is touched
        self.mixer.set_routing(new.get("routing") or {}, new.get("spatial_channels"))
        self.mixer.set_channels(new["channels"])
        running = bool(self._loop_thread and self._loop_thread.is_alive())
        try:
            self.sink.stop()
        except Exception:
            pass
        try:
            sink, note = _make_sink(new.get("sink") or "auto", self.mixer, path=self.root / "recordings" / "engine_output.wav",
                                    device=new.get("device"), targets=self.radio_targets)
        except Exception as exc:              # a device that does not exist or refuses the channel count
            from app.engine.audio import NullSink
            sink, note = NullSink(self.mixer), f"could not open the device: {exc}"
            new["sink"] = "null"
        if note:
            self.notes.append(note)
        self.sink, self.sink_kind = sink, getattr(sink, "name", new.get("sink"))
        if running:
            try:
                self.sink.start()
            except Exception as exc:          # a device that refuses: keep the engine alive on the null sink
                note = f"{self.sink.name}: {exc}"
                self.notes.append(note)
                from app.engine.audio import NullSink
                self.sink = NullSink(self.mixer)
                self.sink.start()
        self.output = new
        self.snapshot()
        self.events.publish("output", {**self.output, "note": note})
        return {**self.output, "note": note, "sink_status": self.sink.status()}

    def save_scenes(self) -> Path:
        out = self.root / "scenes" / "custom.yaml"
        data = {"scenes": [s.to_dict() for s in self.conductor.scenes.values()], "sequence": self.conductor.sequence}
        out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return out

    # --- model -------------------------------------------------------------------
    def load_model(self, model_id: str, loras: Optional[List[Dict[str, Any]]] = None) -> None:
        if self._loader and self._loader.is_alive():
            return
        self.model_id = model_id
        self.loras = list(loras or [])
        self.model_state = "loading"
        self.model_error = ""

        def run() -> None:
            try:
                self.renderer.load(model_id, self.loras)
                self.model_state = "ready"
                self.events.publish("model", {"model_id": model_id, "state": "ready", "anchors": len(self.bank)})
            except Exception as exc:
                self.model_state = "error"
                self.model_error = f"{exc.__class__.__name__}: {exc}"
                traceback.print_exc()
                self.events.publish("model", {"model_id": model_id, "state": "error", "error": self.model_error})

        self._loader = threading.Thread(target=run, name="model-loader", daemon=True)
        self._loader.start()

    def rebuild_anchors(self) -> Dict[str, bool]:
        if self.renderer.encoder is None:
            raise RuntimeError("no model loaded")
        return self.renderer.ensure_anchors(force=True)

    # --- lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        self.renderer.start()
        self.sink.start()
        if self.osc_server:
            try:
                self.osc_server.start()
            except OSError as exc:
                self.notes.append(f"OSC in: {exc}")
                self.osc_server = None
        self._stop.clear()
        self._loop_thread = threading.Thread(target=self._loop, name="engine-control", daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop_thread:
            self._loop_thread.join(timeout=3)
        self.snapshot()          # before stopping: a restart resumes where it left off
        self.conductor.stop()
        self.mixer.stop_record()
        self.sink.stop()
        self.renderer.stop()
        if self.osc_server:
            self.osc_server.stop()
        if self.osc_client:
            self.osc_client.close()

    def _loop(self) -> None:
        dt = 1.0 / CONTROL_HZ
        last = time.perf_counter()
        next_snap = time.time() + SNAPSHOT_EVERY_S
        next_meters = time.time() + METERS_EVERY_S
        while not self._stop.is_set():
            now = time.perf_counter()
            step = now - last
            last = now
            try:
                self.params.tick(step)
                self.mixer.master_gain = float(self.params.get("/master/gain"))
                self.mixer.limiter_on = bool(self.params.get("/master/limiter"))
                if self.model_state == "ready" or self.renderer_kind == "fake":
                    self.conductor.tick(step)
                if not self.renderer.running and not self._stop.is_set():
                    self._renderer_restarts += 1
                    self.events.publish("warning", {"error": "render worker died; restarted", "restarts": self._renderer_restarts})
                    self.renderer.start()
                t = time.time()
                if t >= next_meters:
                    next_meters = t + METERS_EVERY_S
                    self.events.publish("meters", {**self.mixer.meters(), "queue": self.renderer.stats()["queued"],
                                                   "progress": self.renderer.progress})
                if t >= next_snap:
                    next_snap = t + SNAPSHOT_EVERY_S
                    self.snapshot()
            except Exception:
                traceback.print_exc()
            sleep = dt - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)

    # --- transport, render, record ------------------------------------------------
    def transport(self, on: bool) -> None:
        if on:
            self.conductor.start()
        else:
            self.conductor.stop()
        self.events.publish("transport", {"playing": on})

    def render_oneshot(self, spec: Dict[str, Any]) -> str:
        job = RenderJob(
            role=str(spec.get("role", "oneshot")), blend={k: float(v) for k, v in (spec.get("blend") or {}).items()},
            overrides=dict(spec.get("overrides") or {}), prompt=str(spec.get("prompt", "")),
            negative_prompt=str(spec.get("negative_prompt", "")), seed=int(spec.get("seed", -1)),
            duration_s=float(spec.get("duration_s", 10.0)), cfg=float(spec.get("cfg", self.params.get("/render/cfg"))),
            steps=int(spec.get("steps", self.params.get("/render/steps"))), batch_size=int(spec.get("batch_size", 1)),
            init_clip=spec.get("init_clip"), init_strength=float(spec.get("init_strength", 0.0)),
            event_type=str(spec.get("event_type", "")), scene=str(spec.get("scene", "")))
        if not job.blend and not job.prompt:
            job.blend = self.conductor.effective_blend()
        self.oneshots[job.id] = {"id": job.id, "state": "queued", "job": job.to_dict(), "clips": [], "error": ""}
        self.renderer.submit(job)
        return job.id

    def _oneshot_done(self, job: RenderJob, clips, err) -> None:
        rec = self.oneshots.get(job.id)
        if rec is None:
            return
        rec["state"] = "error" if err else "done"
        rec["error"] = err or ""
        rec["clips"] = [c.to_dict() for c in clips]
        rec["render_s"] = self.renderer.last_render_s
        self.events.publish("render", {"id": job.id, "state": rec["state"], "clips": [c.id for c in clips], "error": err or ""})

    def record(self, on: bool) -> Optional[str]:
        if on:
            p = self.mixer.start_record(self.root / "recordings" / time.strftime("engine_%Y%m%d_%H%M%S.wav"))
            self.events.publish("record", {"recording": True, "path": str(p)})
            return str(p)
        p = self.mixer.stop_record()
        if p:
            self._sign(p)
        self.events.publish("record", {"recording": False, "path": str(p) if p else None})
        return str(p) if p else None

    def _sign(self, path: Path) -> None:
        try:
            from app.core.generation.content_credentials import sign_wav
            from app.core.generation.provenance import recording_manifest
            sign_wav(path, recording_manifest())
        except Exception:
            pass  # a failed signature never blocks a recording

    # --- presets and state ---------------------------------------------------------
    def preset_dir(self) -> Path:
        return self.root / "presets"

    def save_preset(self, name: str) -> Dict[str, Any]:
        safe = "".join(c for c in name if c.isalnum() or c in " _-").strip() or "preset"
        data = {"name": safe, "saved": time.time(), "scene": self.conductor.scene.id if self.conductor.scene else None,
                "params": self.params.snapshot(), "model_id": self.model_id, "loras": self.loras}
        (self.preset_dir() / f"{safe}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        return data

    def list_presets(self) -> List[Dict[str, Any]]:
        out = []
        for f in sorted(self.preset_dir().glob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                out.append({"name": d.get("name", f.stem), "saved": d.get("saved"), "scene": d.get("scene")})
            except (OSError, ValueError):
                continue
        return out

    def load_preset(self, name: str) -> Dict[str, Any]:
        f = self.preset_dir() / f"{name}.json"
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("scene") and d["scene"] in self.conductor.scenes:
            self.conductor.activate(d["scene"])
        n = self.params.restore(d.get("params", {}))
        return {"restored": n, "scene": d.get("scene")}

    def delete_preset(self, name: str) -> bool:
        f = self.preset_dir() / f"{name}.json"
        if f.exists():
            f.unlink()
            return True
        return False

    def snapshot(self) -> None:
        data = {"saved": time.time(), "scene": self.conductor.scene.id if self.conductor.scene else None,
                "playing": self.conductor.playing, "params": self.params.snapshot(), "model_id": self.model_id,
                "loras": self.loras, "version": __version__, "output": self.output}
        tmp = self.state_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    def _read_state(self) -> Dict[str, Any]:
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def restore(self) -> bool:
        d = self._read_state()
        if not d:
            return False
        self.params.restore(d.get("params", {}))
        if d.get("loras") and not self.loras:
            self.loras = d["loras"]
        if d.get("scene") in self.conductor.scenes:
            self.conductor.activate(d["scene"], transition_s=0.0)
        self._resume_playing = bool(d.get("playing"))
        self.notes.append(f"resumed state from {time.strftime('%H:%M:%S', time.localtime(d.get('saved', 0)))}")
        return True

    # --- events out ----------------------------------------------------------------
    def emit(self, kind: str, data: Dict[str, Any]) -> None:
        self.events.publish(kind, data)
        if kind == "scene" and hasattr(self.sink, "set_title"):
            sc = self.conductor.scenes.get(data.get("id", ""))
            c = self.conductor.clock_now()
            self.sink.set_title(f"{sc.label or sc.id}" + (f" · {c['local_time']} {c['timezone']}" if c else "") if sc else "")
        if self.osc_client:
            if kind == "event":
                self.osc_client.send(f"{OSC_PREFIX}/out/event", data.get("clip_id", ""), data.get("type", ""),
                                     float(data.get("azimuth", 0.0)), float(data.get("elevation", 0.0)), float(data.get("distance", 1.0)))
            elif kind == "scene":
                self.osc_client.send(f"{OSC_PREFIX}/out/scene", data.get("id", ""))

    def _osc_in(self, address: str, args: List[Any]) -> None:
        if not address.startswith(OSC_PREFIX + "/"):
            return
        rest = address[len(OSC_PREFIX):]
        try:
            if rest.startswith("/param/"):
                path = rest[len("/param"):]
                if args and self.params.exists(path):
                    self.params.set(path, args[0])
            elif rest == "/scene/activate" and args:
                t = float(args[1]) if len(args) > 1 else None
                az = float(args[2]) if len(args) > 2 else None
                self.conductor.activate(str(args[0]), transition_s=t, from_azimuth=az)
            elif rest == "/event/trigger" and args:
                self.conductor.trigger(str(args[0]))
            elif rest == "/transport" and args:
                self.transport(str(args[0]).lower() in ("1", "start", "on", "true"))
        except Exception as exc:
            self.events.publish("warning", {"osc": address, "error": str(exc)})

    # --- status ------------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        r = self.renderer.stats()
        return {
            "version": __version__, "uptime_s": round(time.time() - self.started_at, 1),
            "model": {"id": self.model_id, "state": self.model_state, "error": self.model_error, "loras": self.loras,
                      "loaded": r.get("loaded", False)},
            "renderer": r, "anchors": {"count": len(self.bank), "encoded": sum(1 for a in self.bank.ids() if self.bank.get(a).encoded),
                                       "file": str(self.anchors_path) if self.anchors_path else None},
            "conductor": self.conductor.status(), "audio": {**self.mixer.meters(), **self.sink.status()},
            "clips": self.clips.stats(), "layout": self.layout, "output": {**self.output, **self.sink.status()},
            "osc": {"in_port": self.osc_server.port if self.osc_server else None,
                    "received": self.osc_server.received if self.osc_server else 0,
                    "out": f"{self.osc_client.host}:{self.osc_client.port}" if self.osc_client else None},
            "stream_subscribers": self.events.subscribers, "notes": self.notes[-5:],
            "radio": self.sink.status() if getattr(self.sink, "name", "") == "radio" else None,
            "renderer_restarts": self._renderer_restarts, "scene_file": str(self.scene_path) if self.scene_path else None,
        }
