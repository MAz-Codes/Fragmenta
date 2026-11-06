"""HTTP API (`/v1`), Server-Sent Events (`/v1/stream`), the live monitor
stream (`/v1/monitor.wav`). Flask, so the engine adds no dependency
(perpetual/decisions/04). The Fragmenta UI's Engine tab uses only this.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, request, send_file

from app.engine.anchors import AnchorDef
from app.engine.audio import wav_header
from app.engine.conductor import Scene
from app.engine.engine import Engine

ALLOWED_ORIGINS = ["http://127.0.0.1:5001", "http://localhost:5001", "http://localhost:3000", "http://127.0.0.1:3000"]


def create_app(engine: Engine, shutdown_cb: Optional[Any] = None) -> Flask:
    app = Flask("fragmenta-engine")
    origins = list(ALLOWED_ORIGINS)
    extra = os.environ.get("FRAGMENTA_ALLOWED_ORIGINS", "").strip()
    if extra:
        origins.extend(o.strip() for o in extra.split(",") if o.strip())
    try:
        from flask_cors import CORS
        CORS(app, origins=origins, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    except ImportError:
        @app.after_request
        def _cors(resp):
            o = request.headers.get("Origin")
            if o in origins:
                resp.headers["Access-Control-Allow-Origin"] = o
                resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
                resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            return resp

    @app.errorhandler(Exception)
    def _err(exc):
        code = 404 if isinstance(exc, KeyError) else 400 if isinstance(exc, (ValueError, TypeError)) else 500
        if hasattr(exc, "code") and isinstance(getattr(exc, "code"), int):
            code = exc.code
        return jsonify({"error": f"{exc.__class__.__name__}: {exc}"}), code

    def body() -> Dict[str, Any]:
        return request.get_json(force=True, silent=True) or {}

    # --- status and parameters -----------------------------------------------
    @app.get("/v1/status")
    def status():
        return jsonify(engine.status())

    @app.get("/v1/params")
    def params():
        if request.args.get("flat"):
            return jsonify(engine.params.flat())
        return jsonify(engine.params.tree())

    @app.get("/v1/params/<path:p>")
    def param_get(p):
        return jsonify(engine.params.describe("/" + p))

    @app.patch("/v1/params/<path:p>")
    def param_set(p):
        b = body()
        path = "/" + p
        if "normalized" in b:
            v = engine.params.set_normalized(path, float(b["normalized"]))
        elif "value" in b:
            v = engine.params.set(path, b["value"])
        else:
            raise ValueError("send {value} or {normalized}")
        return jsonify({"path": path, "value": v})

    @app.patch("/v1/params")
    def params_set_many():
        b = body()
        vals = b.get("values", b)
        out = {}
        for k, v in vals.items():
            if engine.params.exists(k):
                out[k] = engine.params.set(k, v)
        return jsonify(out)

    @app.put("/v1/params/<path:p>/modulator")
    def modulator_set(p):
        return jsonify(engine.params.set_modulator("/" + p, body() or None))

    # --- scenes ----------------------------------------------------------------
    @app.get("/v1/scenes")
    def scenes():
        c = engine.conductor
        return jsonify({"scenes": [s.to_dict() for s in c.scenes.values()], "sequence": c.sequence,
                        "current": c.scene.id if c.scene else None})

    @app.post("/v1/scenes")
    def scene_create():
        b = body()
        if "id" not in b:
            raise ValueError("a scene needs an id")
        s = Scene.from_dict(b)
        engine.conductor.add_scene(s)
        if b.get("save", True):
            engine.save_scenes()
        return jsonify(s.to_dict()), 201

    @app.delete("/v1/scenes/<sid>")
    def scene_delete(sid):
        ok = engine.conductor.remove_scene(sid)
        if ok:
            engine.save_scenes()
        return jsonify({"deleted": ok})

    @app.post("/v1/scenes/<sid>/activate")
    def scene_activate(sid):
        b = body()
        s = engine.conductor.activate(sid, transition_s=b.get("transition_s"), from_azimuth=b.get("from_azimuth"))
        return jsonify({"activated": s.id, "transition_s": engine.conductor._trans_len})

    # --- anchors ----------------------------------------------------------------
    @app.get("/v1/anchors")
    def anchors():
        return jsonify({"anchors": engine.bank.describe(), "encoder": engine.bank.encoder_key,
                        "blend": engine.conductor.effective_blend()})

    @app.post("/v1/anchors")
    def anchors_post():
        b = body()
        if b.get("rebuild"):
            return jsonify({"rebuilt": engine.rebuild_anchors()})
        if "id" in b:
            a = AnchorDef.from_dict(b)
            engine.bank.add(a)
            if not engine.params.exists(f"/scene/blend/{a.id}"):
                engine._define_blend_param(a.id)
            if engine.renderer.encoder is not None:
                engine.renderer.ensure_anchors()
            return jsonify(a.to_dict()), 201
        raise ValueError("send {rebuild: true} or an anchor definition")

    @app.delete("/v1/anchors/<aid>")
    def anchor_delete(aid):
        ok = engine.bank.remove(aid)
        engine.params.undefine(f"/scene/blend/{aid}")
        return jsonify({"deleted": ok})

    @app.put("/v1/blend")
    def blend_set():
        """Set every blend weight at once (the pad)."""
        b = body()
        w = b.get("blend", b)
        out = {}
        for aid in engine.bank.ids():
            p = f"/scene/blend/{aid}"
            if engine.params.exists(p):
                out[aid] = engine.params.set(p, float(w.get(aid, 0.0)), notify=False)
        engine.events.publish("blend", out)
        return jsonify(out)

    # --- render, clips ------------------------------------------------------------
    @app.post("/v1/render")
    def render():
        jid = engine.render_oneshot(body())
        return jsonify({"job_id": jid, "state": "queued"}), 202

    @app.get("/v1/render/<jid>")
    def render_status(jid):
        rec = engine.oneshots.get(jid)
        if rec is None:
            raise KeyError(jid)
        cur = engine.renderer.current
        if rec["state"] == "queued" and cur and cur.id == jid:
            return jsonify({**rec, "state": "rendering", "progress": engine.renderer.progress})
        return jsonify(rec)

    @app.get("/v1/clips")
    def clips():
        rows = engine.clips.list(role=request.args.get("role"), event_type=request.args.get("type"), scene=request.args.get("scene"))
        limit = int(request.args.get("limit", 200))
        return jsonify({"clips": [c.to_dict() for c in rows[-limit:]], **engine.clips.stats()})

    @app.get("/v1/clips/<cid>")
    def clip_get(cid):
        c = engine.clips.get(cid)
        if c is None:
            raise KeyError(cid)
        if request.args.get("info"):
            return jsonify(c.to_dict())
        return send_file(c.path, mimetype="audio/wav", as_attachment=False, download_name=Path(c.path).name)

    @app.delete("/v1/clips/<cid>")
    def clip_delete(cid):
        return jsonify({"deleted": engine.clips.remove(cid)})

    # --- transport, events, record ------------------------------------------------
    @app.post("/v1/transport/<action>")
    def transport(action):
        if action not in ("start", "stop"):
            raise ValueError("start or stop")
        engine.transport(action == "start")
        return jsonify({"playing": engine.conductor.playing})

    @app.post("/v1/events/trigger")
    def trigger():
        b = body()
        return jsonify(engine.conductor.trigger(str(b.get("type", ""))))

    @app.post("/v1/record/<action>")
    def record(action):
        if action not in ("start", "stop"):
            raise ValueError("start or stop")
        p = engine.record(action == "start")
        return jsonify({"recording": engine.mixer.recording, "path": p})

    # --- model, output, presets ------------------------------------------------------
    @app.get("/v1/model")
    def model_get():
        return jsonify({"model_id": engine.model_id, "state": engine.model_state, "error": engine.model_error,
                        "loras": engine.loras, "renderer": engine.renderer.name})

    @app.post("/v1/model")
    def model_set():
        b = body()
        engine.load_model(str(b.get("model_id", engine.model_id)), b.get("loras"))
        return jsonify({"model_id": engine.model_id, "state": engine.model_state}), 202

    @app.get("/v1/output")
    def output_get():
        return jsonify({**engine.output, "layout_file": engine.layout, "sink_status": engine.sink.status(),
                        "mixer": engine.mixer.output_config(), "layers": ["bed:%d" % i for i in range(int(engine.params.get("/bed/takes")))] + ["events", "soundmarks"]})

    @app.get("/v1/output/devices")
    def output_devices():
        from app.engine.audio import list_output_devices
        devs = list_output_devices()
        return jsonify({"devices": devs, "note": "" if devs else "no PortAudio devices: install the sounddevice extra for speaker output"})

    @app.put("/v1/output")
    def output_put():
        return jsonify(engine.apply_output(body()))

    @app.get("/v1/presets")
    def presets():
        return jsonify({"presets": engine.list_presets()})

    @app.post("/v1/presets")
    def preset_save():
        b = body()
        return jsonify(engine.save_preset(str(b.get("name", "preset")))), 201

    @app.post("/v1/presets/<name>/load")
    def preset_load(name):
        return jsonify(engine.load_preset(name))

    @app.delete("/v1/presets/<name>")
    def preset_delete(name):
        return jsonify({"deleted": engine.delete_preset(name)})

    # --- streams --------------------------------------------------------------------
    @app.get("/v1/stream")
    def stream():
        q = engine.events.subscribe()

        def gen():
            yield "retry: 2000\nevent: hello\ndata: " + json.dumps({"version": engine.status()["version"]}) + "\n\n"
            last = time.time()
            try:
                while True:
                    try:
                        kind, data, ts = q.get(timeout=5.0)
                        yield f"event: {kind}\ndata: {json.dumps({**data, 't': ts})}\n\n"
                    except queue.Empty:
                        pass
                    if time.time() - last > 15:
                        last = time.time()
                        yield ": keepalive\n\n"
            finally:
                engine.events.unsubscribe(q)

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/v1/monitor.wav")
    def monitor():
        q = engine.mixer.tap()

        def gen():
            yield wav_header(engine.mixer.sr, 2, 16)
            try:
                while True:
                    try:
                        yield q.get(timeout=5.0)
                    except queue.Empty:
                        # keep the connection alive with silence
                        yield b"\0" * (1024 * 4)
            finally:
                engine.mixer.untap(q)

        return Response(gen(), mimetype="audio/wav", headers={"Cache-Control": "no-cache", "Accept-Ranges": "none"})

    @app.post("/v1/shutdown")
    def shutdown():
        def later():
            time.sleep(0.2)
            if shutdown_cb:
                shutdown_cb()
        threading.Thread(target=later, daemon=True).start()
        return jsonify({"stopping": True})

    @app.get("/")
    def root():
        return jsonify({"fragmenta-engine": engine.status()["version"], "api": "/v1"})

    return app
