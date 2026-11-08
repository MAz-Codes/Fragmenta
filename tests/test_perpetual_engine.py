"""The perpetual engine without a model: parameters, anchors, clips, OSC, the
audio layer, the conductor on the fake renderer, and the /v1 API end to end.

Pure stdlib + the project venv; no pytest. Run:
    venv/bin/python tests/test_perpetual_engine.py
"""
import json
import os
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.engine import anchors as A  # noqa: E402
from app.engine import osc  # noqa: E402
from app.engine.audio import BLOCK, SR, BedStream, EventVoice, Limiter, Mixer, NullSink, Spatializer, Trajectory  # noqa: E402
from app.engine.clips import ClipStore, analyse  # noqa: E402
from app.engine.conductor import Conductor, Scene  # noqa: E402
from app.engine.params import Modulator, Parameter, ParameterStore, apply_curve  # noqa: E402
from app.engine.renderer import FakeRenderer, RenderJob, post_process  # noqa: E402

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def wait(pred, timeout=10.0, step=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


# --- params -------------------------------------------------------------------
@check("params: define, set, range clamp, kinds")
def _():
    s = ParameterStore()
    s.define("/events/density", 0.4, (0.0, 1.0), curve="exp")
    s.define("/bed/takes", 2, (1, 4))
    s.define("/master/limiter", True)
    s.define("/render/seed_mode", "spread")
    s.define("/spatial/events/trajectories", ["static"], kind="list")
    assert s.set("/events/density", 1.7) == 1.0
    assert s.set("/bed/takes", 9) == 4 and isinstance(s.get("/bed/takes"), int)
    assert s.set("/master/limiter", "off") is False
    assert s.set("/render/seed_mode", "fixed") == "fixed"
    assert s.set("/spatial/events/trajectories", "passby") == ["passby"]
    assert s.paths("/events") == ["/events/density"]
    assert "events" in s.tree() and s.tree()["events"]["density"]["value"] == 1.0


@check("params: curves and normalised control")
def _():
    assert apply_curve("exp", 0.5) == 0.25 and abs(apply_curve("log", 0.25) - 0.5) < 1e-9
    s = ParameterStore()
    p = s.define("/x", 0.0, (0.0, 10.0), curve="exp")
    s.set_normalized("/x", 0.5)
    assert abs(p.value - 2.5) < 1e-9
    assert abs(p.normalized() - 0.5) < 1e-9


@check("params: smoothing approaches the target; modulators stay in range and are reproducible")
def _():
    s = ParameterStore()
    s.define("/a", 0.0, (0.0, 1.0), smoothing_s=1.0)
    s.set("/a", 1.0)
    s.tick(0.1)
    v1 = s.get("/a")
    assert 0.05 < v1 < 0.2, v1
    for _ in range(60):
        s.tick(0.1)
    assert s.get("/a") > 0.99
    # walk: bounded, deterministic per seed
    s.define("/w", 0.5, (0.0, 1.0), modulator={"type": "walk", "rate": 1.0, "depth": 0.2, "seed": 3})
    vals = [s.tick(0.04) or s.get("/w") for _ in range(200)]
    assert all(0.3 - 1e-6 <= v <= 0.7 + 1e-6 for v in vals), (min(vals), max(vals))
    s2 = ParameterStore()
    s2.define("/w", 0.5, (0.0, 1.0), modulator={"type": "walk", "rate": 1.0, "depth": 0.2, "seed": 3})
    vals2 = [s2.tick(0.04) or s2.get("/w") for _ in range(200)]
    assert vals == vals2
    # lfo
    s.define("/l", 0.5, (0.0, 1.0), modulator={"type": "lfo", "rate": 1.0, "depth": 0.5})
    s.tick(0.25)
    assert abs(s.get("/l") - 1.0) < 1e-6
    # sequence replaces the value
    s.define("/q", 1, (0, 10), modulator={"type": "sequence", "rate": 10.0, "values": [3, 5, 7]})
    s.tick(0.01)
    assert s.get("/q") == 3
    s.tick(0.1)
    assert s.get("/q") == 5
    # distribution
    s.define("/d", 0.5, (0.0, 1.0), modulator={"type": "distribution", "rate": 100.0, "depth": 0.1, "seed": 1})
    ds = set()
    for _ in range(20):
        s.tick(0.02)
        ds.add(round(s.get("/d"), 6))
    assert len(ds) > 3 and all(0.4 - 1e-6 <= d <= 0.6 + 1e-6 for d in ds)


@check("params: snapshot / restore, listeners, modulator replace")
def _():
    s = ParameterStore()
    s.define("/a", 0.1, (0.0, 1.0))
    s.define("/b", "x")
    seen = []
    s.subscribe(lambda p, v: seen.append((p, v)))
    s.set("/a", 0.6)
    s.set_modulator("/a", {"type": "lfo", "rate": 2.0, "depth": 0.1})
    snap = s.snapshot()
    assert snap["/a"]["modulator"]["type"] == "lfo"
    s.set("/a", 0.0)
    s.set_modulator("/a", None)
    assert s.restore(snap) == 2 and s.base("/a") == 0.6 and s.param("/a").modulator.type == "lfo"
    assert seen[0] == ("/a", 0.6)


# --- anchors --------------------------------------------------------------------
@check("anchors: the fixed template, identifiers, canonical sources, ASCII")
def _():
    p = A.render_prompt({"type": "event", "place": "Café", "perspective": "close", "density": "moderate",
                         "time": "evening", "setting": "interior", "sources": ["voices", "vendors", "zzz"]})
    assert p == "TrackType: SFX, Type: event, Place: cafe, Perspective: close, Density: moderate, Time: evening, Setting: interior, Sources: vendors, voices, zzz", p
    assert A.render_prompt({"place": "bazaar"}) == "TrackType: SFX, Place: bazaar"
    assert A.render_prompt({"sources": []}) == "TrackType: SFX"
    d = A.AnchorDef(id="x", fields={"type": "ambience", "place": "park"}, variants=[{"time": "morning"}, {"time": "morning"}, {}])
    assert len(d.prompts()) == 2


@check("anchors: centroid encoding, disk cache keyed by encoder, weighted blend with mask union")
def _():
    with tempfile.TemporaryDirectory() as td:
        defs = [A.AnchorDef(id="a", fields={"place": "bazaar", "sources": ["voices"]}, variants=[{"sources": ["vendors"]}]),
                A.AnchorDef(id="b", fields={"place": "park", "sources": ["birds", "water", "voices"]})]
        bank = A.AnchorBank(defs, Path(td))
        enc = A.FakeEncoder()
        r = bank.encode_all(enc)
        assert r == {"a": True, "b": True} and enc.calls == 2
        a, b = bank.get("a"), bank.get("b")
        e, m = A.FakeEncoder().encode(defs[0].prompts())
        assert np.allclose(a.embedding, e.mean(axis=0)) and (a.mask == m.any(axis=0)).all()
        # cache hit on a second bank with the same encoder key
        bank2 = A.AnchorBank(defs, Path(td))
        r2 = bank2.encode_all(enc)
        assert r2 == {"a": False, "b": False} and enc.calls == 2
        # a different encoder key re-encodes
        bank3 = A.AnchorBank(defs, Path(td))
        assert bank3.encode_all(A.FakeEncoder(key="other")) == {"a": True, "b": True}
        emb, mask, w = bank.blend({"a": 3.0, "b": 1.0}, rescale=False)
        assert w == {"a": 0.75, "b": 0.25}
        assert np.allclose(emb, a.embedding * 0.75 + b.embedding * 0.25)
        assert (mask == (a.mask | b.mask)).all()
        emb2, _, _ = bank.blend({"a": 1.0, "b": 1.0}, rescale=True)
        assert abs(np.linalg.norm(emb2) - (a.norm + b.norm) / 2) < 1e-3
        assert bank.prompt_for({"a": 0.2, "b": 0.8}).startswith("TrackType: SFX, Place: park")
        try:
            bank.blend({"zzz": 1.0})
            assert False
        except ValueError:
            pass
        assert bank.describe()[0]["position"] is not None


# --- clips ----------------------------------------------------------------------
@check("clips: analysis finds the band; store adds, loads, evicts by LRU, keeps pinned")
def _():
    t = np.arange(SR * 2) / SR
    low = np.sin(2 * np.pi * 120 * t).astype(np.float32) * 0.5
    high = np.sin(2 * np.pi * 6000 * t).astype(np.float32) * 0.5
    assert analyse(low, SR)["band"] == "low" and analyse(high, SR)["band"] == "high"
    assert abs(analyse(np.stack([low, low]), SR)["loudness_db"] - (-9.0)) < 1.5
    with tempfile.TemporaryDirectory() as td:
        store = ClipStore(Path(td))
        ids = []
        for i in range(3):
            c = store.add(np.stack([low, high]), SR, "event", event_type="voices", seed=i)
            ids.append(c.id)
            time.sleep(0.01)
        assert store.stats()["count"] == 3
        store.cap_bytes = c.bytes * 3 + 100          # room for three clips, not four
        store.pin(ids[0])
        store.touch(ids[1])
        store.add(np.stack([low, high]), SR, "bed")
        remaining = {c.id for c in store.list()}
        assert ids[0] in remaining and ids[2] not in remaining, remaining
        a = store.load(ids[0])
        assert a.shape == (2, SR * 2) and a.dtype == np.float32
        assert [c.id for c in store.list(role="event", event_type="voices")] == [i for i in ids if i in remaining]
        store2 = ClipStore(Path(td))
        assert store2.stats()["count"] == store.stats()["count"]
        assert store.remove(ids[0]) and store.get(ids[0]) is None


@check("renderer: post-processing per role")
def _():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (2, SR * 3)).astype(np.float32)
    a[:, : SR] *= 0.001                    # a quiet lead-in an event should trim
    bed = post_process(a, SR, "bed")
    assert bed.shape == a.shape and abs(20 * np.log10(np.sqrt(np.mean(bed ** 2))) - (-22)) < 1.0
    ev = post_process(a, SR, "event")
    assert ev.shape[0] == 1 and ev.shape[1] < SR * 2.3 and ev[0, 0] == 0.0
    assert abs(20 * np.log10(np.sqrt(np.mean(ev ** 2))) - (-16)) < 1.5


# --- osc ------------------------------------------------------------------------
@check("osc: codec round trip and a live UDP server")
def _():
    msg = osc.encode("/fragmenta/param/events/density", 0.25, 3, "night_metro", True, b"\x01\x02\x03")
    [(addr, args)] = osc.decode(msg)
    assert addr == "/fragmenta/param/events/density" and args[1] == 3 and args[2] == "night_metro" and args[3] is True
    assert abs(args[0] - 0.25) < 1e-6 and args[4] == b"\x01\x02\x03"
    bundle = b"#bundle\0" + b"\0" * 8
    for m in (osc.encode("/a", 1), osc.encode("/b", 2.0)):
        bundle += len(m).to_bytes(4, "big") + m
    assert [a for a, _ in osc.decode(bundle)] == ["/a", "/b"]
    got = []
    srv = osc.OscServer("127.0.0.1", 0, lambda a, args: got.append((a, args)))
    srv.start()
    try:
        c = osc.OscClient("127.0.0.1", srv.port)
        c.send("/fragmenta/scene/activate", "metro", 20.0)
        assert osc.wait_for(lambda: len(got) == 1) and got[0][0] == "/fragmenta/scene/activate" and got[0][1][0] == "metro"
    finally:
        srv.stop()


# --- audio ----------------------------------------------------------------------
@check("audio: spatializer gains, distance, rear")
def _():
    sp = Spatializer()
    g, cut = sp.gains(0, 0, 1)
    assert abs(g[0] - g[1]) < 1e-6 and cut > 15000
    gl, _ = sp.gains(90, 0, 1)
    assert gl[0] > 0.99 and gl[1] < 1e-6
    gr, _ = sp.gains(-90, 0, 1)
    assert gr[1] > 0.99
    gf, cutf = sp.gains(0, 0, 4)
    assert abs(gf[0] / g[0] - 0.25) < 1e-6 and cutf < cut
    gb, _ = sp.gains(180, 0, 1)
    assert gb.sum() < g.sum()


@check("audio: bed stream chains clips with a crossfade, loops when starved, enters from an azimuth")
def _():
    b = BedStream("t", crossfade_s=0.5)
    one = np.ones((2, SR), dtype=np.float32) * 0.5
    b.push("c1", one)
    b.push("c2", one * 0.5)
    b.fade_in(0.01)
    out = np.concatenate([b.pull(BLOCK) for _ in range(SR // BLOCK + 2)], axis=1)
    assert out.shape[0] == 2 and out[0, int(0.3 * SR)] > 0.45
    # the crossfade region: between the two levels, never a gap
    tail = out[0, SR - int(0.25 * SR)]
    assert 0.25 < tail < 0.56, tail   # equal-power: 0.5·cos(π/4) + 0.25·sin(π/4)
    assert b.played_ids[:2] == ["c1", "c2"]
    assert b.ahead_seconds() > 0
    for _ in range(SR // BLOCK + 2):
        b.pull(BLOCK)
    assert b.starvations >= 1 and np.abs(b.pull(BLOCK)).max() > 0
    b2 = BedStream("u")
    b2.push("c", one)
    b2.enter_from(90.0, 0.5)
    first = b2.pull(BLOCK)
    assert first[0].max() >= first[1].max()
    for _ in range(int(0.6 * SR / BLOCK)):
        last = b2.pull(BLOCK)
    assert abs(b2.azimuth) < 1e-6 and abs(last[0].mean() - last[1].mean()) < 1e-3
    b2.recede(0.1)
    for _ in range(10):
        b2.pull(BLOCK)
    assert b2.done


@check("audio: event voice pans, moves on a pass-by, finishes; limiter holds the ceiling")
def _():
    mono = np.ones(SR // 2, dtype=np.float32) * 0.3
    v = EventVoice("e", mono, "motorcycles", azimuth=80, distance=1.0, trajectory=Trajectory("passby", speed=6.0))
    a0 = v.azimuth
    blocks = []
    while not v.done:
        blocks.append(v.pull(BLOCK))
    out = np.concatenate(blocks, axis=1)
    assert out.shape[0] == 2 and v.azimuth != a0 and v.to_dict()["progress"] == 1.0
    lim = Limiter(ceiling_db=-1.0)
    loud = np.ones((2, BLOCK), dtype=np.float32) * 2.0
    y = lim.process(loud)
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6 and lim.reduction_db < -5


@check("audio: mixer sums beds and voices, meters, taps and records")
def _():
    with tempfile.TemporaryDirectory() as td:
        m = Mixer()
        b = m.bed("g1:0", crossfade_s=0.1)
        b.push("c", np.ones((2, SR), dtype=np.float32) * 0.2)
        b.fade_in(0.001)
        m.add_voice(EventVoice("e", np.ones(BLOCK * 3, dtype=np.float32) * 0.2, "voices", azimuth=0))
        q = m.tap()
        p = m.start_record(Path(td) / "rec.wav")
        done = []
        m.on_voice_done = lambda v: done.append(v.clip_id)
        for _ in range(6):
            out = m.pull(BLOCK)
        assert out.shape == (2, BLOCK) and m.meters()["peak_db"][0] > -20
        assert done == ["e"] and not q.empty() and len(q.get()) == BLOCK * 4
        m.stop_record()
        import soundfile as sf
        data, sr = sf.read(str(p))
        assert sr == SR and data.shape == (BLOCK * 6, 2)
        sink = NullSink(m, realtime=False)
        sink.start()
        assert wait(lambda: sink.blocks > 20, 3.0)
        sink.stop()


# --- conductor on the fake renderer ------------------------------------------------
def _engine_parts(td, render_time=0.0):
    from app.engine.engine import Engine
    eng = Engine(Path(td), scene="places", layout="stereo", renderer="fake", sink="null", osc_port=0, osc_out=None,
                 fake_render_time=render_time, autoload=True, resume=False)
    eng.params.set("/bed/clip_length", 10.0)
    eng.params.set("/bed/crossfade_s", 2.0)
    eng.params.set("/events/pool_size", 1)
    eng.params.set("/events/render_batch", 1)
    eng.params.set("/events/duration_max", 2.0)
    return eng


@check("conductor: beds refill to the low-water mark, all takes share the blend, events fire within the budget")
def _():
    with tempfile.TemporaryDirectory() as td:
        eng = _engine_parts(td)
        assert wait(lambda: eng.model_state == "ready", 5.0)
        assert len(eng.bank) == 8 and all(eng.bank.get(a).encoded for a in eng.bank.ids())
        eng.start()
        try:
            eng.conductor.activate("bazaar_midday", transition_s=0.0)
            eng.params.set("/events/density", 1.0)
            eng.params.set("/events/sparseness", 0.0)
            eng.params.set("/events/min_interval", 0.3)
            eng.params.set("/events/density_span", 0.0)
            eng.params.set("/events/max_concurrent", 2)
            eng.transport(True)
            assert wait(lambda: len(eng.mixer.beds) == 2 and all(b.ahead_seconds() >= 10 for b in eng.mixer.beds.values()), 15.0), eng.mixer.meters()
            beds = eng.clips.list(role="bed")
            assert len(beds) >= 2 and all(c.blend == beds[0].blend for c in beds) and beds[0].blend == {"bazaar": 1.0}
            assert beds[0].prompt.startswith("TrackType: SFX, Type: ambience, Place: bazaar")
            assert wait(lambda: eng.conductor.events_fired >= 3, 20.0), eng.conductor.status()
            assert len(eng.mixer.active_voices()) <= 2
            ev = eng.clips.list(role="event")
            assert ev and all(c.channels == 1 for c in ev) and ev[0].prompt.startswith("TrackType: SFX, Type: event, Place: bazaar, Perspective: close")
            st = eng.conductor.status()
            assert st["scene"] == "bazaar_midday" and st["pools"] and set(st["event_counts"]) <= set(eng.conductor.scene.events)
        finally:
            eng.stop()


@check("conductor: a transition interpolates the blend, old beds recede, new ones enter from the azimuth; trigger plays")
def _():
    with tempfile.TemporaryDirectory() as td:
        eng = _engine_parts(td)
        assert wait(lambda: eng.model_state == "ready", 5.0)
        eng.start()
        try:
            eng.conductor.activate("metro", transition_s=0.0)
            eng.transport(True)
            assert wait(lambda: len(eng.mixer.beds) == 2, 10.0)
            old_keys = set(eng.mixer.beds)
            eng.conductor.activate("park_morning", transition_s=1.0, from_azimuth=90.0)
            assert eng.conductor.effective_blend() == {"metro": 1.0}
            time.sleep(0.4)
            b = eng.conductor.effective_blend()
            assert 0.0 < b.get("metro", 0) < 1.0 and 0.0 < b.get("park", 0) < 1.0, b
            assert all(eng.mixer.beds[k].state == "receding" for k in old_keys if k in eng.mixer.beds)
            assert wait(lambda: any(k not in old_keys for k in eng.mixer.beds), 10.0)
            new = [eng.mixer.beds[k] for k in eng.mixer.beds if k not in old_keys][0]
            assert new.azimuth > 0 or new.state in ("entering", "playing")
            time.sleep(0.8)
            assert eng.conductor.effective_blend() == {"park": 1.0}
            assert wait(lambda: all(k not in eng.mixer.beds for k in old_keys), 5.0)
            r = eng.conductor.trigger("birds")
            assert r.get("played") or r.get("queued")
            assert wait(lambda: any(v.event_type == "birds" for v in eng.mixer.active_voices()) or eng.conductor.events_fired > 0, 10.0)
            # per-type probability parameters follow the scene
            assert eng.params.exists("/events/types/birds/probability") and not eng.params.exists("/events/types/machinery/probability")
        finally:
            eng.stop()


@check("conductor: starvation keeps audio running; wander moves the blend; snapshot resumes the scene")
def _():
    with tempfile.TemporaryDirectory() as td:
        eng = _engine_parts(td, render_time=3.0)     # slow renderer: beds can't keep up with 10 s clips
        assert wait(lambda: eng.model_state == "ready", 5.0)
        eng.params.set("/bed/clip_length", 10.0)
        eng.params.set("/bed/takes", 1)
        eng.start()
        try:
            eng.conductor.activate("shop", transition_s=0.0)
            eng.transport(True)
            assert wait(lambda: eng.mixer.beds and any(b.ahead_seconds() > 0 for b in eng.mixer.beds.values()), 15.0)
            # first clip arrives; the next takes 3 s of render while 10 s play → fine. Force starvation by draining:
            b = list(eng.mixer.beds.values())[0]
            for _ in range(int(16 * SR / BLOCK)):
                b.pull(BLOCK)
            assert b.starvations >= 1
            eng.params.set("/scene/wander/on", True)
            eng.params.set("/scene/wander/rate", 1.0)
            eng.params.set("/scene/wander/depth", 1.0)
            time.sleep(0.6)
            bl = eng.conductor.effective_blend()
            assert len(bl) >= 2 or bl != {"shop": 0.7 / 1.0, "street": 0.3}, bl
            eng.snapshot()
            assert json.loads(eng.state_path.read_text())["scene"] == "shop"
        finally:
            eng.stop()
        from app.engine.engine import Engine
        eng2 = Engine(Path(td), scene="places", renderer="fake", sink="null", osc_port=0, osc_out=None, autoload=False, resume=True)
        assert eng2.conductor.scene and eng2.conductor.scene.id == "shop" and eng2._resume_playing is True
        assert eng2.params.base("/bed/takes") == 1


@check("program: steps advance on time with their own transitions, loop, jump, edit live, resume")
def _():
    with tempfile.TemporaryDirectory() as td:
        eng = _engine_parts(td)
        assert wait(lambda: eng.model_state == "ready", 5.0)
        c = eng.conductor
        clock = [1_000_000.0]
        c.now_fn = lambda: clock[0]
        progs = {p["id"] for p in eng.list_programs()}
        assert "tehran_hour" in progs, progs                      # shipped preset copied on first run
        r = eng.load_program("tehran_hour", start_step=None)
        assert r["problems"] == [], r["problems"]
        assert c.program.total_seconds == 58 * 60
        eng.start()
        try:
            eng.transport(True)
            assert c.step_index == 0 and c.scene.id == "street_midday"
            assert eng.params.get("/bed/source") == "both"
            clock[0] += 9 * 60; c.tick(0.1)
            assert c.step_index == 0
            clock[0] += 61; c.tick(0.1)
            assert c.step_index == 1 and c.scene.id == "bazaar_midday" and c._trans_len == 30.0
            st = c.program_status()
            assert st["step"] == 1 and st["steps"] == 6 and st["loop"] is True
            c.goto_step(5, transition_s=0.0)
            clock[0] += 12 * 60 + 1; c.tick(0.1)
            assert c.step_index == 0, "loops to the first step"
            # edit while playing: a shorter program keeps the step in range
            short = c.program.to_dict(); short["steps"] = short["steps"][:2]; short["loop"] = False; short["id"] = "two_steps"
            eng.save_program(short)
            assert (eng.programs_dir() / "two_steps.yaml").exists()
            eng.load_program("two_steps", start_step=1)
            clock[0] += 10 * 60 + 1; c.tick(0.1)
            assert c.program_ended is True and c.scene.id == "bazaar_midday", "a non-looping program stays on its last scene"
            eng.snapshot()
            d = json.loads(eng.state_path.read_text())
            assert d["program"] == "two_steps" and d["program_step"] == 1
        finally:
            eng.stop()
        from app.engine.engine import Engine
        eng2 = Engine(Path(td), scene="places", renderer="fake", sink="null", osc_port=0, osc_out=None, autoload=False, resume=True)
        assert eng2.conductor.program and eng2.conductor.program.id == "two_steps" and eng2.conductor.step_index == 1
        assert eng2.conductor.scene.id == "bazaar_midday"
        # the command-line / API path: a program by name loads its scene file
        eng3 = Engine(Path(td), scene="tehran_radio", renderer="fake", sink="null", osc_port=0, osc_out=None, autoload=False, resume=False, program="tehran_hour")
        assert eng3.conductor.program.id == "tehran_hour" and "street_midday" in eng3.conductor.scenes


@check("renderer: a dead-context CUDA error stops the worker and flags the engine; OOM does not; beds lose their wind-down tail")
def _():
    from app.engine.renderer import BaseRenderer, _is_fatal_device_error, _trim_bed
    class Dead(Exception):
        pass
    Dead.__name__ = "AcceleratorError"
    assert _is_fatal_device_error(Dead("CUDA error: the launch timed out and was terminated"))
    assert _is_fatal_device_error(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not _is_fatal_device_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    with tempfile.TemporaryDirectory() as td:
        eng = _engine_parts(td)
        assert wait(lambda: eng.model_state == "ready", 5.0)
        r = eng.renderer
        def boom(job):
            raise Dead("CUDA error: the launch timed out and was terminated")
        r._render = boom
        eng.start()
        try:
            eng.transport(True)
            assert wait(lambda: r.fatal is not None, 10.0), "fatal not set"
            assert wait(lambda: eng.fatal_exit and eng.model_state == "error", 5.0)
            time.sleep(0.6)
            assert r.errors <= 3, f"worker kept failing jobs: {r.errors}"
            assert not r.running, "worker should have stopped"
            assert eng._renderer_restarts == 0, "a fatal worker must not be restarted in-process"
            assert "restarts itself" in eng.model_error
        finally:
            eng.stop()
    # bed trim: a clip that decays over its last 6 s loses that tail, keeps >= 70 %
    sr = 44100
    n = 30 * sr
    a = np.random.default_rng(0).normal(0, 0.1, (2, n)).astype(np.float32)
    ramp = np.ones(n, dtype=np.float32); ramp[-6 * sr:] = np.linspace(1, 0.01, 6 * sr)
    a *= ramp
    b = _trim_bed(a, sr)
    assert 0.7 * n <= b.shape[1] <= n - int(2.5 * sr), b.shape   # cut where the level has fallen 6 dB (gain 0.5, ~3 s into the ramp)
    flat = np.random.default_rng(1).normal(0, 0.1, (2, n)).astype(np.float32)
    assert _trim_bed(flat, sr).shape[1] >= n - int(0.5 * sr) - sr, "a steady clip loses only its head"


# --- output routing ---------------------------------------------------------------------
@check("output: layers route to fixed channels or the spatial pair; monitor hears everything; channel count changes")
def _():
    m = Mixer(channels=4)
    m.set_routing({"bed": {"mode": "direct", "channels": [3, 4]}, "events": {"mode": "spatial"}})
    b = m.bed("g1:0", crossfade_s=0.1)
    b.push("c", np.ones((2, SR), dtype=np.float32) * 0.3)
    b.fade_in(0.001)
    m.add_voice(EventVoice("e", np.ones(BLOCK * 8, dtype=np.float32) * 0.2, "voices", azimuth=0))
    for _ in range(4):
        out = m.pull(BLOCK)
    assert out.shape == (4, BLOCK)
    assert out[2].max() > 0.25 and out[3].max() > 0.25          # the bed on 3-4
    assert 0.05 < out[0].max() < 0.2 and abs(out[0].max() - out[1].max()) < 1e-3   # only the event on 1-2
    assert m.monitor_peak[0] > 0.3                              # monitor = bed + event
    assert m.meters()["channels"] == 4 and len(m.meters()["peak_db"]) == 4
    m.set_routing({"bed:0": {"mode": "direct", "channels": [1]}})
    out = m.pull(BLOCK)
    assert out[0].max() > 0.3 and out[2].max() < 0.01           # take 0 overrides "bed"; mono sum on channel 1
    m.set_channels(2)
    out = m.pull(BLOCK)
    assert out.shape == (2, BLOCK)
    for bad in ({"nope": {"mode": "direct", "channels": [1]}}, {"bed": {"mode": "direct"}}, {"bed": {"mode": "x"}}):
        try:
            m.set_routing(bad)
            assert False, bad
        except ValueError:
            pass


@check("output API: devices listed, config applied live, saved and resumed")
def _():
    from app.engine.api import create_app
    from app.engine.engine import Engine
    with tempfile.TemporaryDirectory() as td:
        eng = Engine(Path(td), scene="places", renderer="fake", sink="null", osc_port=0, osc_out=None, autoload=False, resume=False)
        eng.start()
        c = create_app(eng).test_client()
        try:
            d = c.get("/v1/output/devices").get_json()
            assert isinstance(d["devices"], list)
            o = c.get("/v1/output").get_json()
            assert o["channels"] == 2 and o["sink"] == "null" and o["layers"] == ["bed:0", "bed:1", "events", "soundmarks"]
            r = c.put("/v1/output", json={"channels": 6, "routing": {"bed": {"mode": "direct", "channels": [3, 4]}, "events": {"mode": "direct", "channels": [5, 6]}}, "sink": "null"})
            assert r.status_code == 200, r.get_json()
            assert r.get_json()["channels"] == 6 and eng.mixer.channels == 6 and eng.mixer.route_for("bed:1")["channels"] == [3, 4]
            assert c.put("/v1/output", json={"routing": {"bed": {"mode": "direct"}}}).status_code == 400
            assert eng.mixer.channels == 6                      # a refused table changes nothing
            r2 = c.put("/v1/output", json={"layout": "8ch", "sink": "null"}).get_json()
            assert r2["channels"] == 8 and r2["layout"] == "8ch" and eng.layout["name"] == "8ch ring"
            assert c.get("/v1/status").get_json()["output"]["channels"] == 8
            r3 = c.put("/v1/output", json={"sink": "sounddevice", "device": "no such device ever"}).get_json()
            assert r3["sink_status"]["sink"] == "null" and r3["note"]        # unknown device → null sink, with a note
        finally:
            eng.stop()
        eng2 = Engine(Path(td), scene="places", renderer="fake", sink="auto", osc_port=0, osc_out=None, autoload=False, resume=True)
        assert eng2.output["channels"] == 8 and eng2.mixer.route_for("bed:0")["channels"] == [3, 4] and eng2.layout["name"] == "8ch ring"
        eng3 = Engine(Path(td), scene="places", renderer="fake", sink="null", osc_port=0, osc_out=None, autoload=False, resume=True, channels=2)
        assert eng3.output["channels"] == 2                     # the command line wins over the saved state


# --- radio and clock ------------------------------------------------------------------
@check("radio: PCM through ffmpeg to a target, bytes counted, reconnects after ffmpeg dies")
def _():
    import shutil
    import soundfile as sf
    from app.engine.audio import RadioSink, radio_target, _mask
    assert _mask("icecast://source:hunter2@relay:8000/t.mp3") == "icecast://source:***@relay:8000/t.mp3"
    assert radio_target("icecast://s:p@h/tehran.opus")["codec"] == "libopus" and radio_target("x.mp3")["format"] == "mp3"
    if not shutil.which("ffmpeg"):
        print("   (ffmpeg not installed: codec part skipped)")
        return
    with tempfile.TemporaryDirectory() as td:
        m = Mixer()
        m.add_voice(EventVoice("e", np.sin(2 * np.pi * 440 * np.arange(SR * 3) / SR).astype(np.float32) * 0.3, "tone", azimuth=0))
        out = Path(td) / "radio.mp3"
        sink = RadioSink(m, [radio_target(str(out))], realtime=False)
        sink.start()
        assert wait(lambda: sink.bytes_out > SR * 4 * 2, 10.0), sink.status()
        st = sink.status()
        assert st["connected"] and st["restarts"] == 0 and st["targets"] == [str(out)]
        # kill ffmpeg under it: the sink notices, backs off, respawns
        sink._proc.kill()
        assert wait(lambda: sink.restarts >= 1 and sink.connected, 8.0), sink.status()
        sink.stop()
        assert not sink.connected and out.exists() and out.stat().st_size > 1000
        info = sf.info(str(out))
        # the restart overwrote the file target, so only the tail after it is there
        assert info.samplerate == SR and info.duration > 0.3, info


@check("clock: the program picks the scene for the place's local time; include pulls the shared scenes")
def _():
    with tempfile.TemporaryDirectory() as td:
        from app.engine.engine import Engine
        eng = Engine(Path(td), scene="tehran_radio", renderer="fake", sink="null", osc_port=0, osc_out=None, autoload=False, resume=False)
        c = eng.conductor
        assert len(c.scenes) == 10 and c.clock_tz == "Asia/Tehran" and len(c.clock_program) == 10
        assert eng.params.base("/scene/clock/on") is True
        import calendar
        from datetime import datetime, timezone
        # 06:30 in Tehran (UTC+3:30) is 03:00 UTC
        c.now_fn = lambda: calendar.timegm(datetime(2026, 11, 6, 3, 0, tzinfo=timezone.utc).timetuple())
        n = c.clock_now()
        assert n["local_time"] == "06:30" and n["scene"] == "park_morning", n
        c.now_fn = lambda: calendar.timegm(datetime(2026, 11, 6, 21, 15, tzinfo=timezone.utc).timetuple())   # 00:45 Tehran
        assert c.clock_now()["scene"] == "night_metro"
        c.now_fn = lambda: calendar.timegm(datetime(2026, 11, 6, 0, 30, tzinfo=timezone.utc).timetuple())    # 04:00 Tehran: before the first entry → last of the day
        assert c.clock_now()["scene"] == "night_street"
        c.start()
        assert c.scene.id == "night_street"
        c.now_fn = lambda: calendar.timegm(datetime(2026, 11, 6, 7, 0, tzinfo=timezone.utc).timetuple())     # 10:30 Tehran
        c.tick(0.04)
        assert c.scene.id == "bazaar_midday" and c.status()["clock"]["scene"] == "bazaar_midday"
        eng.params.set("/scene/clock/on", False)
        c.now_fn = lambda: calendar.timegm(datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc).timetuple())
        c.tick(0.04)
        assert c.scene.id == "bazaar_midday"      # the clock is off: nothing moves
        eng.stop()


# --- the API -------------------------------------------------------------------------
@check("api: status, params, modulator, blend, scenes, render → clip, transport, record, presets, output, stream, monitor")
def _():
    from app.engine.api import create_app
    with tempfile.TemporaryDirectory() as td:
        eng = _engine_parts(td)
        assert wait(lambda: eng.model_state == "ready", 5.0)
        eng.start()
        app = create_app(eng)
        c = app.test_client()
        try:
            s = c.get("/v1/status").get_json()
            assert s["model"]["state"] == "ready" and s["anchors"]["count"] == 8 and s["renderer"]["renderer"] == "fake"
            tree = c.get("/v1/params").get_json()
            assert tree["events"]["density"]["range"] == [0.0, 1.0]
            r = c.patch("/v1/params/events/density", json={"value": 0.9}).get_json()
            assert r == {"path": "/events/density", "value": 0.9}
            assert c.patch("/v1/params/events/density", json={"normalized": 0.5}).get_json()["value"] == 0.25
            assert c.patch("/v1/params/nope", json={"value": 1}).status_code == 404
            m = c.put("/v1/params/events/density/modulator", json={"type": "lfo", "rate": 0.1, "depth": 0.2}).get_json()
            assert m["type"] == "lfo" and c.get("/v1/params/events/density").get_json()["modulator"]["type"] == "lfo"
            bl = c.put("/v1/blend", json={"blend": {"bazaar": 2, "metro": 2}}).get_json()
            assert bl["bazaar"] == 1.0 and bl["metro"] == 1.0 and bl["park"] == 0.0
            sc = c.get("/v1/scenes").get_json()
            assert len(sc["scenes"]) == 10 and sc["sequence"][0] == "bazaar_midday"
            a = c.post("/v1/scenes/night_metro/activate", json={"transition_s": 0, "from_azimuth": 45}).get_json()
            assert a["activated"] == "night_metro"
            assert c.get("/v1/anchors").get_json()["blend"] == {"metro": 0.5, "late_night": 0.5}
            n = c.post("/v1/scenes", json={"id": "test", "blend": {"park": 1}, "events": {"birds": {"probability": 1}}, "save": True})
            assert n.status_code == 201 and (eng.root / "scenes" / "custom.yaml").exists()
            assert c.delete("/v1/scenes/test").get_json()["deleted"] is True
            an = c.post("/v1/anchors", json={"id": "bridge", "fields": {"place": "bridge", "sources": ["wind"]}})
            assert an.status_code == 201 and eng.bank.get("bridge").encoded and eng.params.exists("/scene/blend/bridge")
            assert c.delete("/v1/anchors/bridge").get_json()["deleted"] is True
            # one-shot render
            rr = c.post("/v1/render", json={"blend": {"bazaar": 0.7, "late_night": 0.3}, "duration_s": 2, "seed": 42}).get_json()
            jid = rr["job_id"]
            assert wait(lambda: c.get(f"/v1/render/{jid}").get_json()["state"] == "done", 10.0)
            rec = c.get(f"/v1/render/{jid}").get_json()
            assert rec["clips"][0]["seed"] == 42 and rec["clips"][0]["duration"] == 2.0
            cid = rec["clips"][0]["id"]
            wav = c.get(f"/v1/clips/{cid}")
            assert wav.status_code == 200 and wav.data[:4] == b"RIFF"
            assert c.get(f"/v1/clips/{cid}?info=1").get_json()["role"] == "oneshot"
            assert c.get("/v1/clips?role=oneshot").get_json()["clips"][0]["id"] == cid
            # transport and record
            assert c.post("/v1/transport/start").get_json()["playing"] is True
            rp = c.post("/v1/record/start").get_json()
            assert rp["recording"] is True
            time.sleep(0.3)
            rs = c.post("/v1/record/stop").get_json()
            assert rs["recording"] is False and Path(rs["path"]).exists()
            assert c.post("/v1/transport/stop").get_json()["playing"] is False
            # events trigger + presets + output
            assert "played" in c.post("/v1/events/trigger", json={"type": "machinery"}).get_json()
            p = c.post("/v1/presets", json={"name": "Night test"}).get_json()
            assert p["scene"] == "night_metro" and c.get("/v1/presets").get_json()["presets"][0]["name"] == "Night test"
            c.patch("/v1/params/events/density", json={"value": 0.1})
            assert c.post("/v1/presets/Night test/load").get_json()["restored"] > 10
            assert c.get("/v1/params/events/density").get_json()["value"] == 0.2   # night_metro's value, saved in the preset
            assert c.delete("/v1/presets/Night test").get_json()["deleted"] is True
            o = c.get("/v1/output").get_json()
            assert o["layout_file"]["decoder"] == "stereo" and o["sink_status"]["sink"] == "null"
            o2 = c.put("/v1/output", json={"layout": "8ch", "sink": "null"}).get_json()
            assert o2["channels"] == 8 and any("not built" in n for n in eng.notes)
            # model endpoint
            assert c.get("/v1/model").get_json()["state"] == "ready"
            # SSE: first bytes carry the hello event and a param change
            with app.test_request_context():
                pass
            resp = c.get("/v1/stream", buffered=False)
            it = resp.response
            first = next(it)
            assert b"event: hello" in first
            c.patch("/v1/params/master/gain", json={"value": 0.5})
            got = b""
            for _ in range(20):
                chunk = next(it)
                got += chunk
                if b"event: param" in got and b"/master/gain" in got:
                    break
            assert b"/master/gain" in got
            resp.close()
            mon = c.get("/v1/monitor.wav", buffered=False)
            mit = mon.response
            head = next(mit)
            assert head[:4] == b"RIFF" and head[8:12] == b"WAVE" and len(head) == 44
            block = next(mit)
            assert len(block) == BLOCK * 4
            mon.close()
            assert c.post("/v1/shutdown").get_json()["stopping"] is True
        finally:
            eng.stop()


@check("backend: /api/engine routes manage a subprocess")
def _():
    from flask import Flask
    from app.backend import engine_routes as er
    app = Flask("t")
    app.register_blueprint(er.engine_bp)
    c = app.test_client()
    st = c.get("/api/engine/status").get_json()
    assert st["available"] is True and st["running"] is False
    os.environ["FRAGMENTA_DOCKER"] = "1"
    try:
        assert c.post("/api/engine/start", json={}).status_code == 403
    finally:
        os.environ.pop("FRAGMENTA_DOCKER")
    # a real start on a free port with the fake renderer, then stop
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as td:
        os.environ["FRAGMENTA_ENGINE_TEST_DATA"] = td
        r = c.post("/api/engine/start", json={"renderer": "fake", "sink": "null", "port": port, "osc_port": 0, "no_resume": True}).get_json()
        try:
            assert r["running"] is True and r["port"] == port, r
            assert wait(lambda: er._ping(r["url"]), 20.0), Path(r["log"]).read_text()[-2000:]
            assert c.get("/api/engine/status").get_json()["api_up"] is True
        finally:
            r2 = c.post("/api/engine/stop").get_json()
        assert r2["running"] is False and r2["stopped"] is True
        assert c.post("/api/engine/stop").get_json()["stopped"] is False


def main() -> int:
    failed = 0
    t0 = time.time()
    for name, fn in CHECKS:
        t = time.time()
        try:
            fn()
            print(f"ok    {name}  ({time.time() - t:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed in {time.time() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
