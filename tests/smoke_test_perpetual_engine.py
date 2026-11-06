"""The perpetual engine on the real SFX base model (plan E1.11).

Loads sa3-small-sfx-base, encodes the shipped anchors, renders a one-shot
from a two-anchor blend, a bed, a batch of events and a five-point blend
sweep at a fixed seed, and prints the timings for the plan's Part 5 table.
WAVs land in output/perpetual_smoke/. Needs the model on disk and a GPU.

    venv/bin/python tests/smoke_test_perpetual_engine.py [--model sa3-small-sfx-base] [--quick]
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def wait(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sa3-small-sfx-base")
    ap.add_argument("--quick", action="store_true", help="shorter clips")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    from app.core.config import get_config
    from app.engine.engine import Engine
    from app.engine.renderer import RenderJob

    out_dir = ROOT / "output" / "perpetual_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    td = tempfile.mkdtemp(prefix="perpetual_smoke_")
    t0 = time.perf_counter()
    eng = Engine(Path(td), scene="places", renderer="torch", sink="null", osc_port=0, osc_out=None,
                 model_id=args.model, config=get_config(), autoload=True, resume=False)
    assert wait(lambda: eng.model_state in ("ready", "error"), 600), "model load timed out"
    assert eng.model_state == "ready", eng.model_error
    t_load = time.perf_counter() - t0
    rows.append(("model load + anchor encode (8 anchors, 42 prompts)", t_load))
    print(f"loaded {args.model} + encoded anchors in {t_load:.1f} s; {eng.renderer.device_stats()}")
    eng.renderer.start()

    done = {}
    eng.renderer.on_done(lambda job, clips, err: done.__setitem__(job.id, (clips, err)))

    def render(label, job, timeout=900):
        t = time.perf_counter()
        eng.renderer.submit(job)
        assert wait(lambda: job.id in done, timeout), f"{label}: timed out"
        clips, err = done[job.id]
        dt = time.perf_counter() - t
        assert not err, f"{label}: {err}"
        rows.append((label, dt))
        for i, c in enumerate(clips):
            dst = out_dir / f"{label.replace(' ', '_').replace('/', '-')}_{i}.wav"
            shutil.copy2(c.path, dst)
        print(f"{label}: {dt:.1f} s → {len(clips)} clip(s), first {clips[0].duration} s, {clips[0].analysis.get('band')}, {clips[0].analysis.get('loudness_db')} dB")
        return clips

    # anchor re-encode timing (one prompt)
    t = time.perf_counter()
    eng.renderer.encoder.encode(["TrackType: SFX, Type: ambience, Place: bazaar"])
    rows.append(("anchor encode, one prompt", time.perf_counter() - t))

    blend = {"bazaar": 0.5, "metro": 0.5}
    render("oneshot 30s bazaar+metro" if not args.quick else "oneshot 10s bazaar+metro",
           RenderJob(role="oneshot", blend=blend, overrides={"type": "ambience"}, duration_s=10 if args.quick else 30, seed=42, steps=args.steps))
    render("bed 90s bazaar" if not args.quick else "bed 30s bazaar",
           RenderJob(role="bed", blend={"bazaar": 1.0}, overrides={"type": "ambience"}, duration_s=30 if args.quick else 90, seed=7, steps=args.steps))
    render("events 4s x4 scooters",
           RenderJob(role="event", blend={"street": 1.0}, overrides={"type": "event", "perspective": "close", "sources": ["scooters"]},
                     duration_s=4, batch_size=4, seed=99, steps=args.steps, event_type="scooters"))
    # blend sweep bazaar → late_night at a fixed seed
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        render(f"sweep {w:.2f} late_night",
               RenderJob(role="oneshot", blend={"bazaar": 1 - w, "late_night": w}, overrides={"type": "ambience"},
                         duration_s=8, seed=1234, steps=args.steps))
    # plain prompt, no anchors, for reference
    render("prompt only train", RenderJob(role="oneshot", prompt="TrackType: SFX, chugging train coming into station with horn", duration_s=7, seed=3, steps=args.steps))

    eng.stop()
    print("\n| Measurement | Seconds |\n|---|---|")
    for label, dt in rows:
        print(f"| {label} | {dt:.1f} |")
    (out_dir / "timings.json").write_text(json.dumps({"model": args.model, "steps": args.steps, "rows": rows,
                                                      "device": eng.renderer.device_stats()}, indent=1))
    print(f"\nclips in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
