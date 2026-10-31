#!/usr/bin/env python
"""What every engine must do with a real model. Same checks for each engine.

Skips an engine that can't run here or whose models aren't downloaded. Loads
real models, so on a small box run it in its own memory-capped group. Run:
    venv/bin/python tests/smoke_test_engine_contract.py
"""
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.generation import audio_generator as G  # noqa: E402
from app.core.generation.engine import Capabilities  # noqa: E402
from app.core.model_manager import ModelManager  # noqa: E402

DURATION = 4.0
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def correlation(a, b):
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n].ravel(), b[:n].ravel())[0, 1])


def make_source(path):
    """Six seconds of steady tones, longer than any clip here, so every part of a
    render has something to be compared with."""
    t = np.arange(44100 * 6) / 44100
    tone = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 331 * t) \
        + 0.1 * np.sin(2 * np.pi * (440 + 30 * t) * t)
    audio = np.stack([tone, np.roll(tone, 50)], axis=1).astype(np.float32)
    sf.write(path, audio, 44100)
    return audio


class Runner:
    def __init__(self, generator, model_id, **common):
        self.generator, self.model_id, self.common = generator, model_id, common

    def __call__(self, **kw):
        options = {"prompt": "warm analog synth arpeggio", "duration": DURATION, "seed": 7,
                   "model_id": self.model_id, **self.common, **kw}
        prompt = options.pop("prompt")
        path = self.generator.generate_audio(prompt, **options)
        audio, rate = sf.read(path, always_2d=True, dtype="float32")
        return audio, rate


def distilled_checks(run, capabilities: Capabilities, source_path, src):
    audio, rate = run()
    check("44.1 kHz stereo", rate == 44100 and audio.shape[1] == 2, f"{rate} Hz, {audio.shape[1]} channels")
    check("the requested length", abs(len(audio) / rate - DURATION) < 0.05, f"{len(audio) / rate:.3f} s")
    check("not silence, not clipped solid", 0.005 < np.sqrt((audio ** 2).mean()) < 0.7,
          f"rms {np.sqrt((audio ** 2).mean()):.3f}")
    check("within full scale", np.abs(audio).max() <= 1.0)
    check("same seed, same audio", np.array_equal(audio, run()[0]))
    check("another seed, other audio", correlation(audio, run(seed=8)[0]) < 0.9)
    check("another prompt, other audio", correlation(audio, run(prompt="deep techno kick drum")[0]) < 0.9)
    check("a distilled model ignores cfg", np.array_equal(audio, run(cfg_scale=9.0)[0]))

    if capabilities.audio_to_audio:
        low, _ = run(init_audio_path=str(source_path), init_noise_level=0.5)
        high, _ = run(init_audio_path=str(source_path), init_noise_level=0.95)
        check("style transfer changes the audio", correlation(low, audio) < 0.9)
        near, far = correlation(low, src), correlation(high, src)
        check("less noise stays closer to the source", near > far, f"{near:.3f} vs {far:.3f}")

    if capabilities.inpaint:
        check_inpaint(run, capabilities, source_path, src)


def check_inpaint(run, capabilities, source_path, src):
    rate = 44100

    def similar(audio, start, end):
        return correlation(audio[int(start * rate):int(end * rate)], src[int(start * rate):int(end * rate)])

    one, _ = run(inpaint_audio_path=str(source_path), inpaint_starts=[1.5], inpaint_ends=[2.5])
    kept, changed = similar(one, 0.2, 1.2), similar(one, 1.7, 2.3)
    check("inpaint keeps what's outside the region", kept > 0.95, f"{kept:.3f}")
    check("inpaint replaces what's inside", changed < kept - 0.1, f"{changed:.3f}")

    if capabilities.multi_region_inpaint:
        two, _ = run(inpaint_audio_path=str(source_path), inpaint_starts=[0.5, 2.5], inpaint_ends=[1.2, 3.2])
        first, second, between = similar(two, 0.65, 1.05), similar(two, 2.65, 3.05), similar(two, 1.6, 2.1)
        check("the part between two regions is kept", between > 0.95, f"{between:.3f}")
        check("both regions change", max(first, second) < between - 0.1, f"{first:.3f}, {second:.3f}")


def base_checks(run, capabilities, lora):
    audio, _ = run()
    check("negative prompt changes the audio", not np.array_equal(audio, run(negative_prompt="vocals")[0]))
    check("cfg changes the audio", not np.array_equal(audio, run(cfg_scale=2.0)[0]))
    check("steps change the audio", not np.array_equal(audio, run(steps=6)[0]))

    if capabilities.loras and lora:
        spec = {"path": str(lora), "strength": 1.0}
        full, _ = run(loras=[spec])
        check("a LoRA changes the audio", not np.array_equal(full, audio), f"corr {correlation(full, audio):.3f}")
        half, _ = run(loras=[{**spec, "strength": 0.5}])
        check("LoRA strength matters", not np.array_equal(half, full))
        zero, _ = run(loras=[{**spec, "strength": 0.0}])
        check("strength 0 sounds like no LoRA", correlation(zero, audio) > 0.999, f"{correlation(zero, audio):.5f}")
        check("removing the LoRA gives the plain model back", np.array_equal(run()[0], audio))


def stop_checks(generator, run, out_dir):
    before = set(out_dir.glob("*.wav"))

    def stopper():
        deadline = time.time() + 60
        while G.get_generation_progress()["phase"] != "sampling" and time.time() < deadline:
            time.sleep(0.002)
        generator.request_stop()

    threading.Thread(target=stopper, daemon=True).start()
    try:
        run(steps=60)
        stopped = False
    except G.GenerationStopped:
        stopped = True
    check("stop ends the run", stopped)
    check("stop leaves no file", set(out_dir.glob("*.wav")) == before)
    check("runs fine after a stop", run()[0].shape[1] == 2)


def lora_for(model_id):
    import json
    runs = ROOT / "models" / "fine_tuned"
    for run_dir in sorted(runs.glob("*")) if runs.is_dir() else []:
        meta = run_dir / "training_metadata.json"
        checkpoints = sorted((run_dir / "checkpoints").glob("*.safetensors")) if run_dir.is_dir() else []
        if meta.exists() and checkpoints:
            if json.loads(meta.read_text(encoding="utf-8")).get("base_model") == model_id:
                return checkpoints[-1]
    return None


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        paths = {"models_pretrained": ROOT / "models/pretrained", "output": out_dir}
        config = SimpleNamespace(get_path=paths.__getitem__)
        manager = ModelManager(config)
        generator = G.AudioGenerator(config)
        source_path = Path(tmp) / "source" / "source.wav"
        source_path.parent.mkdir()
        src = make_source(source_path)

        for engine_name, distilled, base in G.ENGINE_TEST_MODELS:
            if not G.engine_available(engine_name):
                print(f"SKIP {engine_name}: can't run on this machine")
                continue
            capabilities = G.engine_capabilities(engine_name)
            print(f"\n{engine_name} engine")
            if distilled and manager.is_model_downloaded(distilled):
                print(f" {distilled}")
                run = Runner(generator, distilled)
                distilled_checks(run, capabilities, source_path, src)
                stop_checks(generator, Runner(generator, distilled), out_dir)
            else:
                print(f"SKIP {distilled}: not downloaded")
            if base and manager.is_model_downloaded(base):
                print(f" {base}")
                base_checks(Runner(generator, base, steps=10, cfg_scale=4.0), capabilities, lora_for(base))
            else:
                print(f"SKIP {base}: not downloaded")

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'All passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
