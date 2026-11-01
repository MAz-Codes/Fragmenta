"""The shared generation layer, driven by a stand-in engine: no model, no GPU.

Everything here holds for every engine, because it lives outside them. What real
engines must do with a real model is in smoke_test_engine_contract.py.

Pure stdlib + the project venv; no pytest required. Run:
    venv/bin/python tests/test_engine_contract.py
"""
import dataclasses
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.generation import audio_generator as G  # noqa: E402
from app.core.generation.engine import Capabilities, Engine, GenerationRequest  # noqa: E402

SECOND = np.zeros((2, 44100), dtype=np.float32)


class FakeEngine(Engine):
    name = "fake"
    capabilities = Capabilities()
    handled_fields = frozenset(f.name for f in dataclasses.fields(GenerationRequest))

    def __init__(self, config, output=SECOND, steps_to_run=3, on_generate=None):
        super().__init__(config)
        self.output, self.steps_to_run, self.on_generate = output, steps_to_run, on_generate
        self.requests, self.unloads = [], 0

    def load(self, model_id, device=None, half=True):
        self.model, self.model_id, self.device = object(), model_id, device

    def unload(self):
        self.model, self.model_id, self.loaded_loras = None, None, []
        self.unloads += 1

    def apply_loras(self, loras):
        self.loaded_loras = list(loras)

    def set_lora_strength(self, index, strength):
        if index >= len(self.loaded_loras):
            return False
        self.loaded_loras[index]["strength"] = strength
        return True

    def generate(self, request, on_step):
        self.requests.append(request)
        for i in range(self.steps_to_run):
            on_step({"i": i})
        if self.on_generate:
            self.on_generate()
        return self.output


class Env:
    """A generator writing to a temp folder, with every model on the fake engine."""

    def __init__(self, **engine_kwargs):
        self.engine_kwargs = engine_kwargs

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        config = SimpleNamespace(get_path=lambda name: self.out)
        self.engine = FakeEngine(config, **self.engine_kwargs)
        self.patch = mock.patch.object(G, "_make_engine", return_value=self.engine)
        self.patch.start()
        G._reset_progress()
        self.gen = G.AudioGenerator(config)
        return self

    def __exit__(self, *exc):
        self.patch.stop()
        self.tmp.cleanup()

    def run(self, prompt="a prompt", **kw):
        kw.setdefault("model_id", "sa3-small-music")
        return self.gen.generate_audio(prompt, **kw)

    def request(self, **kw):
        self.run(**kw)
        return self.engine.requests[-1]


def pcm(path):
    import soundfile as sf
    return sf.read(path, dtype="int16", always_2d=True)


# --- engines have to keep up with the request -----------------------------------

def request_fields():
    return {f.name for f in dataclasses.fields(GenerationRequest)}


def test_every_engine_handles_every_request_field():
    from app.core.generation.mlx_engine import MlxEngine
    from app.core.generation.torch_engine import TorchEngine
    for engine in (TorchEngine, MlxEngine):
        missing = request_fields() - set(engine.handled_fields)
        unknown = set(engine.handled_fields) - request_fields()
        assert not missing and not unknown, (engine.__name__, missing, unknown)


def test_loras_cross_between_a_base_model_and_its_builds():
    from app.core.generation.models import engine_for, lora_family
    assert lora_family("sa3-small-music-base") == lora_family("sa3-small-music") == lora_family("sa3-mlx-small-music")
    assert lora_family("sa3-medium-base") == lora_family("sa3-mlx-medium")
    assert lora_family("sa3-small-music-base") != lora_family("sa3-small-sfx-base")
    assert engine_for("sa3-mlx-medium") == "mlx" and engine_for("sa3-medium") == "torch"


def test_mlx_bundles_match_what_the_engine_loads():
    from app.core.generation.mlx_engine import weight_files
    from app.core.generation.models import MODEL_INFO
    from app.core.model_manager import _SA3_CATALOG
    for model_id in MODEL_INFO:
        if "mlx" in model_id:
            assert sorted(weight_files(model_id)) == sorted(_SA3_CATALOG[model_id]["files"]), model_id
            assert MODEL_INFO[model_id][2] == _SA3_CATALOG[model_id]["max_duration_sec"], model_id


# --- what the request carries --------------------------------------------------------

def test_distilled_models_get_their_fixed_steps_and_cfg():
    with Env() as e:
        r = e.request(model_id="sa3-small-music", cfg_scale=9.0)
        assert r.steps == 8 and r.cfg_scale == 1.0


def test_base_models_default_to_50_steps_cfg_7_and_take_overrides():
    with Env() as e:
        r = e.request(model_id="sa3-small-music-base")
        assert r.steps == 50 and r.cfg_scale == 7.0
        r = e.request(model_id="sa3-small-music-base", steps=12, cfg_scale=4.5)
        assert r.steps == 12 and r.cfg_scale == 4.5


def test_duration_stays_within_the_models_limits():
    with Env() as e:
        assert e.request(duration=0.2).duration == 1.0
        assert e.request(duration=999).duration == 120.0
        assert e.request(model_id="sa3-medium", duration=999).duration == 380.0


def extend_kwargs(path, source_seconds, want, start=None):
    """An Extend request as the Edit panel sends it, on a source this long."""
    sf.write(path, np.zeros((int(source_seconds * 44100), 2), dtype="float32"), 44100)
    return dict(duration=want, inpaint_audio_path=str(path),
                inpaint_starts=[source_seconds if start is None else start], inpaint_ends=[want])


def test_extending_renders_past_the_end_and_trims_back():
    long_output = np.zeros((2, 44100 * 24), dtype=np.float32)
    with Env(output=long_output) as e:
        path = e.run(**extend_kwargs(e.out / "source.wav", 10.0, 14.0))
        r = e.engine.requests[-1]
        assert (r.duration, r.inpaint_starts, r.inpaint_ends) == (24.0, [10.0], [24.0])
        assert sf.info(path).frames == 44100 * 14


def test_extending_starts_where_the_source_fades():
    rng = np.random.default_rng(1)
    level = np.ones(44100 * 10, dtype="float32")
    level[44100 * 9:] = np.linspace(1, 0, 44100, dtype="float32") ** 3
    level[int(44100 * 9.5):] = 0
    faded = rng.normal(0, 0.2, (2, 44100 * 10)).astype("float32") * level
    plain = rng.normal(0, 0.2, (2, 44100 * 10)).astype("float32")
    quiet_tail = plain.copy()
    quiet_tail[:, 44100 * 2:] = 0  # silent for 8 s: never give up more than 3 s of a source
    assert 8.9 <= G._fade_start((44100, faded)) <= 9.5
    assert G._fade_start((44100, plain)) == 10.0
    assert G._fade_start((44100, quiet_tail)) == 7.0

    with Env() as e:
        path = e.out / "source.wav"
        sf.write(path, faded.T, 44100)
        r = e.request(duration=14.0, inpaint_audio_path=str(path), inpaint_starts=[10.0], inpaint_ends=[14.0])
        assert 8.9 <= r.inpaint_starts[0] <= 9.5


def test_extending_stops_at_the_models_limit():
    with Env() as e:
        r = e.request(**extend_kwargs(e.out / "source.wav", 100.0, 115.0))
        assert (r.duration, r.inpaint_ends) == (120.0, [120.0])


def test_inpainting_is_not_extending():
    with Env() as e:
        # A region inside the clip, and one running to the end of a clip no longer
        # than its source (the Edit panel can send an end past the clip's length).
        for start, end in ((2.5, 7.5), (2.5, 150.0)):
            kw = extend_kwargs(e.out / "source.wav", 10.0, end, start=start)
            kw["duration"] = 10.0
            r = e.request(**kw)
            assert (r.duration, r.inpaint_ends) == (10.0, [end])


def test_every_option_reaches_the_engine():
    fixture = str(ROOT / "tests/fixtures/mini-project/pad.wav")
    with Env() as e:
        r = e.request(prompt="p", duration=4, seed=9, negative_prompt="n", chunked_decode=True,
                      init_audio_path=fixture, init_noise_level=0.7, inpaint_audio_path=fixture,
                      inpaint_starts=[1.0, 2.0], inpaint_ends=[1.5, 2.5])
    assert (r.prompt, r.duration, r.seed, r.negative_prompt, r.chunked_decode) == ("p", 4.0, 9, "n", True)
    assert r.init_noise_level == 0.7 and r.inpaint_starts == [1.0, 2.0] and r.inpaint_ends == [1.5, 2.5]
    for sample_rate, samples in (r.init_audio, r.inpaint_audio):
        assert sample_rate > 0 and isinstance(samples, np.ndarray) and samples.shape[0] == 2


def test_input_audio_always_arrives_stereo():
    import torch
    for channels, expected in ((1, 2), (2, 2), (5, 2)):
        with mock.patch("app.core.audio.decode.read_audio",
                        return_value=(torch.zeros(channels, 10), 44100)):
            sample_rate, samples = G.AudioGenerator._load_audio("x.wav")
        assert sample_rate == 44100 and samples.shape == (expected, 10), channels


def test_an_empty_negative_prompt_is_no_negative_prompt():
    with Env() as e:
        assert e.request(negative_prompt="").negative_prompt is None


def test_unknown_model_is_refused():
    with Env() as e:
        try:
            e.run(model_id="nope")
        except ValueError as exc:
            assert "nope" in str(exc)
        else:
            raise AssertionError("no error")
        assert G.get_generation_progress()["is_generating"] is False


# --- the output file --------------------------------------------------------------------

def test_output_is_44k1_stereo_16_bit_at_the_right_level():
    audio = np.stack([np.full(44100, 0.5, dtype=np.float32), np.full(44100, -0.25, dtype=np.float32)])
    with Env(output=audio) as e:
        samples, rate = pcm(e.run())
    assert rate == 44100 and samples.shape == (44100, 2)
    assert samples[0, 0] == int(0.5 * 32767) and samples[0, 1] == int(-0.25 * 32767)


def test_too_loud_output_is_limited_and_broken_samples_become_silence():
    # MLX can reach 1.95 on some prompts, per the Mac spike.
    audio = np.array([[1.95, -1.95, np.nan, np.inf], [0.0, 0.0, 0.0, -np.inf]], dtype=np.float32)
    with Env(output=audio) as e:
        samples, _ = pcm(e.run())
    assert samples[:, 0].tolist() == [32767, -32767, 0, 32767]
    assert samples[:, 1].tolist() == [0, 0, 0, -32767]


def test_wrong_output_shape_fails_clearly():
    with Env(output=np.zeros((1, 2, 100), dtype=np.float32)) as e:
        try:
            e.run()
        except RuntimeError as exc:
            assert "shape" in str(exc)
        else:
            raise AssertionError("no error")
        assert G.get_generation_progress()["is_generating"] is False


def test_output_is_tagged_as_ai_generated_with_its_loras():
    with Env() as e, mock.patch.object(G, "generation_tags", wraps=G.generation_tags) as tags:
        e.run(loras=[{"path": "/x/a.safetensors", "strength": 1.0}])
    assert tags.call_args.args == ("sa3-small-music", ["/x/a.safetensors"])


def test_two_quick_generations_never_overwrite_each_other():
    with Env() as e, mock.patch.object(G.time, "strftime", return_value="20260101_000000"):
        assert e.run() != e.run()


# --- progress and stop ---------------------------------------------------------------------

def test_progress_follows_the_steps_and_ends_complete():
    seen = []

    def during():
        seen.append(G.get_generation_progress())

    with Env(on_generate=during, steps_to_run=4) as e:
        e.run(steps=4, model_id="sa3-small-music-base")
    assert seen[0]["phase"] == "sampling" and seen[0]["step"] == 4 and seen[0]["total_steps"] == 4
    end = G.get_generation_progress()
    assert end["phase"] == "complete" and end["is_generating"] is False and end["progress"] == 100


def test_stop_during_sampling_leaves_no_file():
    gen_holder = {}

    def stop_now():
        gen_holder["gen"].request_stop()

    class Stopping(FakeEngine):
        def generate(self, request, on_step):
            on_step({"i": 0})
            stop_now()
            on_step({"i": 1})
            raise AssertionError("the step callback should have stopped this")

    with Env() as e:
        gen_holder["gen"] = e.gen
        e.engine = Stopping(e.gen.config)
        with mock.patch.object(G, "_make_engine", return_value=e.engine):
            try:
                e.run()
            except G.GenerationStopped:
                pass
            else:
                raise AssertionError("no stop")
        assert not list(e.out.glob("*.wav"))
        p = G.get_generation_progress()
        assert p["phase"] == "idle" and p["is_generating"] is False


def test_stop_after_the_last_step_leaves_no_file():
    holder = {}
    with Env(on_generate=lambda: holder["gen"].request_stop()) as e:
        holder["gen"] = e.gen
        try:
            e.run()
        except G.GenerationStopped:
            pass
        else:
            raise AssertionError("no stop")
        assert not list(e.out.glob("*.wav"))


def test_a_stop_with_nothing_running_doesnt_hit_the_next_run():
    with Env() as e:
        assert e.gen.request_stop() is False
        assert e.run().exists()


def test_an_engine_failure_is_reported_and_clears_progress():
    def boom():
        raise RuntimeError("engine broke")

    with Env(on_generate=boom) as e:
        try:
            e.run()
        except RuntimeError:
            pass
        p = G.get_generation_progress()
        assert p["phase"] == "failed" and p["is_generating"] is False and "engine broke" in p["error"]


def test_a_failed_load_clears_progress():
    with Env() as e, mock.patch.object(e.engine, "load", side_effect=OSError("no weights")):
        try:
            e.run()
        except OSError:
            pass
        p = G.get_generation_progress()
        assert p["phase"] == "failed" and p["is_generating"] is False


def test_generations_queue_instead_of_overlapping():
    active, peak, lock = [0], [0], threading.Lock()

    def during():
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        threading.Event().wait(0.05)
        with lock:
            active[0] -= 1

    with Env(on_generate=during) as e:
        threads = [threading.Thread(target=e.run) for _ in range(3)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    assert peak[0] == 1


# --- engines, LoRAs, unload -----------------------------------------------------------------------

def test_switching_engines_unloads_the_other_one():
    """Engines share memory on a Mac, so only one may hold a model."""
    with tempfile.TemporaryDirectory() as tmp:
        config = SimpleNamespace(get_path=lambda name: Path(tmp))
        made = {}

        def make(name, _config):
            made[name] = FakeEngine(_config)
            return made[name]

        gen = G.AudioGenerator(config)
        with mock.patch.object(G, "_make_engine", side_effect=make), \
                mock.patch.object(G, "engine_for", side_effect=lambda m: "b" if m == "sa3-medium" else "a"):
            gen.generate_audio("p", model_id="sa3-small-music")
            assert made["a"].model is not None and gen._model_id == "sa3-small-music"
            gen.generate_audio("p", model_id="sa3-medium")
            assert made["a"].model is None and made["a"].unloads == 1 and made["b"].model is not None
            assert gen._model_id == "sa3-medium"
            gen.generate_audio("p", model_id="sa3-small-music")
            assert made["b"].model is None and made["a"].model is not None


def test_loras_are_applied_for_each_run():
    with Env() as e:
        e.run(loras=[{"path": "a", "strength": 0.5}])
        assert e.gen._loaded_loras == [{"path": "a", "strength": 0.5}]
        e.run()
        assert e.gen._loaded_loras == []


def test_live_lora_strength_reaches_the_engine():
    with Env() as e:
        assert e.gen.set_lora_strength(0, 0.5) is False  # nothing loaded yet
        e.run(loras=[{"path": "a", "strength": 1.0}])
        assert e.gen.set_lora_strength(0, 0.25) is True and e.gen._loaded_loras[0]["strength"] == 0.25
        assert e.gen.set_lora_strength(3, 0.25) is False


def test_unload_clears_the_model_but_not_during_a_generation():
    with Env() as e:
        e.run()
        assert e.gen.model is not None
        with e.gen._gen_lock:
            assert e.gen.unload_model() is False
        assert e.gen.unload_model() is True and e.gen.model is None and e.gen._model_id is None


def test_unload_with_nothing_loaded_is_fine():
    with Env() as e:
        assert e.gen.unload_model() is True and e.gen.model is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
