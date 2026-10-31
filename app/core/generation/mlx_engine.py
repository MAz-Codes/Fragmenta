"""The MLX engine: Stability's MLX port of Stable Audio 3, for Apple Silicon.

No account needed, far less memory than PyTorch on a Mac, and it's what makes
Medium possible there (the PyTorch Medium needs CUDA). Weights come from
stabilityai/stable-audio-3-optimized through the Checkpoint Manager.

Wraps the library in vendor/stable-audio-3/optimized/mlx/models/defs. Its CLI in
scripts/ isn't shipped; the steps in _generate follow it.

MLX ties its work to the thread that started it, and Flask gives each request its
own thread, so every MLX call here runs on one worker thread (_call).
"""
import gc
import importlib
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.generation.engine import Capabilities, Engine, GenerationRequest, StepCallback
from app.core.generation.models import MODEL_INFO
from utils.logger import get_logger

logger = get_logger("AudioGenerator")

_MLX_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "stable-audio-3" / "optimized" / "mlx"
SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096  # PatchedPretransform downsample × SAME 16× expansion
MIN_SIGMA = 0.01           # the model is undefined near t=0 and gives NaN below this
APG_SCALE = 1.0            # the same default as the PyTorch sampler
# Below this, free each model after its last use, as upstream does by default.
# Otherwise keep them loaded: repeated generations are what Performance mode does.
KEEP_LOADED_MIN_RAM = 16 * 1024 ** 3

REPO_ID = "stabilityai/stable-audio-3-optimized"
TEXT_ENCODER = "t5gemma_f16.npz"
DITS = {
    "sm-music": ("models.defs.dit_mlx", "dit_sm-music_f16.npz", "same-s"),
    "sm-sfx": ("models.defs.dit_mlx", "dit_sm-sfx_f16.npz", "same-s"),
    "medium": ("models.defs.dit_mlx_medium", "dit_medium_f16.npz", "same-l"),
}
CODECS = {
    # module, weights; decode chunk + overlap; the encoder's length modulo
    "same-s": {"decoder": ("models.defs.same_s_decoder", "same_s_decoder_f32.npz"), "chunk": (8, 2),
               "encoder": ("models.defs.same_s_encoder", "same_s_encoder_f32.npz"), "modulo": 32},
    "same-l": {"decoder": ("models.defs.same_l_decoder", "same_l_decoder_f32.npz"), "chunk": (128, 8),
               "encoder": ("models.defs.same_l_encoder", "same_l_encoder_f32.npz"), "modulo": 16},
}


def weight_files(model_id: str) -> List[str]:
    """The files a model needs, for the Checkpoint Manager."""
    _module, dit_file, codec_name = DITS[MODEL_INFO[model_id][0]]
    codec = CODECS[codec_name]
    return [TEXT_ENCODER, dit_file, codec["decoder"][1], codec["encoder"][1]]


def _to_stereo_44k(audio) -> np.ndarray:
    sample_rate, samples = audio
    if sample_rate == SAMPLE_RATE:
        return samples
    import soxr
    return soxr.resample(samples.T, sample_rate, SAMPLE_RATE).T.astype(np.float32)


def _fit(samples: np.ndarray, length: int) -> np.ndarray:
    if samples.shape[-1] >= length:
        return samples[:, :length]
    return np.pad(samples, ((0, 0), (0, length - samples.shape[-1])))


def _keep_loaded() -> bool:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") >= KEEP_LOADED_MIN_RAM
    except (ValueError, OSError, AttributeError):
        return False


class MlxEngine(Engine):
    name = "mlx"
    capabilities = Capabilities(live_lora_strength=False)
    # chunked_decode and batch_size are acted on by what they promise: long clips
    # always decode in chunks here, which sounds the same, and it renders one clip.
    handled_fields = frozenset({
        "prompt", "model_id", "duration", "steps", "cfg_scale", "seed", "negative_prompt",
        "batch_size", "chunked_decode", "init_audio", "init_noise_level", "inpaint_audio",
        "inpaint_starts", "inpaint_ends",
    })

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._mx: Any = None
        self._parts: Dict[str, Any] = {}

    @classmethod
    def available(cls) -> bool:
        try:
            import mlx.core  # noqa: F401
            import sentencepiece  # noqa: F401
            return True
        except Exception:
            return False

    # --- the one thread MLX may run on ---------------------------------------------
    def _call(self, fn, *args):
        return self._worker.submit(fn, *args).result()

    def _module(self, name: str):
        if str(_MLX_ROOT) not in sys.path:
            sys.path.insert(0, str(_MLX_ROOT))
        return importlib.import_module(name)

    def _weights(self, filename: str) -> str:
        hub = self.config.get_path("models_pretrained") / "sa3" / "hub"
        snapshots = hub / ("models--" + REPO_ID.replace("/", "--")) / "snapshots"
        for found in sorted(snapshots.glob(f"*/MLX/{filename}")):
            if found.exists():
                return str(found)
        raise FileNotFoundError(f"{filename} isn't downloaded. Get this model from the Checkpoint Manager.")

    # --- Engine ---------------------------------------------------------------------
    def load(self, model_id: str, device: Optional[str] = None, half: bool = True) -> None:
        if model_id not in MODEL_INFO:
            raise ValueError(f"Unknown SA3 model_id: {model_id}")
        self._call(self._load, model_id)

    def _load(self, model_id: str) -> None:
        if self.model_id != model_id:
            self._drop()
        for filename in weight_files(model_id):
            self._weights(filename)
        import mlx.core as mx
        self._mx = mx
        self.model = self
        self.model_id = model_id
        self.device = "mlx"

    def unload(self) -> None:
        self._call(self._drop)

    def _drop(self) -> None:
        self._parts = {}
        self.model = None
        self.model_id = None
        self.loaded_loras = []
        self._free()

    def _free(self) -> None:
        gc.collect()
        if self._mx is not None:
            self._mx.clear_cache()

    def apply_loras(self, loras: List[Dict[str, Any]]) -> None:
        self._call(self._set_loras, [dict(l) for l in loras])

    def _set_loras(self, loras: List[Dict[str, Any]]) -> None:
        if loras != self.loaded_loras:
            self._parts.pop("dit", None)  # adapters are merged into the weights at load
        self.loaded_loras = loras

    def set_lora_strength(self, index: int, strength: float) -> bool:
        if index < 0 or index >= len(self.loaded_loras):
            return False
        return self._call(self._set_one_strength, index, float(strength))

    def _set_one_strength(self, index: int, strength: float) -> bool:
        self.loaded_loras[index]["strength"] = strength
        self._parts.pop("dit", None)  # takes effect on the next generation
        return True

    # --- the parts of a model, loaded when first needed --------------------------------
    def _text_encoder(self):
        if "text" not in self._parts:
            T5Gemma = self._module("models.defs.t5gemma_mlx").T5Gemma
            self._parts["text"] = T5Gemma.from_npz(self._weights(TEXT_ENCODER))
        return self._parts["text"]

    def _lora_specs(self):
        return [{"path": l["path"], "strength": float(l["strength"]), "steps": None}
                for l in self.loaded_loras] or None

    def _dit(self, T_lat: int, steps: int):
        if "dit" not in self._parts:
            module, filename, _codec = DITS[MODEL_INFO[self.model_id][0]]
            self._parts["dit"] = self._module(module).load_dit(
                self._weights(filename), T_lat=T_lat, dtype=self._mx.float16, compile_=False,
                lora_specs=self._lora_specs(), num_steps=steps, lora_log=logger.info,
            )
        return self._parts["dit"]

    def _codec(self) -> Dict[str, Any]:
        return CODECS[DITS[MODEL_INFO[self.model_id][0]][2]]

    def _decoder(self):
        if "decoder" not in self._parts:
            module, filename = self._codec()["decoder"]
            self._parts["decoder"] = self._module(module).load_model(
                weights_path=self._weights(filename), dtype=self._mx.float32, compile_=False)
        return self._parts["decoder"]

    def _encode_audio(self, audio, T_lat: int):
        """Audio → latents, always at fp32. The encoder only comes out for this."""
        mx = self._mx
        module, filename = self._codec()["encoder"]
        modulo = self._codec()["modulo"]
        encoder = self._module(module).load_model(weights_path=self._weights(filename), dtype=mx.float32,
                                                  compile_=False)
        # The encoder needs a length divisible by its modulo; the extra is trimmed off.
        grid = T_lat
        if (T_lat * 16) % modulo != 0:
            grid = math.ceil((T_lat * 16) / modulo) * modulo // 16
        samples = _fit(_to_stereo_44k(audio), grid * SAMPLES_PER_LATENT)[None, ...]
        B, C, T = samples.shape
        patches = samples.reshape(B, C, T // 256, 256).transpose(0, 1, 3, 2).reshape(B, C * 256, T // 256)
        latents = encoder(mx.array(patches))[..., :T_lat]
        mx.eval(latents)
        del encoder
        self._free()
        return latents.astype(mx.float16)

    # --- generation ----------------------------------------------------------------------
    def generate(self, request: GenerationRequest, on_step: StepCallback) -> np.ndarray:
        return self._call(self._generate, request, on_step)

    def _generate(self, request: GenerationRequest, on_step: StepCallback) -> np.ndarray:
        mx = self._mx
        pipeline = self._module("models.defs.sa3_pipeline")
        dtype = mx.float16
        T_lat = max(1, math.ceil(request.duration * SAMPLE_RATE / SAMPLES_PER_LATENT))
        keep = _keep_loaded()

        # 1. Text and timing conditioning. Those weights are baked into the DiT file.
        encoder = self._text_encoder()
        embeds, attention = encoder.encode([request.prompt], max_len=256)
        mx.eval(embeds, attention)
        _module, dit_file, _codec = DITS[MODEL_INFO[self.model_id][0]]
        padding, seconds_embedder = pipeline.load_conditioner_from_npz(self._weights(dit_file), prefix="cond.")
        specs = self._lora_specs()
        if specs:
            lora_merge = self._module("models.defs.lora_merge")
            seconds_embedder.W, _count = lora_merge.apply_conditioner_lora(seconds_embedder.W, specs)
        padded = pipeline.apply_prompt_padding(embeds.astype(dtype), attention, padding.astype(dtype))
        seconds = seconds_embedder(request.duration).astype(dtype)
        cross = mx.concatenate([padded, seconds], axis=1)
        global_cond = seconds[:, 0, :]

        cfg = float(request.cfg_scale)
        null_cross = None
        if cfg != 1.0:
            if request.negative_prompt:
                negative, negative_attention = encoder.encode([request.negative_prompt], max_len=256)
                mx.eval(negative, negative_attention)
                negative = pipeline.apply_prompt_padding(negative.astype(dtype), negative_attention,
                                                         padding.astype(dtype))
                null_cross = mx.concatenate([negative, seconds], axis=1)
            else:
                null_cross = mx.zeros_like(cross)
            mx.eval(null_cross)
        mx.eval(cross, global_cond)
        del encoder
        if not keep:
            self._parts.pop("text", None)
            self._free()

        # 2. Input audio, for style transfer and inpainting.
        init_latents = self._encode_audio(request.init_audio, T_lat) if request.init_audio is not None else None
        local_add_cond = paste_back = None
        if request.inpaint_audio is not None:
            source = self._encode_audio(request.inpaint_audio, T_lat)
            # 1 = keep, 0 = regenerate. No regions means regenerate it all, as on PyTorch.
            keep_mask = np.ones((1, 1, T_lat), dtype=np.float32) if request.inpaint_starts else \
                np.zeros((1, 1, T_lat), dtype=np.float32)
            for start, end in zip(request.inpaint_starts or [], request.inpaint_ends or []):
                first = min(T_lat, max(0, int(round(start * SAMPLE_RATE / SAMPLES_PER_LATENT))))
                last = min(T_lat, max(first + 1, int(round(end * SAMPLE_RATE / SAMPLES_PER_LATENT))))
                keep_mask[:, :, first:last] = 0.0
            mask = mx.array(keep_mask)
            masked = source.astype(mx.float32) * mask
            local_add_cond = mx.concatenate([mask, masked], axis=1).transpose(0, 2, 1).astype(dtype)
            paste_back = (source, mask)

        # 3. Sampling.
        sigma_max = max(MIN_SIGMA, float(request.init_noise_level))
        dit = self._dit(T_lat, request.steps)
        sigmas = pipeline.build_pingpong_schedule(request.steps, sigma_max=sigma_max, use_logsnr_shift=True)
        seed = int(request.seed) % (2 ** 32)
        noise = mx.random.normal((1, 256, T_lat), dtype=dtype, key=mx.random.key(seed))
        if init_latents is not None:
            noise = init_latents * (1.0 - sigma_max) + noise * sigma_max
        mx.eval(noise)

        def velocity(x, t):
            if cfg == 1.0:
                return dit(x, t, cross, global_cond, local_add_cond=local_add_cond)
            return self._guided(dit, x, t, cross, null_cross, global_cond, local_add_cond, cfg)

        plan = getattr(dit, "_lora_plan", None)
        latents = pipeline.sample_flow_pingpong(
            velocity, noise, sigmas, seed=(seed + 1) % (2 ** 32), paste_back=paste_back,
            on_step=lambda done, _total: on_step({"i": done - 1}),
            before_step=plan.sync if plan else None,
        )
        mx.eval(latents)
        del dit, velocity
        if not keep:
            self._parts.pop("dit", None)
            self._free()

        # 4. Decoding.
        audio = self._decode(latents.astype(mx.float32), T_lat, pipeline)
        if not keep:
            self._parts.pop("decoder", None)
            self._free()
        return audio[..., :int(round(request.duration * SAMPLE_RATE))]

    def _guided(self, dit, x, t, cross, null_cross, global_cond, local_add_cond, cfg):
        """Classifier-free guidance with APG, in fp32: the projection cancels badly at fp16."""
        mx = self._mx
        batched = dit(mx.concatenate([x, x], axis=0), mx.concatenate([t, t], axis=0),
                      mx.concatenate([cross, null_cross], axis=0),
                      mx.concatenate([global_cond, global_cond], axis=0),
                      local_add_cond=None if local_add_cond is None else
                      mx.concatenate([local_add_cond, local_add_cond], axis=0))
        cond_v, uncond_v = mx.split(batched, 2, axis=0)
        sigma = t.reshape(-1, 1, 1).astype(mx.float32)
        x32 = x.astype(mx.float32)
        cond = x32 - cond_v.astype(mx.float32) * sigma
        diff = cond - (x32 - uncond_v.astype(mx.float32) * sigma)
        norm = mx.sqrt((cond * cond).sum(axis=(-2, -1), keepdims=True))
        unit = cond / mx.maximum(norm, 1e-8)
        orthogonal = diff - (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
        guided_diff = APG_SCALE * orthogonal + (1.0 - APG_SCALE) * diff
        guided = cond + (cfg - 1.0) * guided_diff
        return ((x32 - guided) / sigma).astype(x.dtype)

    def _decode(self, latents, T_lat: int, pipeline) -> np.ndarray:
        mx = self._mx
        decoder = self._decoder()
        module, _filename = self._codec()["decoder"]
        decode_chunked = self._module(module).decode_chunked
        chunk, overlap = self._codec()["chunk"]
        if T_lat > chunk + 2 * overlap:
            patches = decode_chunked(decoder, latents, chunk, overlap)
        elif T_lat % 2 == 0:
            patches = decoder(latents)
        elif T_lat > 6:
            patches = decode_chunked(decoder, latents, 2, 2)
        else:
            # Very short odd lengths: SAME-S needs an even length, so pad one latent and trim.
            even = mx.concatenate([latents, latents[..., -1:]], axis=-1)
            patches = decoder(even)[..., : T_lat * 16]
        mx.eval(patches)
        audio = pipeline.patched_decode(patches, patch_size=256, channels=2)
        mx.eval(audio)
        return np.array(audio.astype(mx.float32))[0]
