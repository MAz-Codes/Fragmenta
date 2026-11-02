# Stable Audio 3 — vendored snapshot

- Upstream: https://github.com/Stability-AI/stable-audio-3
- Pinned commit: 77e0398c26f7d22d801ec1bb036307e8b231f741 (upstream `main`)
- Pinned at:     2026-10-11T03:32:19-04:00
- Vendored on:   2026-10-27 (bumped from `fa5ee84`, 2026-05-21; 202 commits)

Not vendored: `optimized/tensorRT` and `optimized/tflite` (out of scope).
`optimized/mlx/scripts/` is on disk but ignored by Fragmenta's `.gitignore`,
so it isn't tracked or shipped; Fragmenta uses `optimized/mlx/models/defs/`.

To bump: follow the procedure in the master plan (Part 4): 3-way merge every
file below against the new upstream, replace the tree, overlay, re-run tests.
Every local edit carries a `# Fragmenta:` comment in the code.

## Local patches

- `stable_audio_3/model_configs.py`
  - New `localize_conditioners()`: load a T5Gemma conditioner from the
    checkpoint's own snapshot when it ships the subfolder (every `*-base` repo
    does, byte-identical to the gated sibling it points at), so base models
    need no Hugging Face token. 2026-10-22.
  - The standalone-autoencoder fallbacks (`_small_stable_audio_3`,
    `_medium_stable_audio_3`) also probe the `*-base` checkpoints, which hold
    the identical autoencoder (pretransform tensors and config checked equal).
    Without it, pre-encoding offline with only base models cached failed
    looking for `stabilityai/SAME-S` / `SAME-L`. 2026-10-27.
- `stable_audio_3/model.py` — `StableAudioModel.from_pretrained` calls
  `localize_conditioners()` after `json.load`. 2026-10-22.
- `scripts/train_lora.py`
  - `load_model` calls `localize_conditioners()` (token-free base models).
  - Apple Silicon: picks MPS when there's no CUDA; keeps the frozen base in
    fp32 and trains at `32-true` there (bf16 on MPS mismatches dtypes and its
    autocast is unreliable); skips the base-precision downcast on MPS.
  - Spawn-safe DataLoader workers (Windows/macOS): a top-level
    `_worker_seed_init` via `functools.partial` instead of a lambda, and
    `caption_metadata_fn` imports `Path` locally.
  - `persistent_workers` on the training DataLoader: spawned workers otherwise
    re-import torch at the start of every epoch (2026-10-30).
  - `volume_norm=True` on the dataset (−16 LUFS, matching pre-encode).
  - `CSVLogger(flush_logs_every_n_steps=1)` so the UI's loss curve is live.
  - `--no_demo` flag to skip demo rendering entirely.
- `scripts/pre_encode_dataset.py` — `caption_metadata_fn` imports `Path`
  locally (spawn-based workers); `volume_norm=True` so cached latents match
  training.
  - Encodes each clip at its own length (`pad=args.pad or batch_size > 1`)
    instead of padding every clip to `--sample_size` first: a 30 s clip cost
    a 285 s encode (SAME-S on CPU 16 s → 1.7 s; SAME-L on GPU 3.7 → 0.6 s).
    Latents match the padded encode within CPU/GPU noise except the last few
    frames. 2026-10-27.
  - `--chunked`: `ae.encode(..., chunked=True)` for the clips and the
    silence latent, so SAME-L without CUDA stays at ~7.3 GB for any length
    (the unchunked 285 s silence encode alone was OOM-killed on 30 GB).
    Fragmenta passes it for SAME-L when CUDA is unavailable. 2026-10-27.
- `stable_audio_3/models/transformer.py` — use flash-attn only for CUDA
  tensors; CPU/MPS fall through to the SDPA path instead of crashing when
  flash-attn is installed.
- `stable_audio_3/training/utils.py` — `wandb` is optional (import guarded).
- File I/O without torchaudio (torchaudio 2.9+ dropped `load`/`save`).
  2026-10-27.
  - New `stable_audio_3/data/audio_io.py`: `read_audio` (soundfile, PyAV
    fallback for m4a/AAC, optional early stop), `write_audio` (int16 → 16-bit
    PCM, else 32-bit float, as `torchaudio.save` did), `AUDIO_EXTENSIONS`
    (the app's list: adds `.m4a`, `.aac`, `.aiff`).
  - `stable_audio_3/data/dataset.py`: `load_file` uses `read_audio`;
    `get_audio_filenames` defaults to `AUDIO_EXTENSIONS`; `__getitem__` of
    `SampleDataset` and `PreEncodedDataset` retries a bad sample at most 32
    times (`_RETRY` / `_with_bounded_retries`) instead of recursing without
    limit, then raises a clear error.
  - `cli.py`, `interface/diffusion_cond.py` (gradio), `training/diffusion.py`
    (demo callbacks): `torchaudio.load`/`save` → `read_audio`/`write_audio`.

- Partial (init audio) sampling schedules. 2026-10-27.
  - `stable_audio_3/inference/sampling.py`: `build_schedule` shifted
    `linspace(sigma_max, 0)` and then forced its first value back to
    `sigma_max`, so with `LogSNRShift` the second value was higher (0.9 →
    0.995, 0.5 → 0.88). The first step ran backwards (base models with CFG
    clipped heavily: 23k–47k samples at 0.9) and the run restarted from
    near-full noise, so `init_noise_level` above ~0.6 ignored the source. Now
    the linear schedule starts where the shift maps to `sigma_max`
    (`_shifted_partial_schedule`, bisection, per-element rows), so it starts
    at `sigma_max` and only descends. Full schedules are unchanged.
  - `optimized/mlx/models/defs/sa3_pipeline.py`: same fix in
    `build_pingpong_schedule` (closed-form inverse of its LogSNR shift);
    matches the torch schedule to 2e-7.

- `scripts/pre_encode_dataset.py`, `scripts/train_lora.py`: `--filelist`
  (passed to `LocalDatasetConfig.filelist_path`) so Fragmenta can hand over
  only the clips with prompts. Without it, SA3 swaps a clip with no `.txt`
  for a random other clip (duplicate latents) and trains one with an empty
  `.txt` on an empty prompt. 2026-10-27.

- Long files read only the crop window. 2026-10-29.
  `stable_audio_3/data/dataset.py` (`SampleDataset._long_file_window`,
  `load_file(offset, duration)`) and `data/audio_io.py` (`read_audio(offset=)`,
  `audio_length`): a file longer than `sample_size` is decoded and
  loudness-normalised for the window it contributes only, not in full. A
  33-minute track peaked at 15 GB of RAM for a 285 s crop and got the
  pre-encode OOM-killed; now 2.7 GB. `seconds_start` / `seconds_total` /
  `timestamps` are still computed against the whole file. Loudness is now
  measured on the window, not the whole track.

## Dropped at the 2026-10-27 bump

- `stable_audio_3/models/lora/model.py` — the `apply_lora()` removal-branch
  fix (iterate `list(layer.parametrizations.keys())`) is now upstream, so the
  upstream file is used as is.
