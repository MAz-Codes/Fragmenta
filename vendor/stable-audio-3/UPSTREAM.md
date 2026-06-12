# Stable Audio 3 — vendored snapshot

- Upstream: https://github.com/Stability-AI/stable-audio-3
- Pinned commit: fa5ee841dd49bae0fa361fac26904adc27fd400e
- Pinned at:     2026-05-21T09:38:08-07:00
- Vendored on:   2026-05-21T11:55:06-07:00

Spiked locally on misagh's machine (RTX 5080, cu128 swap, CPU + GPU paths
both confirmed working with small-music) before vendoring.

To bump: re-clone upstream, diff against this tree, update this file.

## Local patches (re-apply or upstream when bumping)

- `stable_audio_3/models/lora/model.py` — `apply_lora()` removal branch:
  iterate `list(layer.parametrizations.keys())` instead of the live
  ModuleDict. `remove_parametrizations()` mutates that dict mid-loop, so
  `remove_lora()`/`merge_lora()` raised "dictionary changed size during
  iteration" on every parametrized layer (callers only saw it as a
  warning + full-model-reload fallback). 2026-06-11.
