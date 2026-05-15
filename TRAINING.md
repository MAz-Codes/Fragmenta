# Training Guide

Fragmenta supports two ways of teaching a model your sound: **full fine-tuning** and **LoRA adapters**. They produce different artifacts and have different hardware floors. This guide covers when to pick each, how to drive the UI, and what the knobs actually do.

---

## Full Fine-tune vs. LoRA

| | Full Fine-tune | LoRA |
|---|---|---|
| Output artifact | New full-model `.safetensors` (~1.6 GB small / ~4.6 GB large) | LoRA delta `.ckpt` (~60 MB) layered on top of the base |
| VRAM (Stable Audio Open 1.0) | ≥ 24 GB | ~12 GB |
| VRAM (Stable Audio Open Small) | ≥ 12 GB | ~6 GB |
| Inference cost | Same as the base model | Same as the base model (LoRA runs alongside) |
| Switching styles at inference | Reload the whole model | Swap the LoRA in the picker |
| Best for | Strongest possible style imprint, sufficient GPU | Consumer GPUs, or maintaining several "flavors" without duplicating the base |

If you can fit a full fine-tune in VRAM, it'll generally produce a stronger imprint than a LoRA on the same dataset. If you can't, the LoRA path is what gets you a custom model on consumer hardware.

---

## Training Through the UI

1. **Data Processing tab** — upload your audio with text prompts (or auto-annotate). The dataset folder needs at least `batch_size` clips and a populated `metadata.json` mapping `file_name → prompt`.
2. **Training tab** —
   - Pick a base model (Stable Audio Open Small or 1.0).
   - Pick a mode: **Full fine-tune** or **LoRA**.
   - Set epochs, batch size, learning rate, checkpoint frequency, precision.
   - For LoRA mode, set rank / alpha / dropout / multiplier (see below).
   - Start training. Progress, GPU usage, and the loss curve update live.
3. When training finishes, the artifact appears automatically in the relevant inference picker.

---

## Hyperparameters Reference

### Shared

- **Epochs** — number of passes through the dataset. For small/personal datasets (< 200 clips) plan for 20–50 epochs.
- **Batch size** — per-step batch. Higher = smoother gradients, more VRAM. Defaults to 4. Reduce to 1 if you're tight on memory.
- **Learning rate** — `1e-4` is the safe default. Push to `2e-4` – `5e-4` on small datasets if you want a stronger imprint. Lower (`5e-5`) if you see instability.
- **Checkpoint frequency** — how often to write a checkpoint. "Auto" picks a sensible cadence based on total steps. Manual is in train-steps.
- **Precision** — `auto` selects fp16/bf16 based on your GPU's capabilities. fp32 trades VRAM for stability.

### LoRA-only

- **Rank** — capacity of the adapter. Higher = more expressive, more VRAM, larger checkpoint. `16` is the default; `32` is a reasonable step up. Beyond `64` the cost stops being worth it on a single-style adapter.
- **Alpha** — scaling factor applied to the LoRA delta during training. Conventional choice: `alpha = rank`. Setting `alpha > rank` makes the LoRA "louder" during training; can help with weak imprint but risks overfitting.
- **Dropout** — regularization on the LoRA layers. Leave at `0` for personal-scale datasets; bump to `0.05`–`0.1` only if you see clear overfitting.
- **Multiplier** — saved into the checkpoint metadata and used as the default strength at inference time. The user can always override it from the inference picker (0–2×).

---

## Inference: Applying What You Trained

### Full fine-tune
1. Generation tab → pick your fine-tuned model from the dropdown.
2. The "Select Checkpoint" dropdown appears below. Pick an **unwrapped** checkpoint (the app provides an Unwrap button after training).
3. Generate.

### LoRA
1. Generation tab → pick a **base model** (the same base the LoRA was trained against — the picker filters automatically).
2. A "No LoRA / …" dropdown appears below the model picker. Pick your LoRA.
3. A **multiplier slider** (0–2×) appears. `1.0×` is what was used during training; lower walks back toward the base, higher exaggerates the adapter.
4. Generate. The first generation after switching LoRAs triggers a brief `torch.compile` recompile (~5–15 s); subsequent calls are full-speed.

The same picker is also available on the Performance tab, in a compact form sitting next to the model dropdown.

---

## Reading the Loss Curve

Diffusion training loss is inherently noisy — each step samples a random denoising timestep with very different difficulty, so the per-step loss bounces a lot even when training is going well. Fragmenta's loss chart shows both the raw values (faint) and an EMA-smoothed trend (bold) so the underlying direction stays readable.

Don't panic if the raw curve looks flat. The honest validation for an audio generative model is **listening**, not the loss number. After training, generate with the same prompt + seed both with and without your LoRA (or with your fine-tuned vs base model) and trust your ears.

---

## Common Pitfalls

- **"Checkpoints Saved: 0" after a successful LoRA run.** Fixed in the monitor as of v0.1.1 — LoRA checkpoints are saved nested under `models/fine_tuned/<name>/<run-id>/checkpoints/`. The monitor recursively counts them now.
- **LoRA picker doesn't show your trained adapter.** Make sure the selected base model in the picker matches the base the LoRA was trained against. The picker filters out incompatible LoRAs to prevent undefined behavior.
- **LoRA at multiplier=2.0 sounds distorted.** That's expected — anything above ~1.5× starts pulling the activations out of the base model's well-conditioned regime. Use the high end to confirm the adapter is actually doing work, not as a normal operating point.
- **Loss looks flat for the whole run.** Likely a small dataset relative to model capacity. Options: more epochs, higher LR (`2e-4`–`5e-4`), higher rank (LoRA), or smaller base model.
