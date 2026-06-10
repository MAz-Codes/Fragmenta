# Third-Party Software Notices and Attributions

Fragmenta is licensed under the **GNU Affero General Public License v3.0** (see [`LICENSE`](LICENSE)). It bundles, depends on, and (at runtime) downloads third-party software and models. This document provides the required notices and attribution.

Two categories matter for licensing:

- **Redistributed** — shipped inside this repository and/or the Docker images (vendored code, the pre-built frontend, the flash-attn wheels, and the Python packages baked into the images).
- **Downloaded at runtime** — the model weights. These are **not** redistributed by Fragmenta; you download them yourself from Hugging Face and accept their license at that point.

---

## Included Third-Party Code (redistributed)

### Stable Audio 3 (vendored)
- **Source**: https://github.com/Stability-AI/stable-audio-3
- **Copyright**: Copyright (c) 2026 Stability AI
- **License**: MIT License
- **Location**: `vendor/stable-audio-3/` (pinned snapshot — see `vendor/stable-audio-3/UPSTREAM.md`)
- **Modifications**: Minor changes for integration with Fragmenta (CPU/MPS device gating, packaging).

The MIT License text is in [`vendor/stable-audio-3/LICENSE`](vendor/stable-audio-3/LICENSE).

### FlashAttention (flash-attn) — binary redistribution
- **Source**: https://github.com/Dao-AILab/flash-attention
- **Copyright**: Copyright (c) 2022, Tri Dao and contributors
- **License**: BSD-3-Clause
- **Form**: Redistributed as **binary wheels** referenced from `requirements.txt` — the pinned upstream Linux wheel, plus a Windows wheel built from the same BSD-licensed source (no official PyPI Windows wheel exists). Required by the `sa3-medium` model.

BSD-3-Clause permits binary redistribution provided the copyright notice, license text, and disclaimer are reproduced — satisfied by this entry and the license/AUTHORS files inside each wheel. This attribution does not imply endorsement of Fragmenta by the flash-attn authors.

---

## Included Third-Party Assets (redistributed)

### Convolution reverb impulse responses (Voxengo)
- **Files**: `app/frontend/public/ir/` (and the built copy in `app/frontend/build/ir/`) —
  `Scala Milan Opera Hall.wav`, `Nice Drum Room.wav`, `Narrow Bumpy Space.wav`.
- **Source**: Voxengo free Impulse Response pack — https://www.voxengo.com/impulses/
- **Terms**: Voxengo publishes these IR files for free use in commercial and
  non-commercial productions. They are used here for the performance-bus
  convolution reverb.
- **Action required before public release**: confirm Voxengo's current terms
  permit *redistribution* of the WAV files inside a software repository (their
  grant clearly covers *use* in productions; bundling the raw IRs in a
  distributed app should be double-checked). If redistribution is not clearly
  permitted, replace them with CC0 / explicitly-redistributable IRs or fetch
  them at runtime instead of committing them.

---

## Model Weights (downloaded at runtime — NOT redistributed)

Fragmenta orchestrates the download and use of the following models. They are fetched from Hugging Face under the user's own account, and the user accepts each model's license at download time. Fragmenta does **not** ship these weights.

### Stable Audio 3 model family — Stability AI Community License
- **Models**: `stabilityai/stable-audio-3-small-music`, `-small-sfx`, `-medium`, and the matching `-base` checkpoints; `stabilityai/SAME-S` and `stabilityai/SAME-L` (autoencoders).
- **License**: **Stability AI Community License** — https://stability.ai/community-license-agreement
- **Bundled text encoder**: these checkpoints include a T5Gemma text encoder redistributed under the **Gemma Terms of Use** (https://ai.google.dev/gemma/terms), including the Section 3.2 use restrictions.

Required attribution (per the Community License):

> "This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved"

**Powered by Stability AI** ([stability.ai](https://stability.ai)).

Notes on the Community License obligations:
- Use is permitted for individuals/organizations with **up to USD $1,000,000** in annual revenue; above that threshold an Enterprise license from Stability AI is required.
- Use must comply with Stability AI's Acceptable Use Policy.
- **LoRA adapters trained in Fragmenta** (on the `*-base` checkpoints) are *Derivative Works* under the same Community License. If you distribute one, distribute it under that license, retain the attribution notice above, and do **not** brand it with Stability AI names/marks (e.g. "Stable Audio 3").

### LAION-CLAP tagger (Rich auto-annotation)
- **Checkpoint**: `lukewys/laion_clap` — `music_audioset_epoch_15_esc_90.14.pt`
- **License**: **CC0-1.0** (public-domain dedication) — https://github.com/LAION-AI/CLAP
- At construction, LAION-CLAP also fetches text-encoder snapshots (`roberta-base`, `bert-base-uncased`, `facebook/bart-base`) from Hugging Face. These are downloaded at runtime, not redistributed.

---

## Python Dependencies

Installed from PyPI and baked into the Docker images. All licenses below are GPL/AGPL-compatible. See each package's distribution for full license text.

| Package | License |
|---|---|
| torch, torchvision | BSD-3-Clause |
| torchaudio | BSD-2-Clause |
| transformers, huggingface-hub, safetensors, accelerate | Apache-2.0 |
| laion-clap *(Rich auto-annotation; the checkpoint it loads is attributed above)* | CC0-1.0 |
| pytorch-lightning | Apache-2.0 |
| ftfy | Apache-2.0 |
| requests | Apache-2.0 |
| Flask | BSD-3-Clause |
| Flask-CORS | MIT |
| einops, einops-exts | MIT |
| torchlibrosa | MIT |
| braceexpand | MIT |
| python-rtmidi | MIT |
| setuptools | MIT |
| dill | BSD-3-Clause |
| soundfile | BSD-3-Clause |
| scipy | BSD-3-Clause |
| numpy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| h5py | BSD-3-Clause |
| webdataset | BSD-3-Clause |
| psutil | BSD-3-Clause |
| omegaconf | BSD-3-Clause |
| antlr4-python3-runtime *(via omegaconf)* | BSD-3-Clause |
| click | BSD-3-Clause |
| python-dotenv | BSD-3-Clause |
| pywebview | BSD-3-Clause |
| proxy-tools *(via pywebview)* | MIT |
| numba | BSD-2-Clause |
| librosa | ISC |
| tqdm | MPL-2.0 AND MIT |
| matplotlib | Matplotlib License (PSF-based, BSD-style) |
| Pillow | HPND (PIL/MIT-CMU) |
| wget | Public Domain |
| Pycairo *(Linux)* | LGPL-2.1-only OR MPL-1.1 |
| PyGObject *(Linux)* | LGPL-2.1-or-later |
| flash-attn | BSD-3-Clause (see above) |

---

## JavaScript / Frontend Dependencies

Bundled into the pre-built React app shipped at `app/frontend/build/`.

| Package | License |
|---|---|
| react, react-dom | MIT |
| @mui/material | MIT |
| @emotion/react, @emotion/styled | MIT |
| lucide-react | ISC |
| vite, @vitejs/plugin-react *(build only)* | MIT |

---

## License Texts

- **This project**: [`LICENSE`](LICENSE) (AGPL-3.0).
- **Vendored Stable Audio 3 code**: [`vendor/stable-audio-3/LICENSE`](vendor/stable-audio-3/LICENSE) (MIT).
- **Stable Audio 3 weights**: Stability AI Community License — https://stability.ai/community-license-agreement
- **Python / npm packages**: see each package's metadata or upstream repository.

---

**Last Updated**: 2026-06-10

For licensing questions, open an issue on the project repository or contact the maintainer.
