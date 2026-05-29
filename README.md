<div align="center">

# Fragmenta (beta)

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/MAz-Codes/fragmenta/releases)
[![Docker](https://img.shields.io/badge/Docker_Hub-mazcode%2Ffragmenta-2496ED.svg?logo=docker)](https://hub.docker.com/r/mazcode/fragmenta)
[![Website](https://img.shields.io/badge/website-Fragmenta-purple.svg)](https://www.misaghazimi.com/fragmenta)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

![Header Image](app/frontend/public/fragmenta_background.png)

**Open-source text-to-audio LoRA training, generation, editing and performance for musicians.**

</div>

Fragmenta brings GenAI audio generation to musicians, offering intuitive LoRA training, generation, audio editing, and live performance capabilities — powered by **Stable Audio 3**.

> **Version 0.2.0 runs Stable Audio 3 only.** The previous Stable Audio Open (SA2) engine has been removed; if you need it, use the [`v0.1.x-legacy`](https://github.com/MAz-Codes/fragmenta/releases) tag. SA3 LoRAs and fine-tunes are not compatible with SA2.

This is not commercial software for creating high-fidelity songs or samples. Fragmenta is an open-source pipeline created to facilitate the integration of personalised GenAI technology within the musical workflow for musicians and composers — no coding or machine learning knowledge required. It is therefore more suitable for experimental music and sonic arts applications. This approach corresponds to my [Phd Research](https://www.misaghazimi.com) philosophy that seeks artist-first approaches in AI technology.

---

## Features

- **Desktop app** with a lightweight `pywebview` window and a pre-built React frontend — no Node.js or npm required
- **Docker images** for GPU and CPU — run as a web app on any machine
- **Bulk auto-annotation** — generate text prompts for your audio files via DSP analysis (Basic) or AI tagging with LAION-CLAP (Rich), with optional user-defined vocabulary
- **Project-aware LoRA training** with configurable rank, steps, learning rate, batch size, checkpoint frequency, and precision — trains directly on a Dataset Workbench project
- **LoRA adapters** — train DoRA/BoRA/-XS adapters on top of a frozen `*-base` checkpoint for consumer GPUs; stack up to 4 at once with per-slot strength, bypass, and reorder at generation time
- **Text-to-audio generation** — variable-length clips (up to 120s small / 380s medium), with CFG scale, inference steps, seed control, and a multi-LoRA stack
- **Audio editing (Edit tab)** — style transfer (audio-to-audio), region inpainting, and clip extension/continuation
- **Seamless loops** — bars-mode clips are tempo-locked and the loop seam is smoothed by inpainting, not a crossfade
- **Checkpoint Manager** — pick and download individual SA3 checkpoints (Small Music/SFX, Medium, and the matching `*-base` models) with per-item progress and hardware-compatibility hints
- **Performance Mode** — a 4-channel sampler designed for live performance:
  - Independent channel processing (gain, pan, low-pass filter, delay, reverb)
  - Master output with peak metering (dBFS)
  - **Bars-mode generation** — request clips by bar count; output is beat-aligned and tempo-locked to the master BPM
  - **Launch quantization** with an internal beat clock (works standalone or in sync with **Ableton Link**)
  - **Persistent session** — every panel setting survives page reloads and app restarts
  - **Named presets** — save and recall full panel snapshots
  - **MIDI learn** — assign any hardware control to any UI element; mappings persist
- **Real-time GPU memory monitoring**

![Interface](app/frontend/public/interface.png)

---

## Before You Start

| | Detail |
|---|---|
| **NVIDIA GPU** | Inference is fast (~3 s for 10 s of audio) |
| **Apple Silicon** | Works, but slow (~9 min for 10 s of audio on M1) |
| **Models** | The Checkpoint Manager guides you through downloading. Small Music/SFX run on CPU, Apple Silicon, or any GPU; Medium needs an NVIDIA GPU with Flash Attention 2 (not available on Windows). LoRA training runs against the matching `*-base` checkpoint. |
| **Offline** | After initial setup, everything runs locally — your data stays on your device |
| **Installation** | Fully isolated. Deleting the folder removes everything (except Python 3.11 if auto-installed) |
| **Python** | **Python 3.11 required** for local installs (Options 3) — [download here](https://www.python.org/downloads/release/python-3119/). Newer versions (3.12, 3.13) will fail to install dependencies. Not needed for the Docker option. |

---

## Option 1: Run on Hugging Face 🤗

If you only want to get to know the pipeline, Fragmenta runs on a [Hugging Face Space](https://huggingface.co/spaces/MazCodes/fragmenta) on CPUs. No coding or installation needed, but generation is very slow. Limited GPU-accelerated sessions are available — please get in touch and I can turn them on for you.

## Option 2: Run Locally with Docker

The fastest way to get started locally — no Python installation needed. Use the commands below, then open **http://localhost:5001** in your browser. 

### GPU (NVIDIA)

**Linux:**
```bash
docker run -d -p 5001:5001 --gpus all \
  -v ./models:/app/models \
  -v ./output:/app/output \
  -v ./config:/app/config \
  mazcode/fragmenta:gpu
```

**Windows (PowerShell):**
```powershell
docker run -d -p 5001:5001 --gpus all `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/output:/app/output `
  -v ${PWD}/config:/app/config `
  mazcode/fragmenta:gpu
```

### CPU (Mac / Linux / Windows — no GPU required)

> Audio generation is significantly slower on CPU. The `-v` volume mounts make sure downloaded models and generated audio persist across container restarts.

**Mac / Linux:**
```bash
docker run -d -p 5001:5001 \
  -v ./models:/app/models \
  -v ./output:/app/output \
  -v ./config:/app/config \
  mazcode/fragmenta:cpu
```

**Windows (PowerShell):**
```powershell
docker run -d -p 5001:5001 `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/output:/app/output `
  -v ${PWD}/config:/app/config `
  mazcode/fragmenta:cpu
```

---

## Option 3: Run the App Locally

> **Requires Python 3.11.** > - **Download:** [python.org → Python 3.11.9](https://www.python.org/downloads/release/python-3119/) (on Windows, tick **"Add python.exe to PATH"**).
> - **macOS:** the installer auto-installs via Homebrew if Python 3.11 is missing.
> - **Linux:** the installer auto-installs via `apt` / `dnf`. On Ubuntu 24.04+ it adds the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) (since `python3.11` was dropped from default repos). Arch users need `python311` from the AUR.

```bash
git clone https://github.com/MAz-Codes/fragmenta.git
cd fragmenta
```

Run the installer for your platform:

| Platform | Command |
|---|---|
| **Linux** | `./fragmenta.sh` |
| **macOS** | `fragmenta.command` |
| **Windows** | `./fragmenta.bat` |

The installer verifies Python 3.11 is available, sets up a virtual environment, installs all dependencies, and launches Fragmenta. The first run takes a few minutes; subsequent launches are faster — re-run the same script to start the app.

---

## Usage

### 0. Download Models & Authenticate

The app guides you through downloading models and authenticating with HuggingFace on first launch.

> **HuggingFace Token Requirement:** the Stable Audio 3 models are gated. You must:
> 1. Accept the license on the model page (while logged into your HF account)
> 2. Use a token with **Read access to public gated repositories** enabled
>
> Classic "read" tokens have this by default. Fine-grained tokens require you to **explicitly enable "Read access to public gated repositories"** under repository permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 1. Process Audio Files

Three ways to build the dataset, depending on what you're starting with:

#### a) Upload audio + prompts in the app

Upload audio files with text descriptions directly. The system saves audio and creates training metadata automatically.

#### b) Import CSV + audio folder

If you already have a CSV of `file_name,prompt` rows and a folder of matching audio, use the **Import CSV + Audio Folder** panel. Pick the CSV and the folder, get a preview with conflict status (new / replace / skip) against the existing metadata, then commit. Choose **Copy files into data/** to stage them, or **Leave files in place** to reference their original paths.

#### c) Auto-Annotate (optional)

If you have a folder of audio files without text descriptions, the **Bulk Annotate** panel can generate prompts for you. Point it at a folder and choose a tier:

| Tier | How it works | Requirements |
|---|---|---|
| **Basic** | DSP analysis via librosa — extracts tempo (BPM), musical key, brightness, and melodic/percussive character | No download, CPU only |
| **Rich** | Everything in Basic, plus zero-shot genre, mood, and instrument tagging using [LAION-CLAP](https://github.com/LAION-AI/CLAP) | ~2.35 GB one-time model download, GPU recommended |

**User-defined vocabulary (Rich tier):** customize the CLAP tagger with your own terms for results tailored to your style.

Once annotation finishes, results appear in an editable table — review and tweak any prompt before saving. Choose **Copy files into data/** to move everything into the dataset folder, or **Leave files in place** to reference their original paths.

### 2. Train Model

Training produces a **LoRA adapter** against a `*-base` checkpoint (`sa3-small-music-base`, `sa3-small-sfx-base`, or `sa3-medium-base`). The distilled post-trained models are inference-only and can't be trained against — the Training tab filters the base-model picker accordingly.

| Base | VRAM (LoRA) | When to pick it |
|---|---|---|
| `sa3-small-*-base` | ~2.5 GB (≈2 GB with `-xs`) | Laptop / consumer GPU; fast iteration |
| `sa3-medium-base` | ~6.5 GB (≈5.5 GB with `bf16`) | Best fidelity; needs an NVIDIA GPU |

Pick a project from the Dataset Workbench, choose an **adapter type** (`dora-rows` is the recommended default; `bora` and the compact `-xs` variants are also available), set rank/steps/learning rate, and Start. Optionally **pre-encode latents** first for a 5–10× per-step speedup. Training runs in the background with a live loss chart; each `step_<n>.safetensors` checkpoint appears in the LoRA picker as it's written.

### 3. Generate Audio

Use base or fine-tuned models to generate audio from text prompts (1–11 s for the small model, 1–47 s for the 1.0 model). If you've trained a LoRA, a LoRA picker appears when you select a compatible base model — pick one and dial in a **multiplier** (0–2×) to control how strongly the adapter biases the output.

**Settings:**
- **CFG scale** — how closely the model follows your prompt (higher = stricter)
- **Inference steps** — diffusion steps per generation. More = higher quality, proportionally slower. Locked to 8 for the distilled small model.
- **Seed** — control randomness; same seed produces identical results
- **Batch generation** — generate multiple variations at once

### 4. Performance Mode

Toggle Performance Mode from the sidebar to load a 4-channel sampler designed for live use. Each channel has its own prompt, generated clip, transport, and effect chain.

**Per-channel controls:**
- Text prompt and Generate
- Gain, pan, low-pass filter, delay, reverb
- Mute, solo, loop toggle
- Waveform display and per-channel level meter

**Master controls:**
- Master fader with peak-hold dBFS metering
- Master tempo (BPM) — drives bars-mode generation and launch quantization
- Launch quantization (`Q`) — snap launches to bar/beat boundaries (None, 1/32 → 8 Bars)
- **Ableton Link** toggle — sync tempo with any Link-enabled app on the local network

**Bars-mode generation:**
Switch a channel from `sec` to `bars` and pick a bar length (1–16). The clip is rendered to that bar count at the master BPM, then automatically beat-aligned and tempo-locked so it loops cleanly on the grid. The first beat lands at the start of the file; subsequent loops stay on the bar.

**Launch quantization:**
With `Q` set to anything other than None, channel launches snap to the next quantum on the bar grid. Works standalone via an internal beat clock, or in sync with Ableton Link if enabled. The first launch (when nothing is playing) fires immediately and anchors the transport; subsequent launches snap.

**Sessions and presets:**
- Every panel setting (BPM, knobs, prompts, model choice, etc.) auto-persists, so reloads and tab switches don't lose state.
- Click the floppy-disk icon in the toolbar to save the current configuration as a named preset, or load a saved one. **Restore defaults** wipes the session and MIDI mappings if you want to start clean.

**MIDI:**
Click **MIDI** in the toolbar to enter learn mode, then click any UI element and move a hardware knob/button to bind it. Mappings persist across sessions.

**Other:**
- **Auto-BPM prompt injection** — the master tempo is appended to each prompt automatically (skipped if your prompt already mentions one). Toggle from the bottom bar.
- **Shared inference steps and seed** with the Generation page.

---

## Project Structure

```
fragmenta/
├── app/
│   ├── backend/            # Flask API server
│   ├── frontend/
│   │   ├── build/          # Pre-built React app (served by Flask)
│   │   └── src/            # React source (only needed for development)
│   └── core/               # Core logic (generation, training, model management)
├── vendor/                 # Bundled third-party code
│   └── stable-audio-3/     # Stable Audio 3 (pinned snapshot; see UPSTREAM.md)
├── models/                 # Model configs and checkpoints
├── utils/                  # Utility modules
├── config/                 # Configuration files
├── fragmenta.sh            # Linux — first-time setup and subsequent launches
├── fragmenta.bat           # Windows — first-time setup and subsequent launches
├── fragmenta.command       # macOS — first-time setup and subsequent launches
└── start.py                # App entry point (called by launch scripts)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| pywebview / GTK error on Linux | Install WebKitGTK / GTK runtime packages, then rerun the installer |
| Flash-Attention won't install | Optional dependency — app works without it. Not available on Windows |
| GPU memory issues | Use the **Free GPU** button in the header, or reduce batch size |
| Performance panel feels stuck or its state is wrong | Open the preset menu and click **Restore defaults** |
| Import errors | Verify Python 3.11 is installed — newer versions are not supported. [Download Python 3.11.9](https://www.python.org/downloads/release/python-3119/). If you previously ran the installer with the wrong Python version, delete the `venv/` folder and rerun. |

---

## License

Copyright 2025-2026 Misagh Azimi

Licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE) for the full text.

**What this means in practice:**
- You can use, study, modify, and redistribute Fragmenta freely.
- If you distribute Fragmenta (modified or not), or **run a modified version as a hosted/network service**, you must release your source code under the same AGPL-3.0 license.
- You must keep the copyright notice and attribution.

If you want to use Fragmenta under different terms (e.g. embedded in a closed-source commercial product), please reach out — as the sole copyright holder I'm open to discussing alternative licensing.

### Third-Party Software

Fragmenta is **Powered by Stability AI**, using [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) models.

> "This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved"

Fragmenta includes and depends on various third-party open-source software. See [NOTICE.md](NOTICE.md) for complete attribution and license information.

- **pywebview** — BSD License
- **Stable Audio 3 Models** — governed by the [Stability AI Community License](vendor/stable-audio-3/LICENSE)
- **stable-audio-3** — vendored under `vendor/stable-audio-3/` (pinned snapshot; see `vendor/stable-audio-3/UPSTREAM.md`)
