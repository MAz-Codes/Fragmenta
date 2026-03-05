<div align="center">

# Fragmenta Desktop

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/version-0.0.1-green.svg)](https://github.com/MAz-Codes/fragmenta/releases)
[![Docker](https://img.shields.io/badge/Docker_Hub-mazcode%2Ffragmenta-2496ED.svg?logo=docker)](https://hub.docker.com/r/mazcode/fragmenta)
[![Website](https://img.shields.io/badge/website-Fragmenta-purple.svg)](https://www.misaghazimi.com/fragmenta)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

![Header Image](app/frontend/public/fragmenta_background.png)

**Open-source text-to-audio fine-tuning and generation for musicians.**

</div>

Fragmenta brings GenAI audio generation to musicians, offering intuitive fine-tuning and generation capabilities powered by [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) models. Think of it as a ComfyUI for text-to-audio. You can use it as a user-friendly interface for inference, but fine-tuning is what's made easy here to personalize the models.

This is not commercial software for creating high-fidelity songs or samples. Fragmenta is an open-source pipeline created to facilitate the integration of personalized GenAI technology within the musical workflow for musicians and composers — no coding or machine learning knowledge required. It is therefore more suitable for experimental music and sonic arts applications. This approach corresponds to my [Bending the Algorithm](https://www.misaghazimi.com) philosophy that seeks artist-first approaches in AI technology.

---

## Features

- **Desktop app** with PyQt6 interface and embedded React frontend
- **Docker support** — run as a web app on any machine (GPU or CPU)
- Audio file processing with automatic dataset creation
- Guided model download and HuggingFace authorization
- Model fine-tuning with LoRA, checkpoint saving and unwrapping
- Text-to-audio generation (1–47 seconds)
- Real-time GPU memory monitoring

---

## Before You Start

| | Detail |
|---|---|
| **NVIDIA GPU** | Inference is fast (~3s for 10s audio) |
| **Apple Silicon** | Works, but slow (~9m for 10s audio on M1) |
| **Models** | The app guides you through downloading; use the larger model for more coherent results |
| **Offline** | After initial setup, everything runs locally — your data stays on your device |
| **Installation** | Fully isolated. Deleting the folder removes everything (except Python 3.11 and Node.js if auto-installed) |

---
## Option 1: Run Online on Hugging Face 🤗

Fragmenta runs on a [Hugging Face Space](https://huggingface.co/spaces/MazCodes/fragmenta) on CPUs. No coding or installations necessary. Limited GPU accelerated sessions are available, please get in touch for me to turn these on for you. 

## Option 2: Run Locally with Docker

This is the fastest way to get started locally — no Python or Node.js installation needed. Pull the image from [Docker Hub](https://hub.docker.com/r/mazcode/fragmenta/tags) or use the commands below.

### GPU (NVIDIA)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker run -d -p 5001:5001 --gpus all \
  -v ./models:/app/models \
  -v ./output:/app/output \
  -v ./config:/app/config \
  mazcode/fragmenta:gpu
```

Open **http://localhost:5001** in your browser.

### CPU (Mac / Linux / Windows — no GPU required)

```bash
docker run -d -p 5001:5001 \
  -v ./models:/app/models \
  -v ./output:/app/output \
  -v ./config:/app/config \
  mazcode/fragmenta:cpu
```

> **Note:** Audio generation is significantly slower on CPU.

The `-v` volume mounts ensure downloaded models and generated audio persist across container restarts.


---

## Option 2: Run Locally

```bash
git clone https://github.com/MAz-Codes/fragmenta.git
cd Fragmenta
```

```bash
./run.sh           # Linux
./run.bat          # Windows
./run.command      # macOS
```

The launcher script will install all dependencies (Python 3.11 venv, PyTorch, Node packages) and launch the app. This takes a while on first run.

### Running After Initial Installation

```bash
cd Fragmenta
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
python main.py
```

---

## Usage

### 0. Download Models & Authenticate

The app guides you through downloading models and authenticating with HuggingFace. You don't need both models. If you don't have an NVIDIA GPU, the large model is **not recommended**.

### 1. Process Audio Files

Upload audio files with text descriptions. The system saves audio and creates training metadata automatically.

### 2. Train Model

Configure training parameters:

- **Base model:** [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) Small (341M) or 1.0 (838M)
- **Epochs**, **learning rate**, **checkpoint frequency**

> **Important:** You must unwrap your trained model before using it for generation. This can be done on the generation page.

### 3. Generate Audio

Use base or fine-tuned models to generate audio from text prompts (1–47 seconds). After training, select your checkpoint from the dropdown, unwrap it, and generate.

---

## Project Structure

```
fragmenta/
├── app/
│   ├── backend/        # Flask API server
│   ├── frontend/       # React interface
│   ├── core/           # Core logic (generation, training)
│   └── desktop/        # PyQt6 desktop wrapper
├── stable-audio-tools/ # Stable Audio library (modified)
├── models/             # Model configs and checkpoints
├── utils/              # Utility modules
├── config/             # Configuration files
├── Dockerfile          # GPU Docker image
├── Dockerfile.cpu      # CPU Docker image
├── Dockerfile.hf       # Hugging Face Spaces image
└── main.py             # Application entry point
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Models don't show after download | Restart the application |
| Qt platform plugin error | Setup script auto-installs Qt libraries |
| Flash-Attention won't install | Optional dependency — app works without it. Not available on Windows |
| GPU memory issues | Use the "Free GPU" button or reduce batch size |
| Import errors | Verify Python 3.11 is installed |

---

## License

Copyright 2025-2026 Misagh Azimi

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

### Third-Party Software

Fragmenta includes and depends on various third-party open-source software. See [NOTICE.md](NOTICE.md) for complete attribution and license information.

- **PyQt6** — GPL v3
- **Stable Audio Models** — Subject to Stability AI's model license (review when downloading)
- **stable-audio-tools** — MIT License (included with modifications)
