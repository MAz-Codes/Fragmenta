# Fragmenta — Text-to-Audio Generation & Fine-Tuning

[![GitHub](https://img.shields.io/badge/GitHub-MAz--Codes%2Ffragmenta-181717.svg?logo=github)](https://github.com/MAz-Codes/fragmenta)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Website](https://img.shields.io/badge/website-Fragmenta-purple.svg)](https://www.misaghazimi.com/fragmenta)

Open-source audio generation and fine-tuning for musicians, powered by [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0). No Python or Node.js installation required — just Docker.

## Tags

| Tag | Base | Size | Use case |
|-----|------|------|----------|
| `gpu` | `nvidia/cuda:12.8.0` | ~16 GB | NVIDIA GPU machines (fastest inference) |
| `cpu` | `python:3.11-slim` | ~10 GB | Any machine — Mac (M1/M2/M3), Linux, Windows |

## Quick Start

### GPU (NVIDIA)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker run -d -p 5001:5001 --gpus all \
  -v ./models:/app/models \
  -v ./output:/app/output \
  -v ./config:/app/config \
  --name fragmenta \
  mazcode/fragmenta:gpu
```

### CPU (Mac / Linux / Windows)

```bash
docker run -d -p 5001:5001 \
  -v ./models:/app/models \
  -v ./output:/app/output \
  -v ./config:/app/config \
  --name fragmenta \
  mazcode/fragmenta:cpu
```

Then open **http://localhost:5001** in your browser.

## What It Does

- **Generate audio from text prompts** (1–47 seconds) using Stable Audio Open models
- **Fine-tune models on your own audio** with an intuitive UI — no coding required
- **Process audio files** into training datasets automatically
- **Download models** directly from HuggingFace through the app

## Model Weights

Model weights are **not included** in the image — they're downloaded through the app on first launch. The UI guides you through:

1. Authenticating with HuggingFace
2. Choosing a model (Small 341M or Full 1.0 838M)
3. Downloading weights to the mounted `models/` volume

Downloaded models persist across container restarts via the volume mount.

> **HuggingFace Token Requirement:** The Stable Audio Open models are gated. You must:
> 1. Accept the license on the model page (while logged into your HF account)
> 2. Use a token with **Read access to public gated repositories** enabled
>
> Classic "read" tokens have this by default. Fine-grained tokens require you to **explicitly enable "Read access to public gated repositories"** under repository permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Volumes

| Mount | Purpose |
|-------|---------|
| `/app/models` | Model weights and configs — **required for persistence** |
| `/app/output` | Generated audio files |
| `/app/config` | App preferences (terms acceptance, etc.) |
| `/app/data` | Training datasets (only needed for fine-tuning) |
| `/app/logs` | Application logs |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `FLASK_PORT` | `5001` | Port the server listens on |
| `FRAGMENTA_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

```bash
# Example: run on a different port
docker run -d -p 8080:8080 -e FLASK_PORT=8080 \
  -v ./models:/app/models \
  -v ./output:/app/output \
  mazcode/fragmenta:cpu
```

## Health Check

```bash
curl http://localhost:5001/api/health
```

```json
{
  "status": "ok",
  "components_ready": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5080"
}
```

The container includes a built-in health check. Components (including GPU detection) initialize immediately in the background on startup — the health endpoint responds right away.

## Network Access

The server binds to `0.0.0.0`, so it's accessible from other devices on your network at `http://<your-ip>:5001`.

## Performance

| Hardware | ~Time for 10s audio |
|----------|-------------------|
| NVIDIA GPU (RTX series) | ~3 seconds |
| Apple M1 (CPU) | ~9 minutes |
| x86 CPU | ~10+ minutes |

GPU is strongly recommended for practical use. CPU mode works for testing and experimentation.

## Docker Compose

For building from source or using `docker-compose`:

```yaml
# docker-compose.yml (GPU)
services:
  fragmenta:
    image: mazcode/fragmenta:gpu
    ports:
      - "5001:5001"
    volumes:
      - ./models:/app/models
      - ./output:/app/output
      - ./config:/app/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

See the [GitHub repo](https://github.com/MAz-Codes/fragmenta) for full `docker-compose.yml` files and build-from-source instructions.

## Source Code

**GitHub:** [github.com/MAz-Codes/fragmenta](https://github.com/MAz-Codes/fragmenta)

## License

Apache License 2.0 — Copyright 2025-2026 Misagh Azimi
