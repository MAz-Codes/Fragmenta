# Third-Party Software Notices and Attributions

Fragmenta Desktop uses third-party software and libraries. This document provides the required notices and attribution information for these dependencies.

---

## Included Third-Party Code

### 1. Stable Audio Tools
- **Source**: https://github.com/Stability-AI/stable-audio-tools
- **Copyright**: Copyright (c) 2023 Stability AI
- **License**: MIT License
- **Location**: `stable-audio-tools/`
- **Modifications**: Minor modifications for integration with Fragmenta Desktop

The MIT License text can be found in `stable-audio-tools/LICENSE`.

Stable Audio Tools itself includes code from the following projects (see `stable-audio-tools/LICENSES/`):

#### 1.1 ADP (Audio Diffusion PyTorch)
- **Copyright**: Copyright (c) 2022 archinet.ai
- **License**: MIT License
- **License File**: `stable-audio-tools/LICENSES/LICENSE_ADP.txt`

#### 1.2 AEIOU
- **Copyright**: Copyright (c) 2022 AEIOU team
- **License**: MIT License
- **License File**: `stable-audio-tools/LICENSES/LICENSE_AEIOU.txt`

#### 1.3 Auraloss
- **Copyright**: Copyright (c) Christian Steinmetz
- **License**: Apache License 2.0
- **License File**: `stable-audio-tools/LICENSES/LICENSE_AURALOSS.txt`

#### 1.4 Descript Audio Codec
- **Copyright**: Copyright (c) Descript
- **License**: MIT License
- **License File**: `stable-audio-tools/LICENSES/LICENSE_DESCRIPT.txt`

#### 1.5 Meta AudioCraft
- **Copyright**: Copyright (c) Meta Platforms, Inc.
- **License**: MIT License
- **License File**: `stable-audio-tools/LICENSES/LICENSE_META.txt`

#### 1.6 NVIDIA NeMo
- **Copyright**: Copyright (c) NVIDIA Corporation
- **License**: Apache License 2.0
- **License File**: `stable-audio-tools/LICENSES/LICENSE_NVIDIA.txt`

#### 1.7 x-transformers
- **Copyright**: Copyright (c) Phil Wang
- **License**: MIT License
- **License File**: `stable-audio-tools/LICENSES/LICENSE_XTRANSFORMERS.txt`

---

## Python Dependencies

The following Python packages are distributed with Fragmenta Desktop:

### Deep Learning Frameworks
- **PyTorch** - BSD-3-Clause License - https://pytorch.org/
- **TorchVision** - BSD License - https://github.com/pytorch/vision
- **TorchAudio** - BSD-2-Clause License - https://github.com/pytorch/audio
- **Transformers** - Apache License 2.0 - https://github.com/huggingface/transformers
- **Diffusers** - Apache License 2.0 - https://github.com/huggingface/diffusers
- **Accelerate** - Apache License 2.0 - https://github.com/huggingface/accelerate
- **PEFT** - Apache License 2.0 - https://github.com/huggingface/peft
- **Datasets** - Apache License 2.0 - https://github.com/huggingface/datasets

### Audio Processing
- **librosa** - ISC License - https://github.com/librosa/librosa
- **soundfile** - BSD-3-Clause License - https://github.com/bastibe/python-soundfile
- **scipy** - BSD-3-Clause License - https://github.com/scipy/scipy

### Web Framework & API
- **Flask** - BSD-3-Clause License - https://github.com/pallets/flask
- **Flask-CORS** - MIT License - https://github.com/corydolphin/flask-cors
- **FastAPI** - MIT License - https://github.com/tiangolo/fastapi
- **uvicorn** - BSD-3-Clause License - https://github.com/encode/uvicorn
- **requests** - Apache License 2.0 - https://github.com/psf/requests

### User Interface
- **PyQt6** - GPL v3 / Commercial License - https://www.riverbankcomputing.com/software/pyqt/
- **PyQt6-WebEngine** - GPL v3 / Commercial License - https://www.riverbankcomputing.com/software/pyqt/
- **Gradio** - Apache License 2.0 - https://github.com/gradio-app/gradio

### Utilities
- **numpy** - BSD License - https://numpy.org/
- **matplotlib** - PSF-based License - https://matplotlib.org/
- **plotly** - MIT License - https://github.com/plotly/plotly.py
- **tqdm** - MIT/MPL-2.0 License - https://github.com/tqdm/tqdm
- **psutil** - BSD-3-Clause License - https://github.com/giampaolo/psutil
- **wandb** - MIT License - https://github.com/wandb/wandb
- **omegaconf** - BSD-3-Clause License - https://github.com/omry/omegaconf
- **click** - BSD-3-Clause License - https://github.com/pallets/click
- **python-dotenv** - BSD-3-Clause License - https://github.com/theskumar/python-dotenv
- **Pillow** - PIL License (PIL/Pillow License) - https://github.com/python-pillow/Pillow
- **huggingface-hub** - Apache License 2.0 - https://github.com/huggingface/huggingface_hub

### Development Tools
- **pytest** - MIT License - https://github.com/pytest-dev/pytest
- **black** - MIT License - https://github.com/psf/black
- **isort** - MIT License - https://github.com/PyCQA/isort
- **flake8** - MIT License - https://github.com/PyCQA/flake8

---

## JavaScript/React Dependencies

The following npm packages are included in the React frontend:

### Core Framework
- **react** (^18.2.0) - MIT License - https://github.com/facebook/react
- **react-dom** (^18.2.0) - MIT License - https://github.com/facebook/react
- **react-scripts** (5.0.1) - MIT License - https://github.com/facebook/create-react-app

### UI Components & Styling
- **@mui/material** (^5.14.0) - MIT License - https://github.com/mui/material-ui
- **@mui/icons-material** (^5.14.0) - MIT License - https://github.com/mui/material-ui
- **@emotion/react** (^11.11.0) - MIT License - https://github.com/emotion-js/emotion
- **@emotion/styled** (^11.11.0) - MIT License - https://github.com/emotion-js/emotion

### Utilities & Components
- **axios** (^1.6.0) - MIT License - https://github.com/axios/axios
- **react-dropzone** (^14.2.3) - MIT License - https://github.com/react-dropzone/react-dropzone
- **react-player** (^2.13.0) - MIT License - https://github.com/cookpete/react-player
- **recharts** (^2.8.0) - MIT License - https://github.com/recharts/recharts

---

## Additional Notices

### Flash-Attention (Optional Dependency)
- **flash-attn** - BSD-3-Clause License - https://github.com/Dao-AILab/flash-attention
- Note: This is an optional dependency for performance optimization. The application functions without it.

### PyQt6 Licensing Note
Fragmenta Desktop uses PyQt6, which is dual-licensed under GPL v3 and a commercial license. This application is distributed under the Apache License 2.0 for its own code, and PyQt6 is used under the GPL v3 license. Users who wish to use this software in a manner incompatible with GPL v3 should obtain a commercial PyQt6 license from Riverbank Computing.

### Stable Audio Models
The pre-trained Stable Audio models are subject to their own license terms from Stability AI. Please review the model license at:
- https://huggingface.co/stabilityai/stable-audio-open-1.0

Users must accept the model license terms independently when downloading the models through Fragmenta Desktop.

---

## License Texts

Full license texts for all dependencies can be obtained from their respective repositories or package distributions. For licenses of included code (stable-audio-tools), see the `stable-audio-tools/LICENSES/` directory.

---

## How to View Full License Information

- **This project's license**: See `LICENSE` file in the root directory
- **Stable Audio Tools licenses**: See `stable-audio-tools/LICENSE` and `stable-audio-tools/LICENSES/`
- **Python package licenses**: Visit the URLs provided above or check package metadata
- **npm package licenses**: Visit the URLs provided above or run `npm list --depth=0` in `app/frontend/`

---

**Last Updated**: October 22, 2025

For questions about licensing, please contact the Fragmenta Desktop project maintainers or open an issue on the project repository.
