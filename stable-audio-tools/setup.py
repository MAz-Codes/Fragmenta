from setuptools import setup, find_packages

setup(
    name='stable-audio-tools',
    version='0.0.19',
    url='https://github.com/Stability-AI/stable-audio-tools.git',
    author='Stability AI',
    description='Training and inference tools for generative audio models from Stability AI',
    packages=find_packages(),
    install_requires=[
        # Inference path (always loaded by Fragmenta)
        'alias-free-torch==0.0.6',
        'einops',
        'einops-exts',
        'huggingface_hub',
        'k-diffusion==0.1.1',
        'PyWavelets==1.4.1',
        'safetensors',
        'sentencepiece==0.1.99',
        'torch>=2.5,<=2.8',
        'torchaudio>=2.5,<=2.8',
        'tqdm',
        'transformers',

        # Training subprocess (train.py); top-level imports must resolve
        'auraloss==0.4.0',
        'descript-audio-codec==1.0.0',
        'ema-pytorch==0.2.3',
        'pandas==2.0.2',
        'prefigure==0.0.9',
        'pytorch_lightning==2.1.0',
        'wandb==0.15.4',
        'webdataset==0.2.100',

        # Lazy-loaded by certain model configs (kept for compatibility with
        # configs other than SAO 1.0 / Small):
        'encodec==0.1.1',
        'laion-clap==1.1.4',
        'local-attention==1.8.6',
        'vector-quantize-pytorch==1.14.41',

        # Dropped (Fragmenta does not exercise these paths):
        # 'gradio>=5.20.0'             - run_gradio.py only; Fragmenta uses React
        # 'v-diffusion-pytorch==0.0.2' - zero imports in this codebase
        # 'torchmetrics==0.11.4'       - zero imports in this codebase
        # 'importlib-resources==5.12.0' - stdlib in Python 3.9+
    ],
)
