from setuptools import setup, find_packages

setup(
    name='stable-audio-tools',
    version='0.0.19',
    url='https://github.com/Stability-AI/stable-audio-tools.git',
    author='Stability AI',
    description='Training and inference tools for generative audio models from Stability AI',
    packages=find_packages(),
    install_requires=[

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


        'auraloss==0.4.0',
        'descript-audio-codec==1.0.0',
        'ema-pytorch==0.2.3',
        'pandas==2.0.2',
        'prefigure==0.0.9',
        'pytorch_lightning==2.1.0',
        'wandb==0.15.4',
        'webdataset==0.2.100',

        'encodec==0.1.1',
        'laion-clap==1.1.4',
        'local-attention==1.8.6',
        'vector-quantize-pytorch==1.14.41',

    ],
)
