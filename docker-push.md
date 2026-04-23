** Multi-arch CPU (if you want Apple Silicon + Linux x86 in one tag):


docker buildx create --use --name fragmenta-builder 2>/dev/null || docker buildx use fragmenta-builder
docker buildx build --platform linux/amd64,linux/arm64 \
  -t mazcode/fragmenta:cpu -f Dockerfile.cpu . --push

** GPU (also serves as :latest):


docker build -t mazcode/fragmenta:gpu -t mazcode/fragmenta:latest -f Dockerfile .
docker push mazcode/fragmenta:gpu
docker push mazcode/fragmenta:latest

** CPU:


docker build -t mazcode/fragmenta:cpu -f Dockerfile.cpu .
docker push mazcode/fragmenta:cpu

** HF: 

python scripts/deploy_hf_space.py \
  --space-name MazCodes/fragmenta \
  --token hf_XXXXXXXXXXXXXXXXXXXX