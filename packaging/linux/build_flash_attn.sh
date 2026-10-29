#!/bin/bash
# Build the flash-attn wheel Fragmenta installs on Linux (sa3-medium needs it).
#
# Upstream ships no wheel for Python 3.14, so we build one from the BSD-3
# source. It is built inside a manylinux_2_28 container so it runs on any
# distro with glibc >= 2.28, not just the one it was built on.
#
# Usage (repo root, Docker available):
#   packaging/linux/build_flash_attn.sh [out_dir]
# Env overrides: FA_VERSION, TORCH_VERSION, CUDA_VERSION, PY_TAG,
#   FLASH_ATTN_CUDA_ARCHS, MAX_JOBS, NVCC_THREADS, MEM_LIMIT, CPU_LIMIT.
#
# Takes a few hours. The container is capped (MEM_LIMIT / CPU_LIMIT) so a
# compile that runs out of memory fails on its own instead of dragging the
# desktop session down with it.
set -euo pipefail

FA_VERSION="${FA_VERSION:-2.8.3.post2}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
CUDA_VERSION="${CUDA_VERSION:-12.8}"          # torch's cu128 wheels
PY_TAG="${PY_TAG:-cp314-cp314}"
# Desktop GPUs: sm_80 (A100, and RTX 30xx/40xx run its code) + sm_120
# (RTX 50xx), same as the Windows build. Upstream's 90/100 (H100, B200) are
# datacenter cards and would double the build. FA2 doesn't support older GPUs.
ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80;120}"
# Each ptxas process can take several GB (2 jobs × 4 threads was OOM-killed
# in 18 GB), and on a 30 GB desktop a 20 GB cap starved the session itself.
# Defaults stay small and slow; raise them on a bigger build machine.
MAX_JOBS="${MAX_JOBS:-1}"
NVCC_THREADS="${NVCC_THREADS:-2}"
MEM_LIMIT="${MEM_LIMIT:-12g}"
CPU_LIMIT="${CPU_LIMIT:-10}"
IMAGE="quay.io/pypa/manylinux_2_28_x86_64"

OUT="$(mkdir -p "${1:-packaging/build/flash-attn}" && cd "${1:-packaging/build/flash-attn}" && pwd)"
# Package downloads (CUDA toolkit, torch) are kept between runs.
CACHE="${OUT}/cache" && mkdir -p "${CACHE}/dnf" "${CACHE}/pip"
CUDA_PKG="cuda-toolkit-${CUDA_VERSION/./-}"
TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION/./}"

echo "flash-attn ${FA_VERSION} · torch ${TORCH_VERSION} · CUDA ${CUDA_VERSION} · ${PY_TAG}"
echo "archs ${ARCHS} · MAX_JOBS ${MAX_JOBS} × NVCC_THREADS ${NVCC_THREADS} · limits ${MEM_LIMIT} / ${CPU_LIMIT} CPUs"
echo "output: ${OUT}"

docker run --rm \
    --memory="${MEM_LIMIT}" --memory-swap="${MEM_LIMIT}" --cpus="${CPU_LIMIT}" \
    -v "${OUT}:/out" -v "${CACHE}/dnf:/var/cache/dnf" -v "${CACHE}/pip:/root/.cache/pip" \
    -e FA_VERSION="${FA_VERSION}" -e TORCH_VERSION="${TORCH_VERSION}" \
    -e CUDA_VERSION="${CUDA_VERSION}" -e CUDA_PKG="${CUDA_PKG}" \
    -e TORCH_INDEX="${TORCH_INDEX}" -e PY_TAG="${PY_TAG}" \
    -e FLASH_ATTN_CUDA_ARCHS="${ARCHS}" -e MAX_JOBS="${MAX_JOBS}" \
    -e NVCC_THREADS="${NVCC_THREADS}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    "${IMAGE}" bash -euo pipefail -c '
        echo keepcache=1 >> /etc/dnf/dnf.conf
        dnf -y -q install dnf-plugins-core
        dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
        # CUDA 12.8 refuses GCC 14 (the image default); build with GCC 13.
        dnf -y -q install "${CUDA_PKG}" gcc-toolset-13-gcc-c++
        set +u && source /opt/rh/gcc-toolset-13/enable && set -u
        g++ --version | head -1
        export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}" && export PATH="${CUDA_HOME}/bin:${PATH}"
        nvcc --version | tail -2

        "/opt/python/${PY_TAG}/bin/python" -m venv /venv && . /venv/bin/activate
        pip install -q --upgrade pip
        pip install -q "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
        pip install -q ninja packaging psutil wheel setuptools einops
        python -c "import torch; print(\"torch\", torch.__version__, \"cxx11abi\", torch._C._GLIBCXX_USE_CXX11_ABI)"

        FLASH_ATTENTION_FORCE_BUILD=TRUE pip wheel "flash-attn==${FA_VERSION}" \
            --no-build-isolation --no-deps -w /out
        cd /out && sha256sum flash_attn-*.whl | tee SHA256SUMS
        chown -R "${HOST_UID}:${HOST_GID}" /out
    '
echo "done: $(ls "${OUT}"/flash_attn-*.whl)"
