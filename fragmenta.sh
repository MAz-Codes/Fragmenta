#!/bin/bash

echo "Fragmenta Desktop"
echo "==================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

is_python_311() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)' >/dev/null 2>&1
}

PYTHON_CMD=""

find_python_311() {
    for cmd in python3.11 python3 python; do
        if command_exists "$cmd" && is_python_311 "$cmd"; then
            PYTHON_CMD="$cmd"
            return 0
        fi
    done
    return 1
}

install_python311() {
    if command_exists apt-get; then
        echo "Attempting to install Python 3.11 via apt..."
        # Native python3.11 only exists on Ubuntu 22.04 / Debian 12.
        # Newer releases need the deadsnakes PPA.
        sudo apt update -qq
        if ! sudo apt install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null; then
            if command_exists add-apt-repository; then
                echo "python3.11 not in default repos — adding deadsnakes PPA..."
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt update -qq
                sudo apt install -y python3.11 python3.11-venv python3.11-dev || return 1
            else
                echo "deadsnakes PPA needed but 'add-apt-repository' is not available."
                echo "Install software-properties-common, or install Python 3.11 manually."
                return 1
            fi
        fi
    elif command_exists dnf; then
        sudo dnf install -y python3.11 || return 1
    elif command_exists brew; then
        brew install python@3.11 || return 1
        export PATH="/opt/homebrew/opt/python@3.11/bin:/usr/local/opt/python@3.11/bin:$PATH"
    elif command_exists pacman; then
        # Arch only ships the latest python; 3.11 lives in AUR (python311).
        echo "Arch Linux does not ship Python 3.11 in core repos."
        echo "Install it from the AUR (e.g. 'yay -S python311') or build from source, then rerun."
        return 1
    else
        echo "No supported package manager detected for auto-installation."
        return 1
    fi
    return 0
}

install_linux_webview_deps() {
    if ! command_exists pkg-config; then
        return
    fi

    if pkg-config --exists gobject-introspection-1.0 && pkg-config --exists girepository-2.0; then
        echo "Linux GI/WebKit dependencies already available."
        return
    fi

    echo "Installing missing Linux desktop runtime dependencies for pywebview..."
    if command_exists apt-get; then
        sudo apt update -qq
        sudo apt install -y \
            python3-gi \
            python3-gi-cairo \
            gir1.2-webkit2-4.1 \
            libgirepository1.0-dev \
            libcairo2-dev
    elif command_exists dnf; then
        sudo dnf install -y \
            python3-gobject \
            webkit2gtk4.1 \
            gobject-introspection-devel \
            cairo-gobject-devel
    elif command_exists pacman; then
        sudo pacman -Sy --noconfirm \
            python-gobject \
            webkit2gtk \
            gobject-introspection \
            cairo
    else
        echo "Could not auto-install GI/WebKit dependencies."
        echo "Install your distro packages for Python GI + WebKitGTK, then rerun."
    fi
}

echo "Installing system build dependencies..."
if command_exists apt-get; then
    sudo apt update -qq
    sudo apt install -y \
        pkg-config \
        build-essential \
        ninja-build
fi

echo "Checking for Python 3.11..."
if ! find_python_311; then
    echo "Python 3.11 not found — attempting auto-install..."
    if ! install_python311 || ! find_python_311; then
        echo ""
        echo "ERROR: Python 3.11 is required but could not be installed automatically."
        echo ""
        echo "Fragmenta 0.2 pins torch==2.7.1 + flash-attn cp311 wheels — these wheels"
        echo "ship only for Python 3.11. Newer Pythons (3.12, 3.13) will fail to resolve"
        echo "the binary dependencies."
        echo ""
        echo "Install Python 3.11 manually, then rerun this script:"
        echo "  - Ubuntu 22.04 / Debian 12: sudo apt install python3.11 python3.11-venv python3.11-dev"
        echo "  - Ubuntu 24.04+:            sudo add-apt-repository ppa:deadsnakes/ppa"
        echo "                              sudo apt install python3.11 python3.11-venv python3.11-dev"
        echo "  - Fedora:                   sudo dnf install python3.11"
        echo "  - Arch (AUR):               yay -S python311"
        echo "  - From source:              https://www.python.org/downloads/release/python-3119/"
        echo ""
        exit 1
    fi
fi
echo "Using Python 3.11 via: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Some distros split the venv module into a separate package; verify before continuing.
if ! "$PYTHON_CMD" -m venv --help >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_CMD is missing the 'venv' module."
    echo "On Debian/Ubuntu install: sudo apt install python3.11-venv"
    exit 1
fi

if [ "$(uname -s)" = "Linux" ]; then
    install_linux_webview_deps
fi

VENV_PATH="$PROJECT_ROOT/venv"
if [ -d "$VENV_PATH" ]; then
    # Refuse to reuse a venv built with the wrong Python.
    if [ -x "$VENV_PATH/bin/python" ] && ! is_python_311 "$VENV_PATH/bin/python"; then
        echo "Existing venv was not built with Python 3.11 — removing and recreating..."
        rm -rf "$VENV_PATH"
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating Python virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_PATH"
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

cd "$PROJECT_ROOT"

echo "Updating pip..."
pip install --upgrade pip "setuptools<70" wheel build --quiet

# requirements.txt already declares --extra-index-url for the CUDA 12.8 torch
# wheels at the top of the file, and resolves flash-attn from a pinned wheel
# URL behind a sys_platform == 'linux' marker. A single pip install resolves
# the whole graph — no manual torch/numpy/flash-attn pre-installs needed.
# The Stable Audio 3 vendor lives at vendor/stable-audio-3 and is loaded via
# sys.path from Python, not pip; nothing to install for it here.
echo "Installing dependencies from requirements.txt..."
echo "(first run takes several minutes — torch + transformers are large)"
pip install -r requirements.txt --progress-bar on \
    --find-links "$PROJECT_ROOT/utils/vendor/wheels" --prefer-binary
REQ_STATUS=$?
if [ $REQ_STATUS -ne 0 ]; then
    echo "ERROR: Failed to install dependencies. Check the log above."
    exit 1
fi

# laion-clap pins numpy<2 in its metadata (conflicts with SA3's
# numpy>=2.2.6) but works fine at runtime with numpy 2.x. Install
# without re-resolving its deps — requirements.txt above already
# brought in everything laion-clap actually imports at runtime.
echo "Installing laion-clap (auto-annotator) with --no-deps..."
pip install "laion-clap>=1.1.6" --no-deps --quiet || \
    echo "WARNING: laion-clap install failed — auto-annotation features may not work"

echo "Verifying key installations..."
python3 -c 'import torch; print(f"PyTorch {torch.__version__} with CUDA {torch.cuda.is_available()}")' 2>/dev/null || echo "PyTorch issue"
python3 -c 'import webview; print("pywebview ready")' 2>/dev/null || echo "pywebview issue"
python3 -c 'import flash_attn; print(f"Flash Attention {flash_attn.__version__}")' 2>/dev/null || echo "Flash Attention not available (Linux + CUDA only)"

echo "Starting Fragmenta..."
python3 start.py