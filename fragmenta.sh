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
        echo "Fragmenta pins numpy 1.23.5 / pandas 2.0.2, which only have wheels for"
        echo "Python 3.11. Newer Pythons (3.12, 3.13) will fail to install dependencies."
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

echo "Installing all dependencies..."
echo "Installing requested packages (this may take a few minutes)..."

echo "Ensuring numpy compatibility..."
pip install "numpy==1.23.5"

echo "Installing PyTorch (CUDA 12.8 wheels)..."
pip install "torch>=2.5,<=2.8" "torchvision<0.24" "torchaudio>=2.5,<=2.8" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --progress-bar on

echo "Installing remaining dependencies (flash-attn handled separately)..."
REQ_TMP="$(mktemp)"
grep -v "^flash-attn" requirements.txt > "$REQ_TMP"
pip install -r "$REQ_TMP" --progress-bar on
REQ_STATUS=$?
rm -f "$REQ_TMP"
if [ $REQ_STATUS -ne 0 ]; then
    echo "ERROR: Failed to install core dependencies"
    exit 1
fi

echo "Attempting flash-attn (optional, Linux + CUDA only)..."
pip install "flash-attn>=2.8.3" --no-build-isolation --progress-bar on || \
    echo "flash-attn install failed — continuing without it (optional optimization)"

echo "Installing bundled stable-audio-tools..."
if [ -d "$PROJECT_ROOT/stable-audio-tools" ]; then
    (
        cd "$PROJECT_ROOT/stable-audio-tools" || exit 1
        pip install -e . --quiet
    ) || {
        echo "ERROR: Failed to install bundled stable-audio-tools"
        exit 1
    }
else
    echo "ERROR: stable-audio-tools directory not found at $PROJECT_ROOT/stable-audio-tools"
    exit 1
fi

echo "Verifying key installations..."
python3 -c 'import torch; print(f"PyTorch {torch.__version__} with CUDA {torch.cuda.is_available()}")' 2>/dev/null || echo "PyTorch issue"
python3 -c 'import webview; print("pywebview ready")' 2>/dev/null || echo "pywebview issue"
python3 -c 'import flash_attn; print(f"Flash Attention {flash_attn.__version__}")' 2>/dev/null || echo "Flash Attention not available (optional)"

echo "Starting Fragmenta..."
python3 start.py