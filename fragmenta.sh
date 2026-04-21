#!/bin/bash

echo "Fragmenta Desktop"
echo "==================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"

command_exists() {
    command -v "$1" >/dev/null 2>&1
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

echo "Installing system dependencies..."
if command_exists apt-get; then
    sudo apt update -qq
    sudo apt install -y \
        pkg-config \
        python3.11-venv \
        build-essential \
        python3-dev \
        ninja-build
elif command_exists brew; then
    brew install python@3.11
elif command_exists pacman; then
    sudo pacman -S python
else
    echo "Please install Python 3.11+ manually"
fi

if [ "$(uname -s)" = "Linux" ]; then
    install_linux_webview_deps
fi

VENV_PATH="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating Python virtual environment..."
    python3.11 -m venv "$VENV_PATH" || python3 -m venv "$VENV_PATH"
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