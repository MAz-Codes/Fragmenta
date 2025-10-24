#!/bin/bash

echo "Fragmenta Desktop"
echo "==================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Installing system Qt dependencies..."
if command_exists apt-get; then
    sudo apt update -qq
    sudo apt install -y \
        qt6-base-dev \
        libqt6gui6 \
        libqt6widgets6 \
        libqt6core6 \
        libxcb-cursor0 \
        libxcb-cursor-dev \
        python3.11-venv \
        build-essential \
        python3-dev \
        ninja-build
elif command_exists brew; then
    brew install qt@6 python@3.11
elif command_exists pacman; then
    sudo pacman -S qt6-base python
else
    echo "Please install Qt6 and Python 3.11+ manually"
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
pip install --upgrade pip setuptools wheel build --quiet

echo "Installing all dependencies..."
TOTAL_PACKAGES=$(grep -v '^#' requirements.txt | grep -v '^--' | grep -v '^$' | wc -l)
echo "Installing $TOTAL_PACKAGES packages (this may take a few minutes)..."

echo "Ensuring numpy compatibility..."
pip install "numpy==1.23.5" --force-reinstall

pip install -r requirements.txt --progress-bar on

echo "Installing bundled stable-audio-tools..."
cd stable-audio-tools && pip install -e . --quiet && cd ..

echo "Verifying key installations..."
python3 -c 'import torch; print(f"PyTorch {torch.__version__} with CUDA {torch.cuda.is_available()}")' 2>/dev/null || echo "PyTorch issue"
python3 -c 'import PyQt6.QtCore; print("PyQt6 ready")' 2>/dev/null || echo "PyQt6 issue"
python3 -c 'import flash_attn; print(f"Flash Attention {flash_attn.__version__}")' 2>/dev/null || echo "Flash Attention not available (optional)"

echo "Checking if React frontend is built..."
if [ ! -f "app/frontend/build/index.html" ]; then
    echo "React frontend not built. Building now..."
    cd app/frontend
    
    echo "Installing Node.js dependencies..."
    npm install
    
    echo "Building React app..."
    npm run build
    
    cd ../..
    echo "Frontend build complete!"
else
    echo "Frontend already built. Skipping build step."
fi

echo "Starting Fragmenta..."
python3 main.py