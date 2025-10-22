#!/bin/bash

echo "Fragmenta Desktop"
echo "================================="


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"
echo ""

install_python311_via_brew() {
    echo "Installing Python 3.11 via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install python@3.11
        export PATH="/opt/homebrew/bin:/opt/homebrew/opt/python@3.11/bin:/usr/local/bin:/usr/local/opt/python@3.11/bin:$PATH"
        return 0
    else
        echo "Homebrew not available for auto-installation"
        return 1
    fi
}

setup_python311() {
    echo "Checking for Python 3.11..."
    
    for cmd in python3.11 python3; do
        if command -v "$cmd" &> /dev/null; then
            PYTHON_VERSION=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            if [[ "$PYTHON_VERSION" == "3.11" ]]; then
                PYTHON_CMD="$cmd"
                echo "Found Python 3.11 via '$cmd'"
                $cmd --version
                return 0
            fi
        fi
    done
    
    echo "Python 3.11 not found, attempting auto-installation..."
    
    if install_python311_via_brew; then
        for cmd in python3.11 python3; do
            if command -v "$cmd" &> /dev/null; then
                PYTHON_VERSION=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
                if [[ "$PYTHON_VERSION" == "3.11" ]]; then
                    PYTHON_CMD="$cmd"
                    echo "Successfully installed Python 3.11"
                    return 0
                fi
            fi
        done
    fi
    
    echo ""
    echo "Could not automatically install Python 3.11"
    echo ""
    echo "Please install Python 3.11 manually:"
    echo "  1. Install Homebrew: https://brew.sh"
    echo "  2. Run: brew install python@3.11"
    echo "  3. Or download from: https://python.org"
    echo ""
    echo "Fragmenta requires Python 3.11 for optimal AI model compatibility."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
}

setup_python311
echo ""

if command -v brew &> /dev/null; then
    echo "Installing minimal system dependencies via Homebrew..."
    echo "(Only installing Python 3.11 if needed)"
    if ! command -v python3.11 &> /dev/null; then
        echo "Installing Python 3.11..."
        brew install python@3.11 2>/dev/null || echo "Python 3.11 installation skipped"
    fi
    echo "System dependencies ready"
else
    echo "Homebrew not found - this is OK for basic functionality"
    echo "For optimal performance, consider installing Homebrew:"
    echo "https://brew.sh"
fi
echo ""

VENV_PATH="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating Python virtual environment..."
    echo "(This is a one-time setup)"
    $PYTHON_CMD -m venv "$VENV_PATH"
    echo "Virtual environment created"
else
    echo "Virtual environment exists"
fi
echo ""

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "Virtual environment activated"
echo ""

cd "$PROJECT_ROOT"

echo "Updating Python package tools..."
pip install --upgrade pip setuptools wheel build --quiet
echo "Package tools updated"
echo ""

echo "Installing all dependencies from requirements.txt..."
echo "This may take several minutes on first install..."
pip install -r requirements.txt --quiet || {
    echo "Some dependencies had conflicts, retrying with verbose output..."
    pip install -r requirements.txt
}
echo "Dependencies installed"
echo ""

echo "Installing audio generation engine..."
cd stable-audio-tools && pip install -e . --quiet && cd ..
echo "Audio engine installed"
echo ""

echo "Verifying PyTorch installation..."
if python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "PyTorch working correctly"
else
    echo "PyTorch verification failed"
fi
echo ""

echo "Installation complete! Starting Fragmenta..."
echo "The desktop application will open in a moment..."
echo ""

if [ -f "main.py" ]; then
    python main.py
else
    echo "Error: main.py not found in $(pwd)"
    echo "Available files:"
    ls -la
    echo ""
    echo "Please check the installation and try again."
    read -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo "Thanks for using Fragmenta!"

read -p "Press Enter to continue..."
