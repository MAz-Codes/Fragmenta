#!/bin/bash

echo "Fragmenta Desktop"
echo "==================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"

command_exists() { command -v "$1" >/dev/null 2>&1; }
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
        sudo apt update -qq
        if ! sudo apt install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null; then
            if command_exists add-apt-repository; then
                echo "python3.11 not in default repos — adding deadsnakes PPA..."
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt update -qq
                sudo apt install -y python3.11 python3.11-venv python3.11-dev || return 1
            else
                echo "deadsnakes PPA needed but 'add-apt-repository' is not available."
                return 1
            fi
        fi
    elif command_exists dnf; then
        sudo dnf install -y python3.11 || return 1
    elif command_exists brew; then
        brew install python@3.11 || return 1
        export PATH="/opt/homebrew/opt/python@3.11/bin:/usr/local/opt/python@3.11/bin:$PATH"
    elif command_exists pacman; then
        echo "Arch Linux does not ship Python 3.11 in core repos."
        echo "Install it from the AUR (e.g. 'yay -S python311'), then rerun."
        return 1
    else
        echo "No supported package manager detected for auto-installation."
        return 1
    fi
    return 0
}

install_linux_webview_deps() {
    command_exists pkg-config || return 0
    # Probe girepository-1.0 — that's what the remedy below installs
    # (libgirepository1.0-dev, matching the PyGObject<3.49 pin). Probing
    # girepository-2.0 here meant the check could never pass after the
    # install, so every launch re-ran sudo apt.
    if pkg-config --exists gobject-introspection-1.0 && pkg-config --exists girepository-1.0; then
        echo "Linux GI/WebKit dependencies already available."
        return 0
    fi
    echo "Installing missing Linux desktop runtime dependencies for pywebview..."
    if command_exists apt-get; then
        sudo apt update -qq
        sudo apt install -y python3-gi python3-gi-cairo gir1.2-webkit2-4.1 \
            libgirepository1.0-dev libcairo2-dev
    elif command_exists dnf; then
        sudo dnf install -y python3-gobject webkit2gtk4.1 \
            gobject-introspection-devel cairo-gobject-devel
    elif command_exists pacman; then
        sudo pacman -Sy --noconfirm python-gobject webkit2gtk gobject-introspection cairo
    else
        echo "Could not auto-install GI/WebKit dependencies — install them manually, then rerun."
    fi
}

echo "Checking for Python 3.11..."
if ! find_python_311; then
    echo "Python 3.11 not found — attempting auto-install..."
    if ! install_python311 || ! find_python_311; then
        echo ""
        echo "ERROR: Python 3.11 is required but could not be installed automatically."
        echo "Fragmenta 0.2 pins torch==2.7.1 + flash-attn cp311 wheels — these ship"
        echo "only for Python 3.11. Newer Pythons (3.12, 3.13) will fail to resolve them."
        echo ""
        echo "Install Python 3.11, then rerun this script:"
        echo "  - Ubuntu 22.04 / Debian 12: sudo apt install python3.11 python3.11-venv python3.11-dev"
        echo "  - Ubuntu 24.04+:            sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv"
        echo "  - Fedora:                   sudo dnf install python3.11"
        echo "  - From source:              https://www.python.org/downloads/release/python-3119/"
        exit 1
    fi
fi
echo "Using Python 3.11 via: $PYTHON_CMD ($($PYTHON_CMD --version))"

if ! "$PYTHON_CMD" -m venv --help >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_CMD is missing the 'venv' module (Debian/Ubuntu: sudo apt install python3.11-venv)."
    exit 1
fi

[ "$(uname -s)" = "Linux" ] && install_linux_webview_deps

exec "$PYTHON_CMD" "$PROJECT_ROOT/install.py" --launch
