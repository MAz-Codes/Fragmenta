#!/bin/bash

echo "Fragmenta Desktop"
echo "==================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"

command_exists() { command -v "$1" >/dev/null 2>&1; }
is_python_314() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,14) else 1)' >/dev/null 2>&1
}

PYTHON_CMD=""
find_python_314() {
    for cmd in python3.14 python3 python; do
        if command_exists "$cmd" && is_python_314 "$cmd"; then
            PYTHON_CMD="$cmd"
            return 0
        fi
    done
    if command_exists uv; then
        local uv_py
        uv_py="$(uv python find 3.14 2>/dev/null)"
        if [ -n "$uv_py" ] && is_python_314 "$uv_py"; then
            PYTHON_CMD="$uv_py"
            return 0
        fi
    fi
    return 1
}

install_python314() {
    if command_exists apt-get; then
        echo "Attempting to install Python 3.14 via apt..."
        sudo apt update -qq
        if ! sudo apt install -y python3.14 python3.14-venv python3.14-dev 2>/dev/null; then
            if command_exists add-apt-repository; then
                echo "python3.14 not in default repos — adding deadsnakes PPA..."
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt update -qq
                sudo apt install -y python3.14 python3.14-venv python3.14-dev || return 1
            else
                echo "deadsnakes PPA needed but 'add-apt-repository' is not available."
                return 1
            fi
        fi
    elif command_exists dnf; then
        sudo dnf install -y python3.14 python3.14-devel || return 1
    elif command_exists pacman; then
        sudo pacman -Sy --noconfirm python || return 1
    elif command_exists brew; then
        brew install python@3.14 || return 1
        export PATH="/opt/homebrew/opt/python@3.14/bin:/usr/local/opt/python@3.14/bin:$PATH"
    elif command_exists uv; then
        uv python install 3.14 || return 1
    else
        echo "No supported package manager detected for auto-installation."
        return 1
    fi
    return 0
}

install_linux_webview_deps() {
    # The dataset folder picker shells out to zenity (see /api/pick-folder).
    # Ensure it's present up front — independent of the GI/WebKit early-return
    # below, so existing installs that already have those deps still get it.
    if ! command_exists zenity; then
        echo "Installing zenity (dataset folder picker) ..."
        if command_exists apt-get; then
            sudo apt install -y zenity
        elif command_exists dnf; then
            sudo dnf install -y zenity
        elif command_exists pacman; then
            sudo pacman -Sy --noconfirm zenity
        fi
    fi

    command_exists pkg-config || return 0
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

echo "Checking for Python 3.14..."
if ! find_python_314; then
    echo "Python 3.14 not found — attempting auto-install..."
    if ! install_python314 || ! find_python_314; then
        echo ""
        echo "ERROR: Python 3.14 is required but could not be installed automatically."
        echo ""
        echo "Install Python 3.14, then rerun this script:"
        echo "  - Ubuntu 26.04 / Fedora 43 / Arch: the system python3 is 3.14"
        echo "  - Ubuntu 24.04 / Debian:     sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.14 python3.14-venv python3.14-dev"
        echo "  - Any distro, with uv:       uv python install 3.14   (https://docs.astral.sh/uv/)"
        echo "  - From python.org:           https://www.python.org/downloads/"
        exit 1
    fi
fi
echo "Using Python 3.14 via: $PYTHON_CMD ($($PYTHON_CMD --version))"

if ! "$PYTHON_CMD" -m venv --help >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_CMD is missing the 'venv' module (Debian/Ubuntu: sudo apt install python3.14-venv)."
    exit 1
fi

[ "$(uname -s)" = "Linux" ] && install_linux_webview_deps

exec "$PYTHON_CMD" "$PROJECT_ROOT/install.py" --launch
