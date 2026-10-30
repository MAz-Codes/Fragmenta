#!/bin/bash

echo "Fragmenta Desktop"
echo "================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"
echo ""

PYTHON_CMD=""
is_py314() { "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,14) else 1)' >/dev/null 2>&1; }

find_python_314() {
    for cmd in python3.14 python3; do
        if command -v "$cmd" >/dev/null 2>&1 && is_py314 "$cmd"; then
            PYTHON_CMD="$cmd"
            return 0
        fi
    done
    return 1
}

echo "Checking for Python 3.14..."
if ! find_python_314; then
    echo "Python 3.14 not found — attempting auto-install via Homebrew..."
    if command -v brew >/dev/null 2>&1; then
        brew install python@3.14
        export PATH="/opt/homebrew/opt/python@3.14/bin:/usr/local/opt/python@3.14/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    fi
    if ! find_python_314; then
        echo ""
        echo "Could not automatically install Python 3.14."
        echo "  1. Install Homebrew: https://brew.sh"
        echo "  2. Run: brew install python@3.14"
        echo ""
        read -p "Press Enter to exit..."
        exit 1
    fi
fi
echo "Using Python 3.14 via: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""

"$PYTHON_CMD" "$PROJECT_ROOT/install.py" --launch
STATUS=$?
if [ "$STATUS" != "0" ]; then
    echo ""
    echo "Fragmenta exited with code $STATUS."
    read -p "Press Enter to close..."
fi
