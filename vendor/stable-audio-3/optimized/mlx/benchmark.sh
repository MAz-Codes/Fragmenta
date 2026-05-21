#!/usr/bin/env bash
#
# Thin wrapper around scripts/benchmark.py — runs it via `uv run python`
# so the project's .venv is used automatically (no activation needed).
#
# Sweeps sm-music+same-s and medium+same-l across 5s / 30s / 120s / 380s
# clip lengths, recording wall time and peak RAM per cell. Pre-warms
# weights from HuggingFace so download time is excluded from the timings.
#
# Usage:  ./benchmark.sh
#
set -euo pipefail

# Pre-emptively add the uv installer's default bin dir to PATH (matches ./sa3).
export PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -t 1 ]]; then
    RED=$'\033[1;31m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    RED=""; BOLD=""; RESET=""
fi
fail() { printf '\n%serror%s: %s\n' "$RED" "$RESET" "$1" >&2; exit 1; }

# Sanity: we need uv, the project .venv, and the benchmark script itself.
command -v uv >/dev/null 2>&1 || \
    fail "uv not found. Run ${BOLD}./install.sh${RESET} first to set up the environment."
[[ -d "$SCRIPT_DIR/.venv" ]] || \
    fail "no .venv/ found. Run ${BOLD}./install.sh${RESET} first."
[[ -f "$SCRIPT_DIR/scripts/benchmark.py" ]] || \
    fail "scripts/benchmark.py not found — repo files are missing."

cd "$SCRIPT_DIR"
# Invoke .venv/bin/python directly (same reasoning as ./sa3 — uv run
# walks up to a parent pyproject.toml and --no-project creates an
# empty ephemeral env in newer uv versions).
exec "$SCRIPT_DIR/.venv/bin/python" scripts/benchmark.py "$@"
