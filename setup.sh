#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Quick-start environment setup script
# ---------------------------------------------------------------------------
# Usage (after cloning):
#   $ cd simple_obfuscation
#   $ bash setup.sh
#
# The script will
#   1. install the *uv* package-manager binary (if missing)
#   2. create a local virtual-environment in .venv/
#   3. install the project in editable mode + all dependencies
#   4. run the test-suite as a sanity check
#
# Re-run safe: if .venv already exists we only (re)install deps.
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="$(pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# ---------------------------------------------------------------------------
# 1. Install uv (via pip) if not present
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] Installing *uv* package manager via pip…"
  # Use --user to avoid requiring sudo.  This will place the binary in
  # $HOME/.local/bin on most systems.
  python3 -m pip install --user --upgrade uv || {
    echo "[setup] ERROR: pip install failed – please ensure pip is available." >&2
    exit 1
  }
  # Ensure the install location is on PATH for the remainder of the script
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[setup] *uv* already present: $(command -v uv)"
fi

# ---------------------------------------------------------------------------
# 2. Create / activate virtual-environment
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating virtual-environment at $VENV_DIR"
  uv venv "$VENV_DIR"
else
  echo "[setup] Using existing virtual-environment at $VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# 3. Install project (editable) + dependencies
# ---------------------------------------------------------------------------

echo "[setup] Installing Python dependencies…"
uv pip install --upgrade pip
uv pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Run test-suite (optional but fast)
# ---------------------------------------------------------------------------

echo "[setup] Running unit tests…"
uv pip install pytest >/dev/null 2>&1
pytest -q || {
  echo "[setup] Tests failed – please inspect output." >&2
  exit 1
}

echo "[setup] All done!  Activate the environment with:"
echo "    source .venv/bin/activate"
echo "and start hacking."
