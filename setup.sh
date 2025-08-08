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
# 1b. Install gsutil (Google Cloud SDK) if not present
# ---------------------------------------------------------------------------
if ! command -v gsutil >/dev/null 2>&1; then
  echo "[setup] Installing gsutil (Google Cloud SDK)…"
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y apt-transport-https ca-certificates gnupg curl >/dev/null 2>&1 || true
    # Add Google Cloud apt repository key if missing
    if [ ! -f /usr/share/keyrings/cloud.google.gpg ]; then
      curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg || true
    fi
    # Add repository list if missing
    if [ ! -f /etc/apt/sources.list.d/google-cloud-sdk.list ]; then
      echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list || true
    fi
    apt-get update -y >/dev/null 2>&1 || true
    # Try both package names across distros
    if ! apt-get install -y google-cloud-sdk >/dev/null 2>&1; then
      if ! apt-get install -y google-cloud-cli >/dev/null 2>&1; then
        echo "[setup] ERROR: Failed to install Google Cloud SDK. Please install gsutil manually." >&2
      fi
    fi
  else
    echo "[setup] 'apt-get' not found; please install gsutil manually: https://cloud.google.com/sdk/docs/install" >&2
  fi
else
  echo "[setup] gsutil already present: $(command -v gsutil)"
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
