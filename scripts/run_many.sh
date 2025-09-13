#!/usr/bin/env bash

set -euo pipefail
# Ignore SIGHUP so that child processes (e.g. torchrun workers) aren’t terminated when
# the controlling terminal closes. This prevents the SignalException (signal 1) seen
# when running under nohup.
trap '' HUP

if [ "$#" -lt 2 ]; then
  echo "Usage: bash $(basename "$0") CONFIG_1 [CONFIG_2 ...] NUM_RUNS" >&2
  exit 1
fi

NUM_RUNS="${@: -1}"

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [ "$NUM_RUNS" -le 0 ]; then
  echo "NUM_RUNS must be a positive integer" >&2
  exit 1
fi

CONFIGS=("${@:1:$#-1}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if command -v python >/dev/null 2>&1; then
  PY="python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "Python interpreter not found. Please install python or python3." >&2
  exit 1
fi

for CONFIG_NAME in "${CONFIGS[@]}"; do
  # Resolve config path for reading multi_gpu (match scripts/train.py behavior)
  CONFIG_PATH="$CONFIG_NAME"
  if [[ "$CONFIG_PATH" != /* ]] && [ ! -f "$CONFIG_PATH" ]; then
    PKG_CFG_DIR="$REPO_ROOT/src/config"
    CANDIDATE="$PKG_CFG_DIR/$CONFIG_PATH"
    if [ -f "$CANDIDATE" ]; then
      CONFIG_PATH="$CANDIDATE"
    fi
  fi

  # Read multi_gpu from config (default to "none" if missing)
  MULTI_GPU="$(
  $PY - "$CONFIG_PATH" <<'PY'
import sys
from omegaconf import OmegaConf
try:
    cfg = OmegaConf.load(sys.argv[1])
    print(cfg.get("multi_gpu", "none"))
except Exception:
    print("none")
PY
  )"

  # Determine launcher and nproc if multi_gpu is enabled
  USE_TORCHRUN=false
  NPROC=1
  if [ "$MULTI_GPU" != "none" ]; then
    # Prefer torch.cuda for device count; fallback to nvidia-smi
    GPU_COUNT="$(
    $PY - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(-1)
PY
    )"
    if [ "$GPU_COUNT" -lt 0 ] || [ "$GPU_COUNT" = "" ]; then
      if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d " ")"
      else
        GPU_COUNT=0
      fi
    fi

    if [ "$GPU_COUNT" -lt 1 ]; then
      echo "multi_gpu is '$MULTI_GPU' but no GPUs were detected" >&2
      exit 1
    fi

    NPROC="$GPU_COUNT"
    if ! command -v torchrun >/dev/null 2>&1; then
      echo "multi_gpu is '$MULTI_GPU' but 'torchrun' was not found in PATH. Install PyTorch with distributed utilities to use multi-GPU (torchrun)." >&2
      exit 1
    fi
    LAUNCHER=(torchrun)
    USE_TORCHRUN=true
    echo "[run_many] multi_gpu='$MULTI_GPU' | GPUs detected=$GPU_COUNT | launcher='torchrun' | nproc-per-node=$NPROC | config='$CONFIG_PATH'"
  else
    echo "using single GPU training"
  fi

  for ((i=1; i<=NUM_RUNS; i++)); do
    if [ "$USE_TORCHRUN" = true ]; then
      echo "Starting distributed run ${i}/${NUM_RUNS} | config='${CONFIG_NAME}' | multi_gpu='${MULTI_GPU}' | GPUs=${NPROC}"
      "${LAUNCHER[@]}" --nproc-per-node "$NPROC" "$REPO_ROOT/scripts/train.py" --config "$CONFIG_NAME" || { echo "[run_many] Run ${i}/${NUM_RUNS} failed with exit code $? – skipping."; continue; }
    else
      echo "Starting single-process run ${i}/${NUM_RUNS} | config='${CONFIG_NAME}' | multi_gpu='none'"
      "$PY" "$REPO_ROOT/scripts/train.py" --config "$CONFIG_NAME" || { echo "[run_many] Run ${i}/${NUM_RUNS} failed with exit code $? – skipping."; continue; }
    fi
  done
done
