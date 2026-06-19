#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-results/.mplconfig}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p results "$MPLCONFIGDIR"

{
  echo "started_at=$(date -Is)"
  echo "repo=$(basename "${REPO_ROOT}")"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY
  "${PYTHON_BIN}" -m src.experiments --mode full --results-dir results
  "${PYTHON_BIN}" visualize.py --checkpoint results/spectral_neural_n32_full_model.pt --n 32 --results-dir results
  echo "finished_at=$(date -Is)"
} 2>&1 | tee results/full_experiment.log
