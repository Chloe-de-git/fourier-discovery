#!/usr/bin/env bash
set -euo pipefail

cd /data/ziling/aidf/fourier-discovery

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MPLCONFIGDIR=/data/ziling/aidf/fourier-discovery/results/.mplconfig
export PYTHONUNBUFFERED=1

mkdir -p results "$MPLCONFIGDIR"

{
  echo "started_at=$(date -Is)"
  echo "cwd=$(pwd)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  /data/ziling/miniconda3/envs/fourier-discovery/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY
  /data/ziling/miniconda3/envs/fourier-discovery/bin/python -m src.experiments --mode full --results-dir results
  /data/ziling/miniconda3/envs/fourier-discovery/bin/python visualize.py --checkpoint results/spectral_neural_n32_full_model.pt --n 32 --results-dir results
  echo "finished_at=$(date -Is)"
} 2>&1 | tee results/full_experiment.log
