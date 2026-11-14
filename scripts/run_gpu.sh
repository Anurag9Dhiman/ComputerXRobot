#!/bin/sh
# The runs that need a GPU, for a Linux server (e.g. an A100); see docs/GPU.md.
#   WORKERS=32 sh scripts/run_gpu.sh
# Vision baselines (scripts/run_vision.sh): demonstration collection and evaluation render
# four cameras per step on the CPU cores (one process per worker); training uses the GPU.
# Steps whose outputs exist are skipped, so the script can be rerun after an interruption.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
$PY - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device: run_gpu.sh is meant for a GPU server"
print("GPU:", torch.cuda.get_device_name(0))
PY
# Rendering works headless (EGL) and matches what evaluation will see.
$PY -m benchmark run --agent random --tasks type_key --per-task 1 --track vision --split test --tag smoke \
  --out-dir outputs/smoke
mkdir -p data checkpoints results outputs
sh scripts/run_vision.sh
$PY -m benchmark report results/*.json --table splits
$PY -m benchmark.plot results/*.json --vision --out docs/vision_results.png
