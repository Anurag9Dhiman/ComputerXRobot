#!/bin/sh
# Reproduces the ComputerXRobot-Bench v1.0 baseline results in BENCHMARK.md.
#   PY=python WORKERS=8 sh scripts/run_baselines.sh
set -e
PY=${PY:-python}
W=${WORKERS:-8}

# 1. Oracle demonstrations on the train split (100 per task, id scenes, perturbed).
$PY -m benchmark collect --per-task 100 --seed 1 --out data/train.npz

# 2. Behaviour cloning, three seeds (recipe selected on the val split; see BENCHMARK.md).
for s in 0 1 2; do
  $PY main.py train --data data/train.npz --epochs 80 --seed $s --out checkpoints/bc_s$s.pt
done

# 3. Evaluation on val and the four test splits.
$PY -m benchmark run --agent oracle --all-test --split val --workers $W --out-dir results
$PY -m benchmark run --agent random --all-test --split val --workers $W --out-dir results
for s in 0 1 2; do
  $PY -m benchmark run --agent bc:checkpoints/bc_s$s.pt --tag bc_s$s --all-test --split val --workers $W --out-dir results
done
