#!/bin/sh
# Reproduces the hard-tier baseline results in BENCHMARK.md (realistic physics, hidden dynamics).
#   PY=python WORKERS=8 sh scripts/run_hard_baselines.sh
# Needs the v1.0 checkpoints from scripts/run_baselines.sh for the zero-shot "bc-nominal" rows.
set -e
PY=${PY:-python}
W=${WORKERS:-8}

# 1. Oracle demonstrations under hidden dynamics (100 per task, perturbed).
$PY -m benchmark collect --split hard_train --per-task 100 --seed 1 --out data/hard_train.npz

# 2. Behaviour cloning without and with a 4-step history of observations and actions, 3 seeds each.
for s in 0 1 2; do
  $PY main.py train --data data/hard_train.npz --epochs 80 --seed $s --out checkpoints/hard_bc_s$s.pt
  $PY main.py train --data data/hard_train.npz --epochs 80 --history 4 --seed $s --out checkpoints/hard_bch_s$s.pt
done

# 3. Evaluation on hard_val, hard_test and hard_test_ood.
$PY -m benchmark run --agent oracle --all-hard --split hard_val --workers $W --out-dir results
$PY -m benchmark run --agent random --all-hard --split hard_val --workers $W --out-dir results
for s in 0 1 2; do
  $PY -m benchmark run --agent bc:checkpoints/hard_bc_s$s.pt --name bc --tag hard_bc_s$s --all-hard --split hard_val --workers $W --out-dir results
  $PY -m benchmark run --agent bc:checkpoints/hard_bch_s$s.pt --name bc-history --tag hard_bch_s$s --all-hard --split hard_val --workers $W --out-dir results
  $PY -m benchmark run --agent bc:checkpoints/bc_s$s.pt --name bc-nominal --tag nominal_bc_s$s --all-hard --workers $W --out-dir results
done
