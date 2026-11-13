#!/bin/sh
# Reproduces the hard-tier baseline results in BENCHMARK.md (realistic physics, hidden dynamics).
#   PY=python WORKERS=8 SEEDS="0 1 2 3 4" sh scripts/run_hard_baselines.sh
# Needs the checkpoints from scripts/run_baselines.sh for the zero-shot "bc-nominal" rows.
# Steps whose outputs exist are skipped.
set -e
PY=${PY:-python}
W=${WORKERS:-8}
SEEDS=${SEEDS:-0 1 2 3 4}

# 1. Oracle demonstrations under hidden dynamics (100 per task, perturbed).
[ -f data/hard_train.npz ] || \
  $PY -m benchmark collect --split hard_train --per-task 100 --seed 1 --out data/hard_train.npz

# 2. Behaviour cloning without and with a 4-step history of observations and actions.
for s in $SEEDS; do
  [ -f checkpoints/hard_bc_s$s.pt ] || \
    $PY main.py train --data data/hard_train.npz --epochs 80 --seed $s --out checkpoints/hard_bc_s$s.pt
  [ -f checkpoints/hard_bch_s$s.pt ] || \
    $PY main.py train --data data/hard_train.npz --epochs 80 --history 4 --seed $s --out checkpoints/hard_bch_s$s.pt
done

# 3. Evaluation on hard_val, hard_test and hard_test_ood.
for a in oracle random; do
  [ -f results/${a}_hard_val.json ] || \
    $PY -m benchmark run --agent $a --all-hard --split hard_val --workers $W --out-dir results
done
for s in $SEEDS; do
  [ -f results/hard_bc_s${s}_hard_val.json ] || \
    $PY -m benchmark run --agent bc:checkpoints/hard_bc_s$s.pt --name bc --tag hard_bc_s$s --all-hard --split hard_val --workers $W --out-dir results
  [ -f results/hard_bch_s${s}_hard_val.json ] || \
    $PY -m benchmark run --agent bc:checkpoints/hard_bch_s$s.pt --name bc-history --tag hard_bch_s$s --all-hard --split hard_val --workers $W --out-dir results
  [ -f results/nominal_bc_s${s}_hard_test_ood.json ] || \
    $PY -m benchmark run --agent bc:checkpoints/bc_s$s.pt --name bc-nominal --tag nominal_bc_s$s --all-hard --workers $W --out-dir results
done
