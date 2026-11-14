#!/bin/sh
# Reproduces the ComputerXRobot-Bench baseline results in BENCHMARK.md.
#   PY=python WORKERS=8 SEEDS="0 1 2 3 4" sh scripts/run_baselines.sh
# Steps whose outputs exist are skipped (delete data/, checkpoints/ and results/ for a run from
# scratch), so adding seeds to SEEDS only runs the new ones.
set -e
PY=${PY:-python}
W=${WORKERS:-8}
SEEDS=${SEEDS:-0 1 2 3 4}

# 1. Oracle demonstrations on the train split (100 per task, id scenes, perturbed).
[ -f data/train.npz ] || $PY -m benchmark collect --per-task 100 --seed 1 --out data/train.npz

# 2. Behaviour cloning, one model per seed (recipe selected on the val split; see BENCHMARK.md).
for s in $SEEDS; do
  [ -f checkpoints/bc_s$s.pt ] || \
    $PY main.py train --data data/train.npz --epochs 80 --seed $s --out checkpoints/bc_s$s.pt
done

# 3. Evaluation on val and the four test splits.
for a in oracle random; do
  [ -f results/${a}_val.json ] || $PY -m benchmark run --agent $a --all-test --split val --workers $W --out-dir results
done
for s in $SEEDS; do
  [ -f results/bc_s${s}_val.json ] || \
    $PY -m benchmark run --agent bc:checkpoints/bc_s$s.pt --tag bc_s$s --all-test --split val --workers $W --out-dir results
done
