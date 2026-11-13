#!/bin/sh
# Non-privileged experts (learning/adaptive.py): the oracle's control law on state-track
# observations, with online identification of the hidden dynamics (adaptive) and without it
# (fixed). Deterministic, no training: one run each. Steps whose outputs exist are skipped.
#   PY=python WORKERS=8 sh scripts/run_adaptive.sh
set -e
PY=${PY:-python}
W=${WORKERS:-8}
N=${ABLATION_PER_TASK:-20}

for a in adaptive fixed; do
  [ -f results/${a}_val.json ] || $PY -m benchmark run --agent $a --all-test --split val --workers $W --out-dir results
  [ -f results/${a}_hard_val.json ] || \
    $PY -m benchmark run --agent $a --all-hard --split hard_val --workers $W --out-dir results
  [ -f results/ablation_$a.json ] || \
    $PY -m benchmark ablate --agent $a --per-task $N --workers $W --out-dir results
done
