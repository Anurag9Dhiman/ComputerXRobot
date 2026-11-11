#!/bin/sh
# Non-privileged experts (learning/adaptive.py): the oracle's control law on state-track
# observations, with online identification of the hidden dynamics (adaptive) and without it
# (fixed). Deterministic, no training: one run each.
#   PY=python WORKERS=8 sh scripts/run_adaptive.sh
set -e
PY=${PY:-python}
W=${WORKERS:-8}
N=${ABLATION_PER_TASK:-5}

for a in adaptive fixed; do
  $PY -m benchmark run --agent $a --all-test --split val --workers $W --out-dir results
  $PY -m benchmark run --agent $a --all-hard --split hard_val --workers $W --out-dir results
  $PY -m benchmark ablate --agent $a --per-task $N --workers $W --out-dir results
done
