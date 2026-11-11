#!/bin/sh
# Single-factor dynamics ablation (benchmark/ablation.py), run after run_hard_baselines.sh and
# run_hard_dagger.sh: which hidden real-world effect breaks which policy?
#   PY=python WORKERS=8 sh scripts/run_hard_ablation.sh
# Each run replays 5 hard_test episodes per task under 13 conditions (585 episodes).
set -e
PY=${PY:-python}
W=${WORKERS:-8}
N=${ABLATION_PER_TASK:-5}

$PY -m benchmark ablate --agent oracle --per-task $N --workers $W --out-dir results
for s in 0 1 2; do
  $PY -m benchmark ablate --agent bc:checkpoints/bc_s$s.pt --name bc-nominal --tag nominal_bc_s$s \
    --per-task $N --workers $W --out-dir results
  $PY -m benchmark ablate --agent bc:checkpoints/hard_bc_dagger2_s$s.pt --name bc-dagger --tag hard_bcd_s$s \
    --per-task $N --workers $W --out-dir results
  $PY -m benchmark ablate --agent bc:checkpoints/hard_bch_dagger2_s$s.pt --name bc-history-dagger \
    --tag hard_bchd_s$s --per-task $N --workers $W --out-dir results
done
