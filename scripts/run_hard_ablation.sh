#!/bin/sh
# Single-factor dynamics ablation (benchmark/ablation.py), run after run_hard_baselines.sh and
# run_hard_dagger.sh: which hidden real-world effect breaks which policy?
#   PY=python WORKERS=8 SEEDS="0 1 2 3 4" sh scripts/run_hard_ablation.sh
# Each run replays 20 hard_test episodes per task under 13 conditions (2,340 episodes).
# Ablations whose result file exists are skipped.
set -e
PY=${PY:-python}
W=${WORKERS:-8}
SEEDS=${SEEDS:-0 1 2 3 4}
N=${ABLATION_PER_TASK:-20}

ablate() {  # $1 = tag, then the agent arguments
  tag=$1
  shift
  [ -f results/ablation_$tag.json ] || \
    $PY -m benchmark ablate "$@" --tag $tag --per-task $N --workers $W --out-dir results
}

ablate oracle --agent oracle
for s in $SEEDS; do
  ablate nominal_bc_s$s --agent bc:checkpoints/bc_s$s.pt --name bc-nominal
  ablate hard_bcd_s$s --agent bc:checkpoints/hard_bc_dagger2_s$s.pt --name bc-dagger
  ablate hard_bchd_s$s --agent bc:checkpoints/hard_bch_dagger2_s$s.pt --name bc-history-dagger
done
