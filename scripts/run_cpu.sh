#!/bin/sh
# Every CPU result of the paper, in dependency order: standard-split BC, the hard tier
# (BC, history BC, DAgger), the non-privileged experts and the dynamics ablation.
#   PY=python WORKERS=8 SEEDS="0 1 2 3 4" sh scripts/run_cpu.sh
# Steps whose outputs exist are skipped; delete data/, checkpoints/ and results/ for a run from
# scratch. The vision baselines need a GPU: scripts/run_gpu.sh.
set -e
cd "$(dirname "$0")/.."
sh scripts/run_baselines.sh
sh scripts/run_hard_baselines.sh
sh scripts/run_hard_dagger.sh
sh scripts/run_adaptive.sh
sh scripts/run_hard_ablation.sh
