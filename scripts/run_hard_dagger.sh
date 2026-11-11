#!/bin/sh
# DAgger baselines for the hard tier (run after scripts/run_hard_baselines.sh).
#   PY=python WORKERS=8 sh scripts/run_hard_dagger.sh
#
# The oracle's actions depend on state the policy cannot observe: commands still in flight,
# the true hand position, the mouse's rotation and gain. A behaviour-cloning student therefore
# drifts into states the oracle never visited and has no label for them. DAgger rolls out the
# student on hard_train, lets the oracle label every state the student visits (failed episodes
# included), and retrains on the aggregated data. Two iterations from each BC checkpoint of
# run_hard_baselines.sh, 3 seeds, without and with history; the six runs go in parallel.
set -e
PY=${PY:-python}
W=${WORKERS:-8}
N=${DAGGER_PER_TASK:-25}     # student episodes per task per iteration
export OMP_NUM_THREADS=${THREADS_PER_RUN:-2}
mkdir -p outputs

dagger() {  # $1 = bc | bch (history), $2 = seed
  v=$1 s=$2 hist=""
  [ "$v" = bch ] && hist="--history 4"
  student=checkpoints/hard_${v}_s$s.pt
  data=data/hard_train.npz
  for i in 1 2; do
    $PY -m benchmark collect --split hard_train --per-task $N --student $student --workers 2 \
      --out data/dagger_${v}_s${s}_$i.npz
    data="$data data/dagger_${v}_s${s}_$i.npz"
    $PY main.py train --data $data --epochs 80 $hist --seed $s --out checkpoints/hard_${v}_dagger${i}_s$s.pt
    student=checkpoints/hard_${v}_dagger${i}_s$s.pt
  done
}

pids=""
for s in 0 1 2; do
  for v in bc bch; do
    dagger $v $s > outputs/dagger_${v}_s$s.log 2>&1 &
    pids="$pids $!"
  done
done
for p in $pids; do
  wait $p || { echo "a DAgger run failed; see outputs/dagger_*.log" >&2; exit 1; }
done

for s in 0 1 2; do
  $PY -m benchmark run --agent bc:checkpoints/hard_bc_dagger2_s$s.pt --name bc-dagger --tag hard_bcd_s$s \
    --all-hard --split hard_val --workers $W --out-dir results
  $PY -m benchmark run --agent bc:checkpoints/hard_bch_dagger2_s$s.pt --name bc-history-dagger \
    --tag hard_bchd_s$s --all-hard --split hard_val --workers $W --out-dir results
done
