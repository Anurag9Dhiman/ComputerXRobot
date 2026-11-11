#!/bin/sh
# Vision-track baselines (learning/vision.py): behaviour cloning from pixels (vision-bc) and the
# diagnostic variant that also gets the next sub-goal (vision-bc-subgoal, vision_subgoal track).
#   PY=python WORKERS=7 sh scripts/run_vision.sh
# Steps whose outputs exist are skipped, so an interrupted run can be resumed.
set -e
PY=${PY:-python}
W=${WORKERS:-7}
N=${PER_TASK:-40}
EPOCHS=${EPOCHS:-30}
TASKS="type_key open_app close_window type_text clear_text open_file search open_result write_note"

# 1. Oracle demonstrations with the vision-track inputs: one process (and seed) per task.
i=0
for t in $TASKS; do
  [ -f data/vision_train_$t.npz ] || \
    $PY -m benchmark collect --split train --tasks $t --per-task $N --seed $((200 + i)) --vision \
      --out data/vision_train_$t.npz > data/vision_train_$t.log 2>&1 &
  i=$((i + 1))
done
wait
DATA=$(for t in $TASKS; do printf 'data/vision_train_%s.npz ' $t; done)

# 2. Three seeds of each variant. A seed's evaluation (CPU) overlaps the next seed's training (GPU).
for s in 0 1 2; do
  [ -f checkpoints/vbc_s$s.pt ] || $PY -m learning.vision train --data $DATA --epochs $EPOCHS --seed $s \
    --out checkpoints/vbc_s$s.pt
  [ -f checkpoints/vbcsg_s$s.pt ] || $PY -m learning.vision train --data $DATA --epochs $EPOCHS --seed $s --subgoal \
    --out checkpoints/vbcsg_s$s.pt
  wait
  ( [ -f results/vbc_s${s}_test_dynamics.json ] || \
      $PY -m benchmark run --agent vbc:checkpoints/vbc_s$s.pt --tag vbc_s$s --all-test --workers $W --out-dir results
    [ -f results/vbcsg_s${s}_test.json ] || \
      $PY -m benchmark run --agent vbc:checkpoints/vbcsg_s$s.pt --tag vbcsg_s$s --track vision_subgoal --split test \
        --workers $W --out-dir results ) &
done
wait
