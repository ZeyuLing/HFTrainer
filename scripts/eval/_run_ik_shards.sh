#!/usr/bin/env bash
# Launch N parallel HML263->SMPL motion_135 IK shards, one per GPU, on the
# current host. --skip-existing makes it safe to (re)run / resume; shards split
# the file list by index so two GPUs never touch the same clip.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD"

BASE=${BASE:-outputs/evaluation/mdm_h3d272_repro_1000s}
N=${N:-8}
REFINE_ITERS=${REFINE_ITERS:-80}
REFINE_LR=${REFINE_LR:-0.02}
LOGDIR="$BASE/ik_shards"
mkdir -p "$LOGDIR"

# Idempotent: drop any prior shards so a re-run never double-processes a shard.
pkill -9 -f "scripts/eval/hml263_to_smpl_ik.py" 2>/dev/null || true
sleep 2

# Workers are plain children of THIS script. The caller starts this script with
# `setsid nohup bash _run_ik_shards.sh &`, so the script is already a detached
# session leader; we `wait` at the end to keep it (and thus the workers) alive
# independent of the taiji_client exec session that launched it.
pids=""
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i python3 -u scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$BASE/mdm_263" --out-dir "$BASE/mdm_smpl135" \
    --model-dir ref_repo/MDM/body_models \
    --source-fps 20 --target-fps 30 --floor-align \
    --refine-iters "$REFINE_ITERS" --refine-lr "$REFINE_LR" \
    --rot6d-convention row \
    --device cuda --skip-existing \
    --num-shards "$N" --shard-index "$i" \
    > "$LOGDIR/shard_$i.log" 2>&1 < /dev/null &
  pids="$pids $!"
done
echo "launched $N IK shards on $(hostname); pids:$pids"
echo "$pids" > "$LOGDIR/pids.txt"
wait
echo "all $N IK shards finished on $(hostname)"
