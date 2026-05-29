#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
GPU="${GPU:-7}"
CKPT="${CKPT:-work_dirs/hymotion_m2m_v2_overfit_100_all_tasks_20260528/checkpoint-epoch_1900}"
OUT="${OUT:-work_dirs/hymotion_m2m_v2_overfit_100_all_tasks_20260528/eval_alltasks_epoch1900}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
NUM_STEPS="${NUM_STEPS:-50}"

cd "$REPO"
mkdir -p "$OUT"

LOG="$OUT/eval.log"
PIDFILE="$OUT/pid"

CUDA_VISIBLE_DEVICES="$GPU" nohup python3 scripts/eval/eval_m2m_v2_overfit100_alltasks.py \
  --checkpoint "$CKPT" \
  --max-samples "$MAX_SAMPLES" \
  --batch-size 1 \
  --num-workers 0 \
  --num-steps "$NUM_STEPS" \
  --replacement-guidance skip_last \
  --text-guidance-scale 1.0 \
  --output-dir "$OUT" \
  --save-npz \
  > "$LOG" 2>&1 &

PID="$!"
echo "$PID" > "$PIDFILE"
echo "PID=$PID"
echo "LOG=$LOG"
