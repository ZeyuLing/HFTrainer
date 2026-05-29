#!/usr/bin/env bash
# Launch PRISM overfit-100 cached-T5 evaluation, detached so it survives the
# taiji_exec PTY session closing. Saves per-sample positions NPZ + summary JSON.
#
# Usage: run_prism_overfit_eval.sh <ckpt_epoch> [num_samples] [gpu_id]
set -euo pipefail

REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$REPO"

EPOCH=${1:-260}
NUM_SAMPLES=${2:-100}
GPU=${3:-0}

WORK_DIR=work_dirs/prism_overfit100_kt_toporesid_savefix_0529
CKPT=$WORK_DIR/checkpoint-epoch_${EPOCH}
POS_DIR=$WORK_DIR/eval_overfit_positions_epoch${EPOCH}
OUT_JSON=$WORK_DIR/eval_epoch${EPOCH}_${NUM_SAMPLES}x50.json
LOG_DIR=logs/prism_savefix_0529
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/eval_epoch${EPOCH}_${NUM_SAMPLES}x50.out

CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100.py

CMD="CUDA_VISIBLE_DEVICES=$GPU python3 tools/eval_prism_overfit_cached_t5.py \
  --config $CONFIG \
  --checkpoint $CKPT \
  --num-samples $NUM_SAMPLES --num-steps 50 --decode-frames 360 \
  --positions-dir $POS_DIR \
  --output $OUT_JSON \
  --progress"

echo "[launcher] $(date) launching: $CMD" > "$LOG"
setsid bash -c "$CMD" >> "$LOG" 2>&1 < /dev/null &
echo "[launcher] detached PID $!"
