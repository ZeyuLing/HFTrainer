#!/usr/bin/env bash
# Multi-GPU sharded PRISM overfit-100 cached-T5 evaluation.
# Runs NUM_GPUS shards in parallel (all children of THIS process) and waits.
# Keep the taiji_exec session open until this returns so children survive.
#
# Usage: run_prism_overfit_eval_sharded.sh <ckpt_epoch> [total_samples] [num_gpus]
set -uo pipefail

REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$REPO"

EPOCH=${1:-260}
TOTAL=${2:-100}
NUM_GPUS=${3:-8}

WORK_DIR=work_dirs/prism_overfit100_kt_toporesid_savefix_0529
CKPT=$WORK_DIR/checkpoint-epoch_${EPOCH}
POS_DIR=$WORK_DIR/eval_overfit_positions_epoch${EPOCH}
LOG_DIR=logs/prism_savefix_0529
CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100.py

mkdir -p "$LOG_DIR" "$POS_DIR"

PER=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
echo "[shard] epoch=$EPOCH total=$TOTAL gpus=$NUM_GPUS per_shard=$PER ckpt=$CKPT"

pids=()
for g in $(seq 0 $((NUM_GPUS - 1))); do
  START=$(( g * PER ))
  if [ "$START" -ge "$TOTAL" ]; then break; fi
  COUNT=$PER
  LOG=$LOG_DIR/eval_epoch${EPOCH}_shard${g}.out
  echo "[shard] GPU $g -> samples [$START, $((START+COUNT))) log=$LOG"
  CUDA_VISIBLE_DEVICES=$g python3 tools/eval_prism_overfit_cached_t5.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --start-index "$START" --num-samples "$COUNT" \
    --num-steps 50 --decode-frames 360 \
    --positions-dir "$POS_DIR" \
    --output "$WORK_DIR/eval_epoch${EPOCH}_shard${g}.json" \
    --progress > "$LOG" 2>&1 &
  pids+=("$!")
done

echo "[shard] launched ${#pids[@]} shards: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=$((fail+1))
done
echo "[shard] ALL_DONE failed_shards=$fail npz=$(ls "$POS_DIR" | wc -l)"
