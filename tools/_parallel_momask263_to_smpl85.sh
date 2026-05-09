#!/bin/bash
# Parallel joints2smpl fitting across 8 GPUs.
#
# Usage:
#   ./tools/_parallel_momask263_to_smpl85.sh <pred_dir_263> <out_dir_smpl85> [num_gpus] [num_iters]
#
# Example:
#   ./tools/_parallel_momask263_to_smpl85.sh \
#       work_dirs/momask_eval/momask_pred_263 \
#       work_dirs/momask_eval/momask_pred_smpl85 \
#       8 30
set -e

PRED_DIR=${1:-work_dirs/momask_eval/momask_pred_263}
OUT_DIR=${2:-work_dirs/momask_eval/momask_pred_smpl85}
NUM_GPUS=${3:-8}
NUM_ITERS=${4:-30}

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/_logs"

PIDS=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    LOG="$OUT_DIR/_logs/gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES=$gpu \
        python3 tools/momask263_to_smpl85_sharded.py \
            --pred_dir_263 "$PRED_DIR" \
            --out_dir_smpl85 "$OUT_DIR" \
            --shard_idx $gpu \
            --num_shards $NUM_GPUS \
            --num_iters $NUM_ITERS \
            --device cuda \
            > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "[+] launched GPU $gpu PID=${PIDS[-1]}  log=$LOG"
done

echo "[+] waiting for ${#PIDS[@]} workers ..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "  [!] PID $pid exited with non-zero status"
done

echo "[+] all workers done"
ls "$OUT_DIR" | wc -l
echo "  output files in $OUT_DIR"
