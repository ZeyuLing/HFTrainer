#!/usr/bin/env bash
# Parallel G1-native T2M eval across checkpoints, one GPU per checkpoint.
# Usage: bash scripts/embodied/run_g1_eval_parallel.sh "1000 2000 3000"
set -u
ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
if [ ! -d "$ROOT" ]; then ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer; fi
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

ITERS=${1:-"1000 2000 3000"}
CFG=configs/physflow/hymotion_g1_t2m_38dim.py
ANNO=data/annotation/train_g1_t2m_overfit100.json
LOGDIR=/tmp/g1_eval_logs
mkdir -p "$LOGDIR"

gpu=0
pids=()
for it in $ITERS; do
  CKPT="work_dirs/hymotion_g1_t2m_38dim/checkpoint-iter_${it}"
  if [ ! -f "$CKPT/model.safetensors" ]; then echo "SKIP iter_$it (no ckpt)"; continue; fi
  OUT="output/g1_t2m_eval/iter_${it}"
  echo "launch iter_$it on GPU$gpu -> $OUT (log $LOGDIR/iter_$it.log)"
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/embodied/eval_overfit_g1_t2m.py \
    --config "$CFG" --checkpoint "$CKPT" --anno "$ANNO" \
    --num-clips 100 --batch-size 10 --num-steps 50 --guidance 1.0 --det \
    --out-dir "$OUT" --save-npz 12 > "$LOGDIR/iter_${it}.log" 2>&1 &
  pids+=($!)
  gpu=$((gpu+1))
done

echo "launched ${#pids[@]} jobs, pids: ${pids[*]}"
wait
echo "ALL_EVAL_DONE"
