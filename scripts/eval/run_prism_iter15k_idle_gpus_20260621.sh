#!/usr/bin/env bash
# Fill idle GPUs on a debug machine for strict official-272 iter15000 backfill.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

ID_FILE="${ID_FILE:?set ID_FILE to a newline-separated official id list}"
RUN_ROOT="${RUN_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_kafs_kt_compare_20260621}"
OUT="$RUN_ROOT/iter15k_no_kt_no_kafs/h3d"
CKPT="${CKPT:-/dev/shm/prism_iter15k_ckpt}"
GPU_LIST="${GPU_LIST:-1 3}"
read -r -a GPUS <<< "$GPU_LIST"
TOTAL="${TOTAL_SHARDS:-${#GPUS[@]}}"

mkdir -p "$OUT/none" "$OUT/_logs"
echo "[idle-iter15k] $(date -Is) id_file=$ID_FILE ckpt=$CKPT gpus=${GPU_LIST} total=$TOTAL"

pids=()
idx=0
for gpu in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_prism_kafs_ablation.py \
    --config configs/prism/prism_1b_tp2m_multiframe_iter15k.py \
    --checkpoint "$CKPT" \
    --kafs-mode none \
    --out-subdir none \
    --anno-file data/annotation/test_hml3d_official272_gtlen.json \
    --data-dir . \
    --output-dir "$OUT" \
    --id-file "$ID_FILE" \
    --min-frames 1 \
    --max-frames 360 \
    --num-inference-steps "${STEPS:-50}" \
    --guidance-scale "${GUIDANCE:-5.0}" \
    --translation-decode-mode absolute \
    --seed "${SEED:-42}" \
    --num-shards "$TOTAL" \
    --shard-idx "$idx" \
    --skip-existing \
    --smooth-output \
    --skip-motion-existence-check \
    > "$OUT/_logs/idle_gpu${gpu}_s${idx}of${TOTAL}.log" 2>&1 &
  pids+=("$!")
  idx=$((idx + 1))
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[idle-iter15k done] $(date -Is)"

