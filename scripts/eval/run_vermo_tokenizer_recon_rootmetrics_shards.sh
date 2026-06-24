#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "$ROOT" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

SIZE=${SIZE:?set SIZE to 1k/4k/16k/64k}
NUM_PERSON=${NUM_PERSON:-1}
NUM_SHARDS=${NUM_SHARDS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_DURATION=${MAX_DURATION:-12}
OUT_BASE=${OUT_BASE:-output/evaluation/vermo_tokenizer_recon/table3_0607_rootmetrics}
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"

case "$SIZE" in
  1k)
    CONFIG=${CONFIG:-configs/vermo/vermo_pretrain_1k_llama1b_wavtokenizer.py}
    TOKENIZER=${TOKENIZER:-checkpoints/vermo_vqvae2d_1k_rescale_iter47k}
    ;;
  4k)
    CONFIG=${CONFIG:-configs/vermo/vermo_pretrain_4k_llama1b_wavtokenizer.py}
    TOKENIZER=${TOKENIZER:-checkpoints/vermo_vqvae2d_4k_rescale_iter47k}
    ;;
  16k)
    CONFIG=${CONFIG:-configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer.py}
    TOKENIZER=${TOKENIZER:-checkpoints/vermo_vqvae2d_16k_rescale_iter47k}
    ;;
  64k)
    CONFIG=${CONFIG:-configs/vermo/vermo_pretrain_64k_llama1b_wavtokenizer.py}
    TOKENIZER=${TOKENIZER:-checkpoints/vermo_vqvae2d_64k_rescale_iter47k}
    ;;
  *)
    echo "unknown SIZE=$SIZE" >&2
    exit 2
    ;;
esac

if [ "$NUM_PERSON" = "2" ]; then
  ANNO=${ANNO:-data/annotation/vermo_recon_motionhub_2p_test_20260606.json}
  ID_ARGS=()
  OUT_DIR="$OUT_BASE/2p/$SIZE"
else
  ANNO=${ANNO:-data/annotation/vermo_recon_motionhub_1p_test_20260606.json}
  RAW_ID_LIST=${RAW_ID_LIST:-output/evaluation/table3_recon_baselines_0606/hml263_gt_1p_max12_min40/test.txt}
  HML_GT_DIR=${HML_GT_DIR:-output/evaluation/table3_recon_baselines_0606/hml263_gt_1p_max12_min40/new_joint_vecs}
  FILTERED_ID_LIST=${FILTERED_ID_LIST:-output/evaluation/table3_recon_baselines_0607_metricfix_qualityfiltered/hml263_gt_1p_max12_min40/test_quality_filtered.txt}
  if [ -z "${ID_LIST:-}" ]; then
    if [ ! -f "$FILTERED_ID_LIST" ]; then
      mkdir -p "$(dirname "$FILTERED_ID_LIST")" "$OUT_BASE/logs"
      python3 scripts/eval/build_hml263_quality_filtered_ids.py \
        --gt-dir "$HML_GT_DIR" \
        --ids "$RAW_ID_LIST" \
        --out-ids "$FILTERED_ID_LIST" \
        --out-json "$(dirname "$FILTERED_ID_LIST")/quality_filter_report.json" \
        > "$OUT_BASE/logs/hml263_quality_filter_for_vermo.log" 2>&1
    fi
    ID_LIST="$FILTERED_ID_LIST"
  fi
  ID_ARGS=(--id-list "$ID_LIST")
  OUT_DIR="$OUT_BASE/1p/$SIZE"
fi
mkdir -p "$OUT_DIR" "$LOG_DIR"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
echo "[vermo-rootmetrics] start size=$SIZE people=$NUM_PERSON shards=$NUM_SHARDS gpus=$GPUS out=$OUT_DIR $(date -Is)" | tee "$LOG_DIR/${NUM_PERSON}p_${SIZE}.driver.log"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=${GPU_ARR[$((shard % ${#GPU_ARR[@]}))]}
  shard_out="$OUT_DIR/shard_${shard}"
  mkdir -p "$shard_out"
  (
    CUDA_VISIBLE_DEVICES="$gpu" python3 tools/eval_vermo_tokenizer_recon.py \
      --config "$CONFIG" \
      --tokenizer-path "$TOKENIZER" \
      --anno-file "$ANNO" \
      --data-dir data/motionhub \
      --out-dir "$shard_out" \
      --num-person "$NUM_PERSON" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --max-duration "$MAX_DURATION" \
      --device cuda \
      "${ID_ARGS[@]}"
  ) > "$LOG_DIR/${NUM_PERSON}p_${SIZE}_shard_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "[vermo-rootmetrics] launch shard=$shard gpu=$gpu pid=${pids[-1]}" | tee -a "$LOG_DIR/${NUM_PERSON}p_${SIZE}.driver.log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
  echo "[vermo-rootmetrics] shard failure; see $LOG_DIR/${NUM_PERSON}p_${SIZE}_shard_*.log" | tee -a "$LOG_DIR/${NUM_PERSON}p_${SIZE}.driver.log"
  exit 1
fi

python3 tools/merge_vermo_tokenizer_recon.py \
  --inputs "$OUT_DIR"/shard_*/recon_metrics.json \
  --output "$OUT_DIR/merged/recon_metrics.json" \
  | tee -a "$LOG_DIR/${NUM_PERSON}p_${SIZE}.driver.log"

echo "[vermo-rootmetrics] done size=$SIZE people=$NUM_PERSON out=$OUT_DIR $(date -Is)" | tee -a "$LOG_DIR/${NUM_PERSON}p_${SIZE}.driver.log"
