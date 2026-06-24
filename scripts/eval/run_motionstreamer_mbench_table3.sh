#!/usr/bin/env bash
# Re-measure MotionStreamer on MBench/Table 3 through the official 272 recover path.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
OUT=${OUT:-$ROOT/output/evaluation/table3_mbench/motionstreamer}
MBENCH_INFO=${MBENCH_INFO:-$ROOT/ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json}
PROMPT_JSON=${PROMPT_JSON:-$MBENCH_INFO}
NUM_SHARDS=${NUM_SHARDS:-2}
GPUS=${GPUS:-0,2}
MAX_SAMPLES=${MAX_SAMPLES:-0}
IDS=${IDS:-}
RUN_MBENCH=${RUN_MBENCH:-1}
FIXED_LENGTH=${FIXED_LENGTH:-0}

cd "$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:$ROOT/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt "$NUM_SHARDS" ]; then
  echo "GPUS=$GPUS has fewer entries than NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi
LAST_GPU="${GPU_LIST[$((NUM_SHARDS - 1))]}"

rm -rf "$OUT"
mkdir -p "$OUT/shards" "$OUT/logs"

echo "[motionstreamer-table3] start shards=$NUM_SHARDS gpus=$GPUS ids=${IDS:-all} fixed_length=$FIXED_LENGTH $(date -Is)" | tee "$OUT/logs/run.log"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_LIST[$shard]}"
  shard_out="$OUT/shards/shard_${shard}"
  max_args=()
  if [ "$MAX_SAMPLES" != "0" ]; then
    max_args=(--max-samples "$MAX_SAMPLES")
  fi
  fixed_args=()
  if [ "$FIXED_LENGTH" = "1" ]; then
    fixed_args=(--fixed-length)
  fi
  id_args=()
  if [ -n "$IDS" ]; then
    id_args=(--ids "$IDS")
  fi
  (
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/motionstreamer_mbench_infer.py \
      --prompt-json "$PROMPT_JSON" \
      --eval-info-json "$MBENCH_INFO" \
      --output-dir "$shard_out" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --gpu 0 \
      "${max_args[@]}" \
      "${id_args[@]}" \
      "${fixed_args[@]}"
  ) > "$OUT/logs/infer_shard_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "[motionstreamer-table3] launch shard=$shard gpu=$gpu pid=${pids[-1]}" | tee -a "$OUT/logs/run.log"
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
if [ "$rc" -ne 0 ]; then
  echo "[motionstreamer-table3] inference failed; see $OUT/logs/infer_shard_*.log" | tee -a "$OUT/logs/run.log"
  exit "$rc"
fi

mkdir -p "$OUT/mbench_eval_input" "$OUT/m272"
find "$OUT/mbench_eval_input" -maxdepth 1 -type l -name '*.npy' -delete
find "$OUT/m272" -maxdepth 1 -type l -name '*.npy' -delete
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  find "$OUT/shards/shard_${shard}/mbench_eval_input" -maxdepth 1 -name '*.npy' -print0 \
    | while IFS= read -r -d '' f; do
        ln -sf "$(realpath "$f")" "$OUT/mbench_eval_input/$(basename "$f")"
      done
  find "$OUT/shards/shard_${shard}/m272" -maxdepth 1 -name '*.npy' -print0 \
    | while IFS= read -r -d '' f; do
        ln -sf "$(realpath "$f")" "$OUT/m272/$(basename "$f")"
      done
done
echo "[motionstreamer-table3] merged eval_input=$(find "$OUT/mbench_eval_input" -maxdepth 1 -name '*.npy' | wc -l)" | tee -a "$OUT/logs/run.log"

python3 tools/validate_mbench_eval_input.py \
  --eval-input-dir "$OUT/mbench_eval_input" \
  --eval-info-json "$MBENCH_INFO" \
  --output-json "$OUT/mbench_eval_input_manifest.json" \
  2>&1 | tee "$OUT/validate.log"

if [ "$RUN_MBENCH" != "1" ]; then
  echo "[motionstreamer-table3] skip MBench evaluator RUN_MBENCH=$RUN_MBENCH" | tee -a "$OUT/logs/run.log"
  exit 0
fi

PHYS_DIMS=(
  Jitter_Degree
  Ground_Penetration
  Foot_Floating
  Foot_Sliding
  Dynamic_Degree
)
(
  cd "$ROOT/ref_repo/ViMoGen"
  PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$LAST_GPU" python3 evaluate_mbench.py \
    --evaluation_path "$OUT/mbench_eval_input" \
    --output_path "$OUT/mbench_results_5phys_local" \
    --full_info_json "$MBENCH_INFO" \
    --device cuda \
    --dimension "${PHYS_DIMS[@]}"
) 2>&1 | tee "$OUT/mbench_5phys.log"

echo "[motionstreamer-table3] done out=$OUT $(date -Is)" | tee -a "$OUT/logs/run.log"
