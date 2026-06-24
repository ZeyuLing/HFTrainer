#!/usr/bin/env bash
# Re-measure HY-Motion-1.0-Lite on MBench/Table 3.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
OUT=${OUT:-$ROOT/output/evaluation/table3_mbench/hymotion_lite}
MBENCH_INFO=${MBENCH_INFO:-$ROOT/ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json}
NUM_SHARDS=${NUM_SHARDS:-2}
GPUS=${GPUS:-1,7}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-5.0}
MAX_SAMPLES=${MAX_SAMPLES:-0}

cd "$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt "$NUM_SHARDS" ]; then
  echo "GPUS=$GPUS has fewer entries than NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi
LAST_GPU="${GPU_LIST[$((NUM_SHARDS - 1))]}"

rm -rf "$OUT"
mkdir -p "$OUT/shards" "$OUT/logs"

echo "[hylite-table3] start shards=$NUM_SHARDS gpus=$GPUS steps=$NUM_STEPS cfg=$CFG_SCALE $(date -Is)" | tee "$OUT/logs/run.log"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_LIST[$shard]}"
  shard_out="$OUT/shards/shard_${shard}"
  max_args=()
  if [ "$MAX_SAMPLES" != "0" ]; then
    max_args=(--max-samples "$MAX_SAMPLES")
  fi
  (
    python3 scripts/eval/hylite_mbench_infer.py \
      --output-dir "$shard_out" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --gpu "$gpu" \
      --batch-size "$BATCH_SIZE" \
      --num-steps "$NUM_STEPS" \
      --cfg-scale "$CFG_SCALE" \
      "${max_args[@]}"
  ) > "$OUT/logs/infer_shard_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "[hylite-table3] launch shard=$shard gpu=$gpu pid=${pids[-1]}" | tee -a "$OUT/logs/run.log"
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
if [ "$rc" -ne 0 ]; then
  echo "[hylite-table3] inference failed; see $OUT/logs/infer_shard_*.log" | tee -a "$OUT/logs/run.log"
  exit "$rc"
fi

mkdir -p "$OUT/m135"
find "$OUT/m135" -maxdepth 1 -type l -name '*.npy' -delete
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  find "$OUT/shards/shard_${shard}/m135" -maxdepth 1 -name '*.npy' -print0 \
    | while IFS= read -r -d '' f; do
        ln -sf "$(realpath "$f")" "$OUT/m135/$(basename "$f")"
      done
done
echo "[hylite-table3] merged m135=$(find "$OUT/m135" -maxdepth 1 -name '*.npy' | wc -l)" | tee -a "$OUT/logs/run.log"

CUDA_VISIBLE_DEVICES="$LAST_GPU" python3 tools/convert_motion135_to_mbench_joints.py \
  --in-dir "$OUT/m135" \
  --out-dir "$OUT" \
  --source-fps 30 \
  --target-fps 20 \
  --match-eval-frames \
  --device cuda \
  --force \
  2>&1 | tee "$OUT/convert.log"

python3 tools/validate_mbench_eval_input.py \
  --eval-input-dir "$OUT/mbench_eval_input" \
  --eval-info-json "$MBENCH_INFO" \
  --output-json "$OUT/mbench_eval_input_manifest.json" \
  2>&1 | tee "$OUT/validate.log"

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

echo "[hylite-table3] done out=$OUT $(date -Is)" | tee -a "$OUT/logs/run.log"
