#!/usr/bin/env bash
# Re-measure ViMoGen on the 450 MBench/Table-3 prompts.
set -euo pipefail

ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
if [ ! -d "$ROOT" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

TAG=${TAG:-table3_mbench_0605}
NUM_SHARDS=${NUM_SHARDS:-8}
OUT_ROOT=${OUT_ROOT:-output/evaluation/table3_mbench/vimogen}
RUN_ROOT="$OUT_ROOT/runs"
LOG_DIR="$OUT_ROOT/logs"
MBENCH_INFO=${MBENCH_INFO:-ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json}
mkdir -p "$RUN_ROOT" "$LOG_DIR"

echo "[start] ViMoGen MBench tag=$TAG shards=$NUM_SHARDS $(date -Is)" | tee "$LOG_DIR/run.log"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((shard % 8))
  run_tag="mbench_${TAG}_s${shard}of${NUM_SHARDS}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export ENC_GPU="$gpu"
    export DATASET=mbench
    export MAX_SAMPLES=0
    export TAG="$TAG"
    export NUM_SHARDS="$NUM_SHARDS"
    export SHARD_IDX="$shard"
    export NPROC=1
    export TEST_BS=${TEST_BS:-4}
    export STEPS=${STEPS:-50}
    export CFG=${CFG:-5.0}
    export DENOISING_STRENGTH=${DENOISING_STRENGTH:-1.0}
    export DTYPE=${DTYPE:-fp16}
    export SKIP_EVAL=1
    export VIMOGEN_DST_FPS=20
    export VIMOGEN_TEXT_BATCH_SIZE=${VIMOGEN_TEXT_BATCH_SIZE:-4}
    export OUT_ROOT="$RUN_ROOT/$run_tag"
    export MASTER_PORT=$((31400 + shard))
    bash scripts/eval/run_vimogen_t2m_eval_0605.sh
    echo "exit_code=0 finished_at=$(date -Is)" > "$LOG_DIR/shard_${shard}.status"
  ) > "$LOG_DIR/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
if [ "$rc" -ne 0 ]; then
  echo "[fail] ViMoGen shard inference failed; see $LOG_DIR/shard_*.log" | tee -a "$LOG_DIR/run.log"
  exit "$rc"
fi

MERGED_INPUT="$OUT_ROOT/mbench_eval_input"
rm -rf "$MERGED_INPUT"
mkdir -p "$MERGED_INPUT"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  src="$RUN_ROOT/mbench_${TAG}_s${shard}of${NUM_SHARDS}/mbench/mbench_eval_input"
  test -d "$src"
  find "$src" -maxdepth 1 -name '*.npy' -print0 \
    | while IFS= read -r -d '' f; do
        ln -sf "$(realpath "$f")" "$MERGED_INPUT/$(basename "$f")"
      done
done

python3 tools/validate_mbench_eval_input.py \
  --eval-input-dir "$MERGED_INPUT" \
  --eval-info-json "$MBENCH_INFO" \
  --output-json "$OUT_ROOT/mbench_eval_input_manifest.json" \
  2>&1 | tee "$OUT_ROOT/validate.log"

PHYS_DIMS=(
  Jitter_Degree
  Ground_Penetration
  Foot_Floating
  Foot_Sliding
  Dynamic_Degree
)
(
  cd ref_repo/ViMoGen
  PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 python3 evaluate_mbench.py \
    --evaluation_path "$ROOT/$MERGED_INPUT" \
    --output_path "$ROOT/$OUT_ROOT/mbench_results_5phys" \
    --full_info_json "$ROOT/$MBENCH_INFO" \
    --device cuda \
    --dimension "${PHYS_DIMS[@]}"
) 2>&1 | tee "$OUT_ROOT/mbench_5phys.log"

echo "[done] ViMoGen MBench out=$OUT_ROOT $(date -Is)" | tee -a "$LOG_DIR/run.log"
