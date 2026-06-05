#!/usr/bin/env bash
# Re-measure one HumanML3D-263 T2M baseline on MBench/Table 3.
#
# The generated 263D motions are converted directly to MBench raw joints
# (T, 22, 3). This intentionally avoids the old SMPL IK retargeting path.
set -euo pipefail

ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
if [ ! -d "$ROOT" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

METHOD=${METHOD:?Set METHOD to one of: mdm, t2mgpt, momask, motiongpt, motiongpt3, motionlcm}
NUM_SHARDS=${NUM_SHARDS:-8}
RESET=${RESET:-1}
SKIP_EXISTING=${SKIP_EXISTING:-0}
OUT_ROOT=${OUT_ROOT:-output/evaluation/table3_mbench}
ANNO=${ANNO:-data/annotation/mbench_450_hml263_prompts.json}
CAPTIONS=${CAPTIONS:-data/annotation/mbench_450_hml263_captions.json}
MBENCH_INFO=${MBENCH_INFO:-ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json}
MAX_SAMPLES=${MAX_SAMPLES:-0}
RUN_MBENCH=${RUN_MBENCH:-1}

GEN_DIR="$OUT_ROOT/hml263_generations/$METHOD"
METHOD_OUT="$OUT_ROOT/$METHOD"
LOG_DIR="$METHOD_OUT/logs"

if [ "$RESET" = "1" ]; then
  rm -rf "$GEN_DIR" "$METHOD_OUT"
fi
mkdir -p "$GEN_DIR" "$METHOD_OUT" "$LOG_DIR"

ensure_python_import() {
  local module="$1"
  local package="$2"
  if ! python3 - <<PY > "$LOG_DIR/import_${module}.log" 2>&1
import ${module}
print("ok")
PY
  then
    python3 -m pip install -q "$package" >> "$LOG_DIR/import_${module}.log" 2>&1
  fi
}

if [ ! -f "$ANNO" ] || [ ! -f "$CAPTIONS" ]; then
  python3 tools/prepare_mbench_hml263_prompts.py \
    --out-anno "$ANNO" \
    --out-captions "$CAPTIONS"
fi

case "$METHOD" in
  motiongpt|motiongpt3)
    ensure_python_import spacy spacy
    ;;
esac

COMMON_SKIP=()
COMMON_SKIP_HYPHEN=()
if [ "$SKIP_EXISTING" = "1" ]; then
  COMMON_SKIP=(--skip_existing)
  COMMON_SKIP_HYPHEN=(--skip-existing)
fi
COMMON_MAX=()
COMMON_MAX_HYPHEN=()
if [ "$MAX_SAMPLES" != "0" ]; then
  COMMON_MAX=(--max_samples "$MAX_SAMPLES")
  COMMON_MAX_HYPHEN=(--max-samples "$MAX_SAMPLES")
fi

run_method_shard() {
  local shard="$1"
  local gpu="$2"
  local log="$LOG_DIR/infer_shard_${shard}.log"

  case "$METHOD" in
    mdm)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/mdm_infer_hml3d263.py \
        --model_path ref_repo/MDM/save/humanml_enc_512_50steps/model000750000.pt \
        --anno_file "$ANNO" \
        --anno_data_dir data/motionhub \
        --rewritten_file "$CAPTIONS" \
        --caption_protocol rewritten \
        --out_dir "$GEN_DIR" \
        --num_shards "$NUM_SHARDS" \
        --shard_index "$shard" \
        --batch_size 16 \
        --device 0 \
        "${COMMON_MAX[@]}" \
        "${COMMON_SKIP[@]}" \
        > "$log" 2>&1
      ;;
    t2mgpt)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/t2mgpt_infer_hml3d263.py \
        --anno-file "$ANNO" \
        --caption-file "$CAPTIONS" \
        --out-dir "$GEN_DIR" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --batch-size 32 \
        "${COMMON_MAX_HYPHEN[@]}" \
        "${COMMON_SKIP_HYPHEN[@]}" \
        > "$log" 2>&1
      ;;
    momask)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/momask_infer_h3d_test.py \
        --momask_root ref_repo/Momask/momask-codes \
        --humanml3d_272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --anno_file "$ANNO" \
        --data_dir data/motionhub \
        --rewritten_file "$CAPTIONS" \
        --caption_protocol rewritten \
        --out_dir "$GEN_DIR" \
        --num_shards "$NUM_SHARDS" \
        --shard_index "$shard" \
        --batch_size 32 \
        --gumbel_sample \
        "${COMMON_MAX[@]}" \
        "${COMMON_SKIP[@]}" \
        > "$log" 2>&1
      ;;
    motiongpt)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/motiongpt_infer_hml3d263.py \
        --anno-file "$ANNO" \
        --caption-file "$CAPTIONS" \
        --out-dir "$GEN_DIR" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --batch-size 16 \
        "${COMMON_MAX_HYPHEN[@]}" \
        "${COMMON_SKIP_HYPHEN[@]}" \
        > "$log" 2>&1
      ;;
    motiongpt3)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/motiongpt3_infer_hml3d263.py \
        --anno_file "$ANNO" \
        --anno_data_dir data/motionhub \
        --rewritten_file "$CAPTIONS" \
        --caption_protocol rewritten \
        --out_dir "$GEN_DIR" \
        --num_shards "$NUM_SHARDS" \
        --shard_index "$shard" \
        --batch_size 8 \
        --guidance_scale 3.0 \
        "${COMMON_MAX[@]}" \
        "${COMMON_SKIP[@]}" \
        > "$log" 2>&1
      ;;
    motionlcm)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/mld_infer_hml3d263.py \
        --cfg ref_repo/MotionLCM/configs/motionlcm_t2m.yaml \
        --checkpoint ref_repo/MotionLCM/experiments_t2m/motionlcm_humanml/motionlcm_humanml_v1.ckpt \
        --anno_file "$ANNO" \
        --anno_data_dir data/motionhub \
        --rewritten_file "$CAPTIONS" \
        --caption_protocol rewritten \
        --out_dir "$GEN_DIR" \
        --num_shards "$NUM_SHARDS" \
        --shard_index "$shard" \
        --batch_size 16 \
        --num_inference_timesteps 1 \
        "${COMMON_MAX[@]}" \
        "${COMMON_SKIP[@]}" \
        > "$log" 2>&1
      ;;
    *)
      echo "Unsupported METHOD=$METHOD" >&2
      exit 2
      ;;
  esac
}

echo "[start] method=$METHOD num_shards=$NUM_SHARDS reset=$RESET $(date -Is)" | tee "$LOG_DIR/run.log"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((shard % 8))
  (
    run_method_shard "$shard" "$gpu"
    rc=$?
    echo "exit_code=$rc finished_at=$(date -Is)" > "$LOG_DIR/infer_shard_${shard}.status"
    exit "$rc"
  ) &
  pids+=("$!")
done

infer_rc=0
for job in "${pids[@]}"; do
  wait "$job" || infer_rc=1
done
if [ "$infer_rc" -ne 0 ]; then
  echo "[fail] inference failed; see $LOG_DIR/infer_shard_*.log" | tee -a "$LOG_DIR/run.log"
  exit "$infer_rc"
fi

find "$GEN_DIR" -maxdepth 1 -name '*.npy' | wc -l | awk '{print "[infer-count] hml263_npy="$1}' | tee -a "$LOG_DIR/run.log"

python3 tools/convert_hml263_to_mbench_joints.py \
  --in-dir "$GEN_DIR" \
  --out-dir "$METHOD_OUT" \
  --eval-info-json "$MBENCH_INFO" \
  --source-fps 20 \
  --target-fps 20 \
  --force \
  2>&1 | tee "$METHOD_OUT/convert.log"

python3 tools/validate_mbench_eval_input.py \
  --eval-input-dir "$METHOD_OUT/mbench_eval_input" \
  --eval-info-json "$MBENCH_INFO" \
  --output-json "$METHOD_OUT/mbench_eval_input_manifest.json" \
  2>&1 | tee "$METHOD_OUT/validate.log"

if [ "$RUN_MBENCH" != "1" ]; then
  echo "[skip] RUN_MBENCH=$RUN_MBENCH; official MBench evaluator not run." | tee -a "$LOG_DIR/run.log"
  exit 0
fi

NON_VLM_DIMS=(
  Jitter_Degree
  Ground_Penetration
  Foot_Floating
  Foot_Sliding
  Dynamic_Degree
  Body_Penetration
  Pose_Quality
)

(
  cd ref_repo/ViMoGen
  CUDA_VISIBLE_DEVICES=0 python evaluate_mbench.py \
    --evaluation_path "$ROOT/$METHOD_OUT/mbench_eval_input" \
    --output_path "$ROOT/$METHOD_OUT/mbench_results_non_vlm" \
    --full_info_json "$ROOT/$MBENCH_INFO" \
    --device cuda \
    --dimension "${NON_VLM_DIMS[@]}"
) 2>&1 | tee "$METHOD_OUT/mbench_non_vlm.log"

if [ -n "${GEMINI_API_KEY:-}" ]; then
  (
    cd ref_repo/ViMoGen
    CUDA_VISIBLE_DEVICES=0 python evaluate_mbench.py \
      --evaluation_path "$ROOT/$METHOD_OUT/mbench_eval_input" \
      --output_path "$ROOT/$METHOD_OUT/mbench_results_vlm" \
      --full_info_json "$ROOT/$MBENCH_INFO" \
      --device cuda \
      --gemini_api_key "$GEMINI_API_KEY" \
      --dimension Motion_Condition_Consistency Motion_Generalizability
  ) 2>&1 | tee "$METHOD_OUT/mbench_vlm.log"
else
  echo "GEMINI_API_KEY is not set; VLM metrics are pending." | tee "$METHOD_OUT/vlm_missing_key.txt"
fi

echo "[done] method=$METHOD out=$METHOD_OUT $(date -Is)" | tee -a "$LOG_DIR/run.log"
