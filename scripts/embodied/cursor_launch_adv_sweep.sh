#!/usr/bin/env bash
# Reusable launcher for PhysFlow KIMODO-G1 adversarial sweeps (background via nohup).
# Usage:
#   cursor_launch_adv_sweep.sh OUT_DIR GPU_ID SEED SPLIT MAX_PROMPTS SAMPLES_PER_PROMPT [PROMPT_BANK]
set -euo pipefail

PROJECT_ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${PROJECT_ROOT}"

OUT_DIR="$1"
GPU_ID="$2"
SEED="$3"
SPLIT="$4"
MAX_PROMPTS="$5"
SAMPLES_PER_PROMPT="$6"
PROMPT_BANK="${7:-configs/experiments/physflow_kimodo_g1/prompt_bank_v0.jsonl}"

ONNX="ref_repo/ProtoMotions/results/physflow_g1_xyvel_stable_isaacgym_train_v1/compiled_models/unified_pipeline.onnx"

mkdir -p "${OUT_DIR}"

export PHYSFLOW_PYTHON_CMD=/usr/local/bin/python3
export PHYSFLOW_G1_ONNX="${ONNX}"
export PHYSFLOW_MODE=adv-sweep
export PHYSFLOW_PROMPT_BANK="${PROMPT_BANK}"
export PHYSFLOW_PROMPT_SPLIT="${SPLIT}"
export PHYSFLOW_MAX_PROMPTS="${MAX_PROMPTS}"
export PHYSFLOW_SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT}"
export PHYSFLOW_HARD_CASES="${PHYSFLOW_HARD_CASES:-20}"
export PHYSFLOW_GOOD_CASES="${PHYSFLOW_GOOD_CASES:-60}"
export PHYSFLOW_HARD_MIN_SCORE="${PHYSFLOW_HARD_MIN_SCORE:-1.0}"

nohup bash scripts/embodied/run_kimodo_g1_smoke3_lzy.sh "${OUT_DIR}" "${GPU_ID}" "${SEED}" \
  > "${OUT_DIR}/nohup.log" 2>&1 &
PID=$!
disown "${PID}" 2>/dev/null || true
echo "LAUNCHED pid=${PID} out=${OUT_DIR} gpu=${GPU_ID} split=${SPLIT} prompts=${MAX_PROMPTS} samples=${SAMPLES_PER_PROMPT}"
