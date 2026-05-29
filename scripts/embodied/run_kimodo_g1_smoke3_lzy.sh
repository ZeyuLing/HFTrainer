#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUT_DIR="${1:-output/physflow_kimodo_g1/smoke_lzy2_caption_rp_3prompts}"
GPU_ID="${2:-1}"
SEED="${3:-42}"
PYTHON_CMD="${PHYSFLOW_PYTHON_CMD:-python3}"
SAMPLES_PER_PROMPT="${PHYSFLOW_SAMPLES_PER_PROMPT:-1}"
MODE="${PHYSFLOW_MODE:-loop-smoke}"
PROMPT_BANK="${PHYSFLOW_PROMPT_BANK:-configs/experiments/physflow_kimodo_g1/prompt_bank_v0.jsonl}"
PROMPT_SPLIT="${PHYSFLOW_PROMPT_SPLIT:-smoke}"
MAX_PROMPTS="${PHYSFLOW_MAX_PROMPTS:-3}"
HARD_CASES="${PHYSFLOW_HARD_CASES:-8}"
HARD_MIN_SCORE="${PHYSFLOW_HARD_MIN_SCORE:-1.0}"
GOOD_CASES="${PHYSFLOW_GOOD_CASES:-8}"
GOOD_MIN_COMPLETION="${PHYSFLOW_GOOD_MIN_COMPLETION:-0.95}"
GOOD_MAX_JOINT_ERROR="${PHYSFLOW_GOOD_MAX_JOINT_ERROR:-0.7}"
GOOD_MAX_ROOT_TRAJECTORY_ERROR="${PHYSFLOW_GOOD_MAX_ROOT_TRAJECTORY_ERROR:-0.25}"
GOOD_MAX_ROOT_DISPLACEMENT_ERROR="${PHYSFLOW_GOOD_MAX_ROOT_DISPLACEMENT_ERROR:-0.35}"

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${PYTHON_CMD}" scripts/embodied/physflow_kimodo_g1_runner.py \
  --mode "${MODE}" \
  --output-dir "${OUT_DIR}" \
  --prompt-bank "${PROMPT_BANK}" \
  --prompt-split "${PROMPT_SPLIT}" \
  --max-prompts "${MAX_PROMPTS}" \
  --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --hard-cases "${HARD_CASES}" \
  --hard-min-score "${HARD_MIN_SCORE}" \
  --good-cases "${GOOD_CASES}" \
  --good-min-completion "${GOOD_MIN_COMPLETION}" \
  --good-max-joint-error "${GOOD_MAX_JOINT_ERROR}" \
  --good-max-root-trajectory-error "${GOOD_MAX_ROOT_TRAJECTORY_ERROR}" \
  --good-max-root-displacement-error "${GOOD_MAX_ROOT_DISPLACEMENT_ERROR}" \
  --max-difficulty 0 \
  --kimodo-model Kimodo-G1-RP-v1 \
  --diffusion-steps 100 \
  --seed "${SEED}" \
  --cfg-type separated \
  --cfg-weight 2.0 2.0 \
  --local-cache \
  --require-ready \
  --robot-json-subsample 1
