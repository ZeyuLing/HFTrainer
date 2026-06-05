#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"

if [[ -z "${TOKEN:-}" ]]; then
  for token_file in /root/.claude-dashboard/taiji_token /root/.codex/skills/taiji/.token; do
    if [[ -r "${token_file}" ]]; then
      TOKEN="$(<"${token_file}")"
      export TOKEN
      break
    fi
  done
fi
if [[ -z "${TOKEN:-}" ]]; then
  echo "ERROR: TOKEN is not set and no readable Taiji token file was found." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TASK_FLAG="${PHYSFLOW_JUMP_TASK_FLAG:-physflow_jump_overfit_hml3d40_${TIMESTAMP}}"
HOST_NUM="${PHYSFLOW_HOST_NUM:-1}"
HOST_GPU_NUM="${PHYSFLOW_HOST_GPU_NUM:-8}"
BUSINESS_FLAG="${PHYSFLOW_BUSINESS_FLAG:-AILab_DHC_DD}"

JUMP_TAG="${PHYSFLOW_JUMP_TAG:-jump_hml3d40_noscene_mn1500_reset_env256_40m_noproj}"
PROMPT_BANK="${PHYSFLOW_JUMP_PROMPT_BANK:-configs/experiments/physflow_kimodo_g1/physflow_jump_overfit_prompts_hml3d_g1_noscene.jsonl}"
CONFIG="${PHYSFLOW_CONFIG:-configs/physflow/physflow_online_adv_mn.py}"
CKPT="${PHYSFLOW_CKPT:-work_dirs/physflow_online_adv_mn/checkpoint-iter_1500}"
NGPU="${PHYSFLOW_NGPU:-8}"
TRAINING_MAX_STEPS="${PHYSFLOW_TRAINING_MAX_STEPS:-40000000}"
NUM_ENVS="${PHYSFLOW_NUM_ENVS:-256}"
NUM_STEPS="${PHYSFLOW_NUM_STEPS:-32}"

START_CMD=$(cat <<EOF
cd ${PROJECT_ROOT} && \
PHYSFLOW_NODE_SETUP_ONLY=1 bash scripts/embodied/cursor_physflow_taiji_node_setup.sh && \
export PHYSFLOW_JUMP_TAG=${JUMP_TAG} && \
export PHYSFLOW_JUMP_PROMPT_BANK=${PROMPT_BANK} && \
export PHYSFLOW_CONFIG=${CONFIG} && \
export PHYSFLOW_CKPT=${CKPT} && \
export PHYSFLOW_NGPU=${NGPU} && \
export PHYSFLOW_TRAINING_MAX_STEPS=${TRAINING_MAX_STEPS} && \
export PHYSFLOW_NUM_ENVS=${NUM_ENVS} && \
export PHYSFLOW_NUM_STEPS=${NUM_STEPS} && \
bash scripts/embodied/launch_jump_overfit_tracker.sh
EOF
)

python3 tools/taiji_submit.py "${TASK_FLAG}" __UNUSED__ \
  --host_num "${HOST_NUM}" \
  --host_gpu_num "${HOST_GPU_NUM}" \
  --business_flag "${BUSINESS_FLAG}" \
  --start-cmd "${START_CMD}"
