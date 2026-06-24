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

MODE="${BEYONDMIMIC_MODE:-preflight}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TASK_FLAG="${BEYONDMIMIC_TASK_FLAG:-beyondmimic_${MODE}_${TIMESTAMP}}"
HOST_NUM="${BEYONDMIMIC_HOST_NUM:-1}"
HOST_GPU_NUM="${BEYONDMIMIC_HOST_GPU_NUM:-1}"
BUSINESS_FLAG="${BEYONDMIMIC_BUSINESS_FLAG:-AILab_DHA}"
MOTION_NAME="${BEYONDMIMIC_MOTION_NAME:-dance1_subject1}"
MAX_ITERATIONS="${BEYONDMIMIC_MAX_ITERATIONS:-20}"
NUM_ENVS="${BEYONDMIMIC_NUM_ENVS:-512}"

START_CMD=$(cat <<EOF
cd ${PROJECT_ROOT} && \
export BEYONDMIMIC_MODE=${MODE} && \
export BEYONDMIMIC_TAG=${TASK_FLAG} && \
export BEYONDMIMIC_MOTION_NAME=${MOTION_NAME} && \
export BEYONDMIMIC_MAX_ITERATIONS=${MAX_ITERATIONS} && \
export BEYONDMIMIC_NUM_ENVS=${NUM_ENVS} && \
bash scripts/embodied/taiji_beyondmimic_official_train.sh
EOF
)

python3 tools/taiji_submit.py "${TASK_FLAG}" __UNUSED__ \
  --host_num "${HOST_NUM}" \
  --host_gpu_num "${HOST_GPU_NUM}" \
  --gpu_name V100 \
  --business_flag "${BUSINESS_FLAG}" \
  --start-cmd "${START_CMD}"
