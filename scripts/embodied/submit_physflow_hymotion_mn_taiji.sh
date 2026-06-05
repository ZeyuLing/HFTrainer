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
TASK_FLAG="${PHYSFLOW_TASK_FLAG:-physflow_hymotion_real_mn4_${TIMESTAMP}}"
HOST_NUM="${PHYSFLOW_HOST_NUM:-4}"
HOST_GPU_NUM="${PHYSFLOW_HOST_GPU_NUM:-8}"
BUSINESS_FLAG="${PHYSFLOW_BUSINESS_FLAG:-AILab_DHC_DD}"
CONFIG="${PHYSFLOW_CONFIG:-configs/physflow/physflow_online_adv_mn_hymotion_real.py}"
TEXT_NSHARDS="${PHYSFLOW_TEXT_NSHARDS:-8}"
TEXT_BATCH_SIZE="${PHYSFLOW_TEXT_BATCH_SIZE:-16}"

START_CMD=$(cat <<EOF
cd ${PROJECT_ROOT} && \
export PHYSFLOW_TEXT_NSHARDS=${TEXT_NSHARDS} && \
export PHYSFLOW_TEXT_BATCH_SIZE=${TEXT_BATCH_SIZE} && \
bash tools/physflow_hymotion_mn_start.sh ${CONFIG} --auto-resume
EOF
)

python3 tools/taiji_submit.py "${TASK_FLAG}" "${CONFIG}" \
  --host_num "${HOST_NUM}" \
  --host_gpu_num "${HOST_GPU_NUM}" \
  --business_flag "${BUSINESS_FLAG}" \
  --start-cmd "${START_CMD}"
