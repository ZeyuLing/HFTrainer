#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"

if [[ -z "${TOKEN:-}" ]]; then
  echo "ERROR: TOKEN is not set. Export TOKEN before submitting a Taiji task." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TASK_FLAG="${PHYSFLOW_TRACKER_TASK_FLAG:-physflow_g1_tracker_${TIMESTAMP}}"
HOST_NUM="${PHYSFLOW_HOST_NUM:-1}"
HOST_GPU_NUM="${PHYSFLOW_HOST_GPU_NUM:-1}"
BUSINESS_FLAG="${PHYSFLOW_BUSINESS_FLAG:-AILab_DHC_DC}"
PYTHON_CMD="${PHYSFLOW_PYTHON_CMD:-/usr/local/bin/python3}"
SIMULATOR="${PHYSFLOW_SIMULATOR:-isaacgym}"
EXPERIMENT_NAME="${PHYSFLOW_EXPERIMENT_NAME:-physflow_g1_xyvel_global_tracker_pool}"
MOTION_FILE="${PHYSFLOW_MOTION_FILE:-output/physflow_kimodo_g1/global_tracker_motion_pool_codex_20260529}"
TRAINING_MAX_STEPS="${PHYSFLOW_TRAINING_MAX_STEPS:-20000}"
NUM_ENVS="${PHYSFLOW_NUM_ENVS:-256}"
BATCH_SIZE="${PHYSFLOW_BATCH_SIZE:-4096}"
SAVE_EVERY="${PHYSFLOW_SAVE_EVERY:-5}"
CHECKPOINT="${PHYSFLOW_CHECKPOINT:-../../output/physflow_kimodo_g1/checkpoints/g1_xyvel_partial_warmstart.ckpt}"
EXPERIMENT_PATH="${PHYSFLOW_EXPERIMENT_PATH:-examples/experiments/mimic/physflow_g1_xy_offset_stable.py}"

START_CMD=$(cat <<EOF
cd ${PROJECT_ROOT} && \
export PHYSFLOW_PYTHON_CMD=${PYTHON_CMD} && \
export PHYSFLOW_SIMULATOR=${SIMULATOR} && \
export PHYSFLOW_EXPERIMENT_NAME=${EXPERIMENT_NAME} && \
export PHYSFLOW_MOTION_FILE=${MOTION_FILE} && \
export PHYSFLOW_TRAINING_MAX_STEPS=${TRAINING_MAX_STEPS} && \
export PHYSFLOW_NUM_ENVS=${NUM_ENVS} && \
export PHYSFLOW_BATCH_SIZE=${BATCH_SIZE} && \
export PHYSFLOW_SAVE_EVERY=${SAVE_EVERY} && \
export PHYSFLOW_CHECKPOINT=${CHECKPOINT} && \
export PHYSFLOW_EXPERIMENT_PATH=${EXPERIMENT_PATH} && \
bash scripts/embodied/launch_position_aware_g1_tracker_train.sh
EOF
)

python3 tools/taiji_submit.py "${TASK_FLAG}" __UNUSED__ \
  --host_num "${HOST_NUM}" \
  --host_gpu_num "${HOST_GPU_NUM}" \
  --business_flag "${BUSINESS_FLAG}" \
  --start-cmd "${START_CMD}"
