#!/usr/bin/env bash
# Fine-tune the position-aware G1 tracker on a pool of KIMODO-G1 generated .motion
# files so the tracker learns to follow the generator's actual output distribution.
#
# Usage:
#   cursor_finetune_tracker.sh EXP_NAME GPU_ID PROTO_DIR [EXTRA_PROTO_DIR...]
#
# Warmstarts from the current stable position-aware checkpoint.
set -euo pipefail

PROJECT_ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${PROJECT_ROOT}"

EXP_NAME="$1"
GPU_ID="$2"
shift 2
PROTO_DIRS=("$@")

PM_ROOT="${PROJECT_ROOT}/ref_repo/ProtoMotions"
POOL_DIR="${PROJECT_ROOT}/output/physflow_kimodo_g1/${EXP_NAME}_pool"
STABLE_CKPT="${PM_ROOT}/results/physflow_g1_xyvel_stable_isaacgym_train_v1/last.ckpt"
SEED_MOTIONS="${PM_ROOT}/data/motion_for_trackers/g1_bones_seed_mini.pt"

# --- Stage the motion pool: all generated .motion files ---
rm -rf "${POOL_DIR}"
mkdir -p "${POOL_DIR}"
n=0
for d in "${PROTO_DIRS[@]}"; do
  if [[ -d "${d}" ]]; then
    for f in "${d}"/*.motion; do
      [[ -e "${f}" ]] || continue
      cp -f "${f}" "${POOL_DIR}/"
      n=$((n+1))
    done
  fi
done
echo "[cursor-ft] staged ${n} .motion files into ${POOL_DIR}"
ls "${POOL_DIR}" | head -50

NUM_ENVS="${PHYSFLOW_NUM_ENVS:-512}"
BATCH_SIZE="${PHYSFLOW_BATCH_SIZE:-8192}"
MAX_STEPS="${PHYSFLOW_TRAINING_MAX_STEPS:-300000}"
SAVE_EVERY="${PHYSFLOW_SAVE_EVERY:-5}"

PHYSFLOW_EXPERIMENT_NAME="${EXP_NAME}" \
PHYSFLOW_MOTION_FILE="${POOL_DIR}" \
PHYSFLOW_CHECKPOINT="${STABLE_CKPT}" \
PHYSFLOW_EXPERIMENT_PATH="examples/experiments/mimic/physflow_g1_xy_offset_stable.py" \
PHYSFLOW_NUM_ENVS="${NUM_ENVS}" \
PHYSFLOW_BATCH_SIZE="${BATCH_SIZE}" \
PHYSFLOW_TRAINING_MAX_STEPS="${MAX_STEPS}" \
PHYSFLOW_SAVE_EVERY="${SAVE_EVERY}" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  bash scripts/embodied/launch_position_aware_g1_tracker_train.sh
