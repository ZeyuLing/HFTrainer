#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
SIMULATOR="${PHYSFLOW_SIMULATOR:-isaacgym}"
EXPERIMENT_NAME="${PHYSFLOW_EXPERIMENT_NAME:-physflow_g1_xyvel_stable_isaacgym_train_v1}"
NUM_ENVS="${PHYSFLOW_NUM_ENVS:-256}"
BATCH_SIZE="${PHYSFLOW_BATCH_SIZE:-4096}"
TRAINING_MAX_STEPS="${PHYSFLOW_TRAINING_MAX_STEPS:-1048576}"
SAVE_EVERY="${PHYSFLOW_SAVE_EVERY:-5}"
MOTION_FILE="${PHYSFLOW_MOTION_FILE:-data/motion_for_trackers/g1_bones_seed_mini.pt}"
CHECKPOINT="${PHYSFLOW_CHECKPOINT:-../../output/physflow_kimodo_g1/checkpoints/g1_xyvel_partial_warmstart.ckpt}"
EXPERIMENT_PATH="${PHYSFLOW_EXPERIMENT_PATH:-examples/experiments/mimic/physflow_g1_xy_offset_stable.py}"
OUTPUT_DIR="${PROJECT_ROOT}/output/physflow_kimodo_g1/${EXPERIMENT_NAME}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[physflow] start $(date)"
echo "[physflow] project=${PROJECT_ROOT}"
echo "[physflow] simulator=${SIMULATOR}"
echo "[physflow] experiment=${EXPERIMENT_NAME}"
echo "[physflow] num_envs=${NUM_ENVS} batch_size=${BATCH_SIZE} max_steps=${TRAINING_MAX_STEPS}"
echo "[physflow] motion_file=${MOTION_FILE}"
echo "[physflow] checkpoint=${CHECKPOINT}"

cd "${PROJECT_ROOT}/ref_repo/ProtoMotions"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_DISABLE_SENTRY="${WANDB_DISABLE_SENTRY:-true}"

if [[ "${PHYSFLOW_USE_GCC_TOOLSET:-1}" == "1" ]]; then
    for toolset in /opt/rh/gcc-toolset-14/enable /opt/rh/gcc-toolset-13/enable /opt/rh/gcc-toolset-12/enable /opt/rh/gcc-toolset-11/enable /opt/rh/gcc-toolset-10/enable /opt/rh/gcc-toolset-9/enable; do
        if [[ -f "${toolset}" ]]; then
            # IsaacGym's gymtorch extension needs a newer compiler with PyTorch 2.x.
            # shellcheck disable=SC1090
            source "${toolset}"
            echo "[physflow] enabled compiler toolset: ${toolset}"
            break
        fi
    done
fi

if [[ -n "${PHYSFLOW_PYTHON_CMD:-}" ]]; then
    read -r -a PYTHON_CMD <<< "${PHYSFLOW_PYTHON_CMD}"
elif [[ "${SIMULATOR}" == "isaaclab" ]]; then
    if [[ -x /workspace/isaaclab/isaaclab.sh ]]; then
        PYTHON_CMD=(/workspace/isaaclab/isaaclab.sh -p)
    elif [[ -x /isaaclab/isaaclab.sh ]]; then
        PYTHON_CMD=(/isaaclab/isaaclab.sh -p)
    else
        PYTHON_CMD=(python3)
    fi
else
    PYTHON_CMD=(python3)
fi

echo "[physflow] python command: ${PYTHON_CMD[*]}"
"${PYTHON_CMD[@]}" - <<'PY'
import importlib.util
import sys

print("python", sys.version)
for name in ("torch", "isaaclab", "isaacgym", "mujoco", "lightning", "tensordict"):
    spec = importlib.util.find_spec(name)
    print(f"import_check {name}: {'OK' if spec else 'MISSING'}")
PY

if [[ "${PHYSFLOW_INSTALL_REQUIREMENTS:-0}" == "1" ]]; then
    if [[ "${SIMULATOR}" == "isaaclab" ]]; then
        "${PYTHON_CMD[@]}" -m pip install --no-cache-dir -r requirements_isaaclab.txt
    elif [[ "${SIMULATOR}" == "isaacgym" ]]; then
        "${PYTHON_CMD[@]}" -m pip install --no-cache-dir -r requirements_isaacgym.txt
    fi
fi

echo "[physflow] launching train_agent.py"
"${PYTHON_CMD[@]}" protomotions/train_agent.py \
    --robot-name g1 \
    --simulator "${SIMULATOR}" \
    --num-envs "${NUM_ENVS}" \
    --batch-size "${BATCH_SIZE}" \
    --motion-file "${MOTION_FILE}" \
    --experiment-path "${EXPERIMENT_PATH}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --training-max-steps "${TRAINING_MAX_STEPS}" \
    --checkpoint "${CHECKPOINT}" \
    --skip-initial-eval \
    --headless True \
    --overrides "agent.save_last_checkpoint_every=${SAVE_EVERY}"

echo "[physflow] finished $(date)"
