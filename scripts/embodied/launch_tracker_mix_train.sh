#!/usr/bin/env bash
# Train a G1 tracker on an explicit weighted motion yaml/directory.
#
# This skips KIMODO generation and is intended for short replay-mix iterations.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${PROJECT_ROOT}"

MOTION_FILE="${PHYSFLOW_TRACKER_MOTION_FILE:?Set PHYSFLOW_TRACKER_MOTION_FILE to a .yaml/.motion directory}"
EXPERIMENT_NAME="${PHYSFLOW_EXPERIMENT_NAME:-physflow_g1_tracker_mix_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_PATH="${PHYSFLOW_EXPERIMENT_PATH:-examples/experiments/mimic/physflow_g1_released_rehearsal.py}"
CHECKPOINT="${PHYSFLOW_CHECKPOINT:-${PROJECT_ROOT}/output/physflow_kimodo_g1/checkpoints/g1_released_warmstart_epoch0_warmopt.ckpt}"
TRAINING_MAX_STEPS="${PHYSFLOW_TRAINING_MAX_STEPS:-4000000}"
NUM_ENVS="${PHYSFLOW_NUM_ENVS:-256}"
NUM_STEPS="${PHYSFLOW_NUM_STEPS:-32}"
BATCH_SIZE="${PHYSFLOW_BATCH_SIZE:-4096}"
NGPU="${PHYSFLOW_NGPU:-8}"
SAVE_EVERY="${PHYSFLOW_SAVE_EVERY:-5}"
EVAL_EVERY="${PHYSFLOW_EVAL_EVERY:-999999}"

LOG_DIR="${PROJECT_ROOT}/output/physflow_reports/20260604_tracker_mix"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/${EXPERIMENT_NAME}.log") 2>&1

echo "[tracker-mix] start $(date)"
echo "[tracker-mix] host=$(hostname)"
echo "[tracker-mix] motion_file=${MOTION_FILE}"
echo "[tracker-mix] experiment=${EXPERIMENT_NAME}"
echo "[tracker-mix] checkpoint=${CHECKPOINT}"
echo "[tracker-mix] ngpu=${NGPU} envs=${NUM_ENVS} rollout_steps=${NUM_STEPS} max_steps=${TRAINING_MAX_STEPS}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/[tracker-mix] gpu /' || true

cd "${PROJECT_ROOT}/ref_repo/ProtoMotions"
export PYTHONPATH="${PWD}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_DISABLE_SENTRY="${WANDB_DISABLE_SENTRY:-true}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/.cache/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ "${PHYSFLOW_USE_GCC_TOOLSET:-1}" == "1" ]]; then
    for toolset in /opt/rh/gcc-toolset-14/enable /opt/rh/gcc-toolset-13/enable /opt/rh/gcc-toolset-12/enable /opt/rh/gcc-toolset-11/enable /opt/rh/gcc-toolset-10/enable /opt/rh/gcc-toolset-9/enable; do
        if [[ -f "${toolset}" ]]; then
            # shellcheck disable=SC1090
            source "${toolset}"
            echo "[tracker-mix] enabled compiler toolset: ${toolset}"
            break
        fi
    done
fi

if [[ -n "${PHYSFLOW_TRACKER_PYTHON_CMD:-}" ]]; then
    read -r -a TRACKER_PY <<< "${PHYSFLOW_TRACKER_PYTHON_CMD}"
elif [[ -x /root/physflow_isaacgym_py38_cu118/bin/python ]]; then
    TRACKER_PY=(/root/physflow_isaacgym_py38_cu118/bin/python)
else
    TRACKER_PY=(python3)
fi

"${TRACKER_PY[@]}" - "${CHECKPOINT}" "${TRAINING_MAX_STEPS}" <<'PY'
import os
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
training_max_steps = int(sys.argv[2])
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
epoch = int(state.get("epoch") or 0)
step_count = int(state.get("step_count") or 0)
print(
    "[tracker-mix] checkpoint_meta "
    f"epoch={epoch} step_count={step_count} "
    f"skip_optimizer_load={state.get('skip_optimizer_load')} "
    f"best_evaluated_score={state.get('best_evaluated_score')}"
)
if os.environ.get("PHYSFLOW_ALLOW_NONZERO_CHECKPOINT") != "1" and (epoch != 0 or step_count != 0):
    raise SystemExit("[tracker-mix] ERROR: warm-start checkpoint must be epoch=0 step_count=0")
if step_count >= training_max_steps:
    raise SystemExit("[tracker-mix] ERROR: checkpoint step_count already >= training_max_steps")
PY

"${TRACKER_PY[@]}" protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaacgym \
    --num-envs "${NUM_ENVS}" \
    --batch-size "${BATCH_SIZE}" \
    --motion-file "${MOTION_FILE}" \
    --experiment-path "${EXPERIMENT_PATH}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --training-max-steps "${TRAINING_MAX_STEPS}" \
    --checkpoint "${CHECKPOINT}" \
    --ngpu "${NGPU}" \
    --nodes 1 \
    --skip-initial-eval \
    --headless True \
    --overrides \
        "agent.save_last_checkpoint_every=${SAVE_EVERY}" \
        "agent.evaluator.eval_metrics_every=${EVAL_EVERY}" \
        "agent.num_steps=${NUM_STEPS}" \
        ${PHYSFLOW_EXTRA_OVERRIDES:-}

echo "[tracker-mix] done $(date)"
echo "[tracker-mix] result=${PROJECT_ROOT}/ref_repo/ProtoMotions/results/${EXPERIMENT_NAME}"
