#!/usr/bin/env bash
# Clean restart of the position-aware G1 tracker WITHOUT catastrophic forgetting.
#
# Strategy (see docs/temp/physflow_online_adversarial_iteration_log.md "CRITICAL
# FINDING" + "Correct path forward"):
#   - warm-start from the RELEASED BeyondMimic G1 tracker weights (epoch reset to 0)
#   - keep the released architecture EXACTLY (include_xy_offset=False -> no input
#     structural change, which is what destroyed the prior fine-tunes)
#   - fine-tune on a MIXED pool: KIMODO motions + standard rehearsal motions
#   - evaluate the reference-reconstruction metrics frequently so we can judge
#     training effectiveness by the gt_error / max_joint_error / relative_body_pos
#     CURVE (not by survival / episode_length).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
SIMULATOR="isaacgym"
EXPERIMENT_NAME="${PHYSFLOW_EXPERIMENT_NAME:-physflow_g1_released_rehearsal_v1}"
NUM_ENVS="${PHYSFLOW_NUM_ENVS:-256}"
BATCH_SIZE="${PHYSFLOW_BATCH_SIZE:-4096}"
TRAINING_MAX_STEPS="${PHYSFLOW_TRAINING_MAX_STEPS:-40000000}"
EVAL_EVERY="${PHYSFLOW_EVAL_EVERY:-50}"
SAVE_EVERY="${PHYSFLOW_SAVE_EVERY:-10}"
SAVE_EPOCH_EVERY="${PHYSFLOW_SAVE_EPOCH_EVERY:-100}"
MOTION_FILE="${PHYSFLOW_MOTION_FILE:-${PROJECT_ROOT}/output/physflow_kimodo_g1/physflow_g1_released_rehearsal_v1_pool}"
CHECKPOINT="${PHYSFLOW_CHECKPOINT:-${PROJECT_ROOT}/output/physflow_kimodo_g1/checkpoints/g1_released_warmstart_epoch0.ckpt}"
EXPERIMENT_PATH="${PHYSFLOW_EXPERIMENT_PATH:-examples/experiments/mimic/physflow_g1_released_rehearsal.py}"
OUTPUT_DIR="${PROJECT_ROOT}/output/physflow_kimodo_g1/${EXPERIMENT_NAME}"
LOG_FILE="${OUTPUT_DIR}/train.log"
PYTHON_CMD="${PHYSFLOW_PYTHON_CMD:-/root/physflow_isaacgym_py38_cu118/bin/python}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[physflow] start $(date)"
echo "[physflow] experiment=${EXPERIMENT_NAME}"
echo "[physflow] motion_file=${MOTION_FILE}"
echo "[physflow] checkpoint=${CHECKPOINT}  (warm-start, epoch reset to 0)"
echo "[physflow] num_envs=${NUM_ENVS} batch=${BATCH_SIZE} max_steps=${TRAINING_MAX_STEPS} eval_every=${EVAL_EVERY}"

cd "${PROJECT_ROOT}/ref_repo/ProtoMotions"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export WANDB_SILENT=true WANDB_DISABLE_SENTRY=true

for toolset in /opt/rh/gcc-toolset-14/enable /opt/rh/gcc-toolset-13/enable /opt/rh/gcc-toolset-12/enable /opt/rh/gcc-toolset-11/enable; do
    if [[ -f "${toolset}" ]]; then source "${toolset}"; echo "[physflow] toolset ${toolset}"; break; fi
done

echo "[physflow] python: ${PYTHON_CMD}"
"${PYTHON_CMD}" -c "import isaacgym, torch, mujoco, lightning; print('imports OK', torch.__version__)" || {
    echo "[physflow] import check FAILED"; exit 1; }

echo "[physflow] launching train_agent.py (NO --skip-initial-eval => capture epoch-0 reconstruction baseline)"
"${PYTHON_CMD}" protomotions/train_agent.py \
    --robot-name g1 \
    --simulator "${SIMULATOR}" \
    --num-envs "${NUM_ENVS}" \
    --batch-size "${BATCH_SIZE}" \
    --motion-file "${MOTION_FILE}" \
    --experiment-path "${EXPERIMENT_PATH}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --training-max-steps "${TRAINING_MAX_STEPS}" \
    --checkpoint "${CHECKPOINT}" \
    --headless True \
    --overrides \
        "agent.evaluator.eval_metrics_every=${EVAL_EVERY}" \
        "agent.save_last_checkpoint_every=${SAVE_EVERY}" \
        "agent.save_epoch_checkpoint_every=${SAVE_EPOCH_EVERY}" \
        ${PHYSFLOW_EXTRA_OVERRIDES:-}

echo "[physflow] finished $(date)"
