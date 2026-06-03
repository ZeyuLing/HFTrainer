#!/usr/bin/env bash
# Launch the position-aware G1 tracker OVERFIT sanity run on the debug machine.
#
# - architecture: physflow_g1_xy_offset_overfit.py (369-d, xy_offset=True),
#   domain randomization + reset noise disabled for a clean reconstruction curve.
# - warm start: the FIXED interleaved warm-start (bit-identical to released at init).
# - data: the KIMODO-G1 motions generated from HumanML3D prompts (overfit pool).
#
# If reconstruction error (eval/gt_error, eval/max_joint_error) drives toward ~0
# under these clean conditions, the pipeline is correct.
set -euo pipefail

export PROJECT_ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PHYSFLOW_PYTHON_CMD="${PHYSFLOW_PYTHON_CMD:-/root/physflow_isaacgym_py38_cu118/bin/python}"
export PHYSFLOW_SIMULATOR=isaacgym
export PHYSFLOW_EXPERIMENT_NAME="${PHYSFLOW_EXPERIMENT_NAME:-physflow_g1_xyvel_overfit100_FIXED}"
export PHYSFLOW_EXPERIMENT_PATH="${PHYSFLOW_EXPERIMENT_PATH:-examples/experiments/mimic/physflow_g1_xy_offset_overfit.py}"
export PHYSFLOW_CHECKPOINT="${PHYSFLOW_CHECKPOINT:-../../output/physflow_kimodo_g1/checkpoints/g1_xyvel_partial_warmstart_FIXED.ckpt}"
export PHYSFLOW_MOTION_FILE="${PHYSFLOW_MOTION_FILE:-output/physflow_kimodo_g1/overfit100_pool/proto}"
# train_agent.py runs with cwd=ref_repo/ProtoMotions, so a project-relative motion
# path won't resolve. Absolutize it against PROJECT_ROOT if it isn't already.
case "${PHYSFLOW_MOTION_FILE}" in
  /*) : ;;
  *) export PHYSFLOW_MOTION_FILE="${PROJECT_ROOT}/${PHYSFLOW_MOTION_FILE}" ;;
esac
export PHYSFLOW_NUM_ENVS="${PHYSFLOW_NUM_ENVS:-2048}"
export PHYSFLOW_BATCH_SIZE="${PHYSFLOW_BATCH_SIZE:-16384}"
export PHYSFLOW_TRAINING_MAX_STEPS="${PHYSFLOW_TRAINING_MAX_STEPS:-30000}"
export PHYSFLOW_SAVE_EVERY="${PHYSFLOW_SAVE_EVERY:-50}"
export PHYSFLOW_EVAL_EVERY="${PHYSFLOW_EVAL_EVERY:-25}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

cd "${PROJECT_ROOT}"
bash scripts/embodied/launch_position_aware_g1_tracker_train.sh
