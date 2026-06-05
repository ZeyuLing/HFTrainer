#!/usr/bin/env bash
# Build a jump-focused PhysFlow-G1 motion pool from HumanML3D prompts, then run
# an 8-GPU small-data tracker overfit experiment on that pool.
set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${PROJECT_ROOT}"

TAG="${PHYSFLOW_JUMP_TAG:-jump_hml3d40_noscene_mn1500_reset_env256_40m_noproj}"
OUT="${PHYSFLOW_JUMP_OUT:-output/physflow_kimodo_g1/${TAG}}"
PROMPT_BANK="${PHYSFLOW_JUMP_PROMPT_BANK:-configs/experiments/physflow_kimodo_g1/physflow_jump_overfit_prompts_hml3d_g1_noscene.jsonl}"
CONFIG="${PHYSFLOW_CONFIG:-configs/physflow/physflow_online_adv_mn.py}"
CKPT="${PHYSFLOW_CKPT:-work_dirs/physflow_online_adv_mn/checkpoint-iter_1500}"
GEN_PY="${PHYSFLOW_PYTHON_CMD:-/usr/local/bin/python3}"
GEN_GPU="${PHYSFLOW_GEN_GPU:-0}"
TEXT_NS="${PHYSFLOW_JUMP_TEXT_NS:-kimodo_g1_llm2vec_${TAG}}"
FEATURE_DIR="${PHYSFLOW_JUMP_FEATURE_DIR:-data/kimodo_text_feature/${TEXT_NS}}"
GEN_RUN="${OUT}/physflow_${TAG}_run"
GEN_MANIFEST="${OUT}/physflow_${TAG}_manifest"
MOTION_DIR="${PHYSFLOW_JUMP_MOTION_DIR:-${GEN_RUN}/proto}"
if [[ "${MOTION_DIR}" = /* ]]; then
    MOTION_FILE="${MOTION_DIR}"
else
    MOTION_FILE="${PROJECT_ROOT}/${MOTION_DIR}"
fi

EXPERIMENT_NAME="${PHYSFLOW_EXPERIMENT_NAME:-physflow_g1_jump_overfit_${TAG}}"
EXPERIMENT_PATH="${PHYSFLOW_EXPERIMENT_PATH:-examples/experiments/mimic/physflow_g1_jump_overfit.py}"
TRAINING_MAX_STEPS="${PHYSFLOW_TRAINING_MAX_STEPS:-40000000}"
NUM_ENVS="${PHYSFLOW_NUM_ENVS:-256}"
NUM_STEPS="${PHYSFLOW_NUM_STEPS:-32}"
BATCH_SIZE="${PHYSFLOW_BATCH_SIZE:-4096}"
NGPU="${PHYSFLOW_NGPU:-8}"
SAVE_EVERY="${PHYSFLOW_SAVE_EVERY:-5}"
EVAL_EVERY="${PHYSFLOW_EVAL_EVERY:-25}"
CHECKPOINT="${PHYSFLOW_CHECKPOINT:-${PROJECT_ROOT}/output/physflow_kimodo_g1/checkpoints/g1_released_warmstart_epoch0.ckpt}"

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/launch.log") 2>&1

echo "[jump-overfit] start $(date)"
echo "[jump-overfit] host=$(hostname) out=${OUT}"
echo "[jump-overfit] prompt_bank=${PROMPT_BANK}"
echo "[jump-overfit] gen_ckpt=${CKPT}"
echo "[jump-overfit] tracker_checkpoint=${CHECKPOINT}"
MAX_EPOCHS=$(( TRAINING_MAX_STEPS / NGPU / NUM_ENVS / NUM_STEPS ))
echo "[jump-overfit] experiment=${EXPERIMENT_NAME} ngpu=${NGPU} envs_per_rank=${NUM_ENVS} rollout_steps=${NUM_STEPS} max_steps=${TRAINING_MAX_STEPS} max_epochs=${MAX_EPOCHS}"
if [[ "${MAX_EPOCHS}" -lt 1 ]]; then
    echo "[jump-overfit] ERROR: training budget gives 0 epochs; increase PHYSFLOW_TRAINING_MAX_STEPS or reduce PHYSFLOW_NUM_ENVS/PHYSFLOW_NUM_STEPS."
    exit 4
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/[jump-overfit] gpu /' || true

export HF_HOME="${PROJECT_ROOT}/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="${PROJECT_ROOT}/checkpoints/kimodo/text_encoders"
export PHYSFLOW_CONVERT_PYTHON="${GEN_PY}"

"${GEN_PY}" -c "import mujoco, onnxruntime, dm_control, typer" 2>/dev/null || {
    echo "[jump-overfit] installing generation/scoring deps ..."
    "${GEN_PY}" -m pip install --quiet mujoco onnxruntime dm_control typer
}

CUDA_VISIBLE_DEVICES="${GEN_GPU}" "${GEN_PY}" scripts/embodied/cursor_extract_kimodo_text_feature.py \
    --corpus "${PROMPT_BANK}" --namespace "${TEXT_NS}" --text-encoder llm2vec --device cuda --batch-size 16

CUDA_VISIBLE_DEVICES="${GEN_GPU}" "${GEN_PY}" scripts/embodied/physflow_coevolve_viz.py \
    --config "${CONFIG}" --ckpt "${CKPT}" \
    --eval-corpus "${PROMPT_BANK}" --feature-dir "${FEATURE_DIR}" --split train \
    --num-prompts 40 --gen-batch 8 \
    --out-dir "${GEN_RUN}" \
    --manifest-dir "${GEN_MANIFEST}" --iteration 1500

motion_count=$(find "${MOTION_FILE}" -type f -name '*.motion' | wc -l)
echo "[jump-overfit] motion_dir=${MOTION_FILE} motion_count=${motion_count}"
if [[ "${motion_count}" -lt 1 ]]; then
    echo "[jump-overfit] ERROR: no .motion files generated"
    exit 2
fi

cd "${PROJECT_ROOT}/ref_repo/ProtoMotions"
export PYTHONPATH="${PWD}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_DISABLE_SENTRY="${WANDB_DISABLE_SENTRY:-true}"

if [[ "${PHYSFLOW_USE_GCC_TOOLSET:-1}" == "1" ]]; then
    for toolset in /opt/rh/gcc-toolset-14/enable /opt/rh/gcc-toolset-13/enable /opt/rh/gcc-toolset-12/enable /opt/rh/gcc-toolset-11/enable /opt/rh/gcc-toolset-10/enable /opt/rh/gcc-toolset-9/enable; do
        if [[ -f "${toolset}" ]]; then
            # shellcheck disable=SC1090
            source "${toolset}"
            echo "[jump-overfit] enabled compiler toolset: ${toolset}"
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

echo "[jump-overfit] tracker_python=${TRACKER_PY[*]}"
"${TRACKER_PY[@]}" - <<'PY'
import importlib.util
import sys

print("python", sys.version)
for name in ("torch", "isaacgym", "mujoco", "lightning", "tensordict"):
    print(f"import_check {name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
PY

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/.cache/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-8}"
rm -rf "${TORCH_EXTENSIONS_DIR}/py38_cu118/gymtorch"
echo "[jump-overfit] warming up IsaacGym gymtorch JIT cache at ${TORCH_EXTENSIONS_DIR}"
"${TRACKER_PY[@]}" - <<'PY'
import isaacgym  # noqa: F401
from isaacgym import gymtorch  # noqa: F401

print("gymtorch warmup OK")
PY

RESULT_DIR="${PROJECT_ROOT}/ref_repo/ProtoMotions/results/${EXPERIMENT_NAME}"
if [[ -f "${RESULT_DIR}/last.ckpt" && "${PHYSFLOW_ALLOW_EXISTING_RESULT:-0}" != "1" ]]; then
    echo "[jump-overfit] ERROR: result checkpoint already exists: ${RESULT_DIR}/last.ckpt"
    echo "[jump-overfit] Set PHYSFLOW_ALLOW_EXISTING_RESULT=1 only when intentionally resuming."
    exit 3
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
skip_optimizer_load = state.get("skip_optimizer_load")
print(
    "[jump-overfit] checkpoint_meta "
    f"epoch={epoch} step_count={step_count} "
    f"skip_optimizer_load={skip_optimizer_load} "
    f"best_evaluated_score={state.get('best_evaluated_score')}"
)

if os.environ.get("PHYSFLOW_ALLOW_NONZERO_CHECKPOINT") != "1":
    if epoch != 0 or step_count != 0:
        raise SystemExit(
            "[jump-overfit] ERROR: overfit warm-start checkpoint must have "
            "epoch=0 and step_count=0. Refusing to launch a fake-complete run. "
            "Set PHYSFLOW_ALLOW_NONZERO_CHECKPOINT=1 only for intentional resume."
        )
if step_count >= training_max_steps:
    raise SystemExit(
        "[jump-overfit] ERROR: checkpoint step_count is already >= "
        f"training_max_steps ({step_count} >= {training_max_steps})."
    )
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

echo "[jump-overfit] done $(date)"
echo "[jump-overfit] result=${PROJECT_ROOT}/ref_repo/ProtoMotions/results/${EXPERIMENT_NAME}"
