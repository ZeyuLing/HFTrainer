#!/usr/bin/env bash
# Guarded adversarial tracker training entrypoint.
#
# This is the replacement for the small unweighted rehearsal-v2 style runs:
# generated/adversarial motions are admitted through an explicit weighted
# manifest, while native/rehearsal motions keep a fixed replay mass to prevent
# regression against the released G1 tracker distribution.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/guarded_adversarial_tracker/$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-physflow_g1_guarded_adv_amass_replay_v1}"
NUM_ENVS="${NUM_ENVS:-256}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
TRAINING_MAX_STEPS="${TRAINING_MAX_STEPS:-500000}"
NGPU="${NGPU:-8}"
EVAL_METRICS_EVERY="${EVAL_METRICS_EVERY:-1000000}"
SKIP_INITIAL_EVAL="${SKIP_INITIAL_EVAL:-1}"
NATIVE_WEIGHT="${NATIVE_WEIGHT:-0.90}"
ADVERSARIAL_WEIGHT="${ADVERSARIAL_WEIGHT:-0.08}"
JUMP_WEIGHT="${JUMP_WEIGHT:-0.02}"
MAX_PER_GROUP="${MAX_PER_GROUP:-4096}"
MANIFEST_SEED="${MANIFEST_SEED:-0}"
TASK_REWARD_W="${TASK_REWARD_W:-0.5}"
DISCRIMINATOR_REWARD_W="${DISCRIMINATOR_REWARD_W:-2.0}"
ACTOR_LR="${ACTOR_LR:-2e-6}"
CRITIC_LR="${CRITIC_LR:-1e-5}"
DISCRIMINATOR_LR="${DISCRIMINATOR_LR:-1e-5}"
DISC_CRITIC_LR="${DISC_CRITIC_LR:-1e-5}"

# Required motion pools. Each may be a directory of .motion files, a .motion, or a YAML manifest.
DEFAULT_NATIVE_MOTION_DIR="${PROJECT_ROOT}/output/amass_g1_proto_baseline_eval/debug2_20260604_1904_wxyz_4gpu/motion_shards"
if [[ ! -d "${DEFAULT_NATIVE_MOTION_DIR}" ]]; then
    DEFAULT_NATIVE_MOTION_DIR="${PROJECT_ROOT}/ref_repo/ProtoMotions/data/motion_for_trackers/amass_g1_full_smoke8_proto"
fi
NATIVE_MOTION_DIR="${NATIVE_MOTION_DIR:-${DEFAULT_NATIVE_MOTION_DIR}}"
ADVERSARIAL_MOTION_DIR="${ADVERSARIAL_MOTION_DIR:-${PROJECT_ROOT}/output/physflow_kimodo_g1/physflow_g1_released_rehearsal_v1_pool}"
DEFAULT_JUMP_MOTION_DIR="${PROJECT_ROOT}/output/physflow_kimodo_g1/jump_hml3d40_noscene_mn1500_reset_env256_40m_noproj/physflow_jump_hml3d40_noscene_mn1500_reset_env256_40m_noproj_run/proto"
if [[ ! -d "${DEFAULT_JUMP_MOTION_DIR}" ]]; then
    DEFAULT_JUMP_MOTION_DIR="${PROJECT_ROOT}/output/physflow_kimodo_g1/physflow_g1_released_rehearsal_v1_pool"
fi
JUMP_MOTION_DIR="${JUMP_MOTION_DIR:-${DEFAULT_JUMP_MOTION_DIR}}"

# This warmstart is derived from the released g1-bones checkpoint but adapted to
# the trainable rehearsal experiment config. Evaluation must still compare
# against the official g1-bones-deploy checkpoint.
WARMSTART_CKPT="${WARMSTART_CKPT:-${PROJECT_ROOT}/output/physflow_kimodo_g1/checkpoints/g1_released_warmstart_epoch0.ckpt}"
EXPERIMENT_PATH="${EXPERIMENT_PATH:-examples/experiments/mimic/physflow_g1_released_rehearsal.py}"
PACK_MOTION_LIB="${PACK_MOTION_LIB:-0}"
PACKED_MOTION_LIB="${PACKED_MOTION_LIB:-${OUT_ROOT}/weighted_motion_manifest.pt}"
REBUILD_GYMTORCH="${REBUILD_GYMTORCH:-0}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[guarded-adv] start $(date)"
echo "[guarded-adv] project=${PROJECT_ROOT}"
echo "[guarded-adv] out=${OUT_ROOT}"
echo "[guarded-adv] experiment=${EXPERIMENT_NAME}"
echo "[guarded-adv] ngpu=${NGPU} num_envs=${NUM_ENVS} batch_size=${BATCH_SIZE} max_steps=${TRAINING_MAX_STEPS}"
echo "[guarded-adv] eval_metrics_every=${EVAL_METRICS_EVERY} skip_initial_eval=${SKIP_INITIAL_EVAL}"
echo "[guarded-adv] weights native=${NATIVE_WEIGHT} adversarial=${ADVERSARIAL_WEIGHT} jump=${JUMP_WEIGHT}"
echo "[guarded-adv] pack_motion_lib=${PACK_MOTION_LIB} rebuild_gymtorch=${REBUILD_GYMTORCH}"
echo "[guarded-adv] pools native=${NATIVE_MOTION_DIR} adversarial=${ADVERSARIAL_MOTION_DIR} jump=${JUMP_MOTION_DIR}"

if [[ "${RUN_NODE_SETUP:-1}" == "1" ]]; then
    PHYSFLOW_NODE_SETUP_ONLY=1 bash scripts/embodied/cursor_physflow_taiji_node_setup.sh
fi

if [[ -n "${PHYSFLOW_TRACKER_PYTHON_CMD:-}" ]]; then
    read -r -a TRACKER_PY <<< "${PHYSFLOW_TRACKER_PYTHON_CMD}"
elif [[ -x /root/physflow_isaacgym_py38_cu118/bin/python ]]; then
    TRACKER_PY=(/root/physflow_isaacgym_py38_cu118/bin/python)
else
    TRACKER_PY=(python3)
fi

MANIFEST="${MANIFEST:-${OUT_ROOT}/weighted_motion_manifest.yaml}"
if [[ -s "${MANIFEST}" ]]; then
    echo "[guarded-adv] using existing manifest=${MANIFEST}"
else
    python3 scripts/embodied/build_weighted_motion_manifest.py \
        --group "native::${NATIVE_MOTION_DIR}::${NATIVE_WEIGHT}" \
        --group "adversarial::${ADVERSARIAL_MOTION_DIR}::${ADVERSARIAL_WEIGHT}" \
        --group "jump::${JUMP_MOTION_DIR}::${JUMP_WEIGHT}" \
        --max-per-group "${MAX_PER_GROUP}" \
        --seed "${MANIFEST_SEED}" \
        --output "${MANIFEST}"
fi

cd "${PROJECT_ROOT}/ref_repo/ProtoMotions"
export PYTHONPATH="${PWD}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/.cache/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-8}"

for version in 14 13 12 11 10 9; do
    gcc_root="/opt/rh/gcc-toolset-${version}/root/usr"
    if [[ -d "${gcc_root}/bin" ]]; then
        export PATH="${gcc_root}/bin:${PATH}"
        export CC="${gcc_root}/bin/gcc"
        export CXX="${gcc_root}/bin/g++"
        export LD_LIBRARY_PATH="${gcc_root}/lib64:${LD_LIBRARY_PATH:-}"
        echo "[guarded-adv] using gcc-toolset-${version}: CC=${CC}"
        break
    fi
done

echo "[guarded-adv] tracker_python=${TRACKER_PY[*]}"
"${TRACKER_PY[@]}" - <<'PY'
import importlib.util, sys
print("python", sys.version)
for name in ("torch", "isaacgym", "mujoco", "lightning", "tensordict"):
    print(f"import_check {name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
PY

echo "[guarded-adv] warm IsaacGym gymtorch"
if [[ "${REBUILD_GYMTORCH}" == "1" ]]; then
    rm -rf "${TORCH_EXTENSIONS_DIR}/gymtorch"
fi
"${TRACKER_PY[@]}" - <<'PY'
import isaacgym  # noqa: F401
from isaacgym import gymtorch  # noqa: F401
print("gymtorch warmup OK")
PY

if [[ "${PACK_MOTION_LIB}" == "1" ]]; then
    if [[ -s "${PACKED_MOTION_LIB}" ]]; then
        echo "[guarded-adv] using existing packed motion lib=${PACKED_MOTION_LIB}"
    else
        echo "[guarded-adv] packing motion lib -> ${PACKED_MOTION_LIB}"
        "${TRACKER_PY[@]}" protomotions/components/motion_lib.py \
            --motion-path "${MANIFEST}" \
            --output-file "${PACKED_MOTION_LIB}" \
            --device cpu
    fi
    MANIFEST="${PACKED_MOTION_LIB}"
fi

overrides=(
    "agent.evaluator.eval_metrics_every=${EVAL_METRICS_EVERY}"
    "agent.save_last_checkpoint_every=5"
    "agent.save_epoch_checkpoint_every=50"
    "agent.task_reward_w=${TASK_REWARD_W}"
    "agent.amp_parameters.discriminator_reward_w=${DISCRIMINATOR_REWARD_W}"
    "agent.model.actor_optimizer.lr=${ACTOR_LR}"
    "agent.model.critic_optimizer.lr=${CRITIC_LR}"
    "agent.model.discriminator_optimizer.lr=${DISCRIMINATOR_LR}"
    "agent.model.disc_critic_optimizer.lr=${DISC_CRITIC_LR}"
)
if [[ -n "${PHYSFLOW_EXTRA_OVERRIDES:-}" ]]; then
    read -r -a extra_overrides <<< "${PHYSFLOW_EXTRA_OVERRIDES}"
    overrides+=("${extra_overrides[@]}")
fi

skip_initial_eval_args=()
if [[ "${SKIP_INITIAL_EVAL}" == "1" ]]; then
    skip_initial_eval_args=(--skip-initial-eval)
fi

"${TRACKER_PY[@]}" protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaacgym \
    --experiment-path "${EXPERIMENT_PATH}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --motion-file "${MANIFEST}" \
    --checkpoint "${WARMSTART_CKPT}" \
    --num-envs "${NUM_ENVS}" \
    --batch-size "${BATCH_SIZE}" \
    --training-max-steps "${TRAINING_MAX_STEPS}" \
    --ngpu "${NGPU}" \
    --headless True \
    "${skip_initial_eval_args[@]}" \
    --overrides "${overrides[@]}"

RESULT_DIR="${PROJECT_ROOT}/ref_repo/ProtoMotions/results/${EXPERIMENT_NAME}"
SNAPSHOT_DIR="${OUT_ROOT}/checkpoints"
mkdir -p "${SNAPSHOT_DIR}"
if [[ -s "${RESULT_DIR}/last.ckpt" ]]; then
    cp -f "${RESULT_DIR}/last.ckpt" "${SNAPSHOT_DIR}/last.ckpt"
fi
for meta in \
    config.yaml \
    experiment_config.py \
    resolved_configs.pt \
    resolved_configs.yaml \
    resolved_configs_inference.pt \
    resolved_configs_inference.yaml
do
    if [[ -e "${RESULT_DIR}/${meta}" ]]; then
        cp -f "${RESULT_DIR}/${meta}" "${SNAPSHOT_DIR}/${meta}"
    fi
done

echo "[guarded-adv] done $(date)"
echo "[guarded-adv] manifest=${MANIFEST}"
echo "[guarded-adv] checkpoint=${PROJECT_ROOT}/ref_repo/ProtoMotions/results/${EXPERIMENT_NAME}/last.ckpt"
echo "[guarded-adv] checkpoint_snapshot=${SNAPSHOT_DIR}/last.ckpt"
