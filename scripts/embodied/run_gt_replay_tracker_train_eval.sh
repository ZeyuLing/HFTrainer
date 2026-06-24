#!/usr/bin/env bash
# GT-only replay training sanity check for the G1 tracker.
#
# Goal: isolate tracker learning from generator quality.  The script fine-tunes
# the released G1 tracker warmstart on retargeted GT motions, then evaluates the
# official pretrained tracker and the fine-tuned checkpoint on the same AMASS-G1
# benchmark shards.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
PROTO_ROOT="${PROTO_ROOT:-${PROJECT_ROOT}/hftrainer/models/motion/physflow/trackers/protomotions/vendor}"
ENVDIR="${ENVDIR:-/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env}"
RUN_TAG="${RUN_TAG:-amass_sanity}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/gt_replay_tracker_train_eval/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

AMASS_ROOT="${AMASS_ROOT:-${PROJECT_ROOT}/data/AMASS_Retarged_for_G1/g1}"
CACHED_MOTION_DIR="${CACHED_MOTION_DIR:-${PROJECT_ROOT}/output/amass_g1_proto_baseline_eval/debug2_20260604_1904_wxyz_4gpu/motion_shards}"
TRAIN_MOTION_DIR="${TRAIN_MOTION_DIR:-}"
TRAIN_CONVERT_NUM_RANK="${TRAIN_CONVERT_NUM_RANK:-8}"
TRAIN_CONVERT_RANK="${TRAIN_CONVERT_RANK:-0}"
OUTPUT_FPS="${OUTPUT_FPS:-30}"
QUAT_ORDER="${QUAT_ORDER:-wxyz}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-physflow_g1_gt_replay_${RUN_TAG}}"
EXPERIMENT_PATH="${EXPERIMENT_PATH:-examples/experiments/mimic/physflow_g1_released_rehearsal.py}"
OFFICIAL_CKPT="${OFFICIAL_CKPT:-${PROTO_ROOT}/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt}"
WARMSTART_CKPT="${WARMSTART_CKPT:-${PROJECT_ROOT}/output/physflow_kimodo_g1/checkpoints/g1_released_warmstart_epoch0.ckpt}"

TRAINING_MAX_STEPS="${TRAINING_MAX_STEPS:-100000}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-32}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NGPU="${NGPU:-1}"
SAVE_EVERY="${SAVE_EVERY:-5}"
EVAL_METRICS_EVERY="${EVAL_METRICS_EVERY:-1000000}"
MAX_PER_GROUP="${MAX_PER_GROUP:-4096}"
MANIFEST_SEED="${MANIFEST_SEED:-20260615}"

TASK_REWARD_W="${TASK_REWARD_W:-0.5}"
DISCRIMINATOR_REWARD_W="${DISCRIMINATOR_REWARD_W:-2.0}"
ACTOR_LR="${ACTOR_LR:-2e-6}"
CRITIC_LR="${CRITIC_LR:-1e-5}"
DISCRIMINATOR_LR="${DISCRIMINATOR_LR:-1e-5}"
DISC_CRITIC_LR="${DISC_CRITIC_LR:-1e-5}"

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-1}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-256}"
EVAL_MAX_EVAL_STEPS="${EVAL_MAX_EVAL_STEPS:-600}"
EVAL_FORCE="${EVAL_FORCE:-1}"
SAVE_PREDICTED_MOTION_LIB="${SAVE_PREDICTED_MOTION_LIB:-0}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[gt-replay] start $(date)"
echo "[gt-replay] host=$(hostname)"
echo "[gt-replay] out=${OUT_ROOT}"
echo "[gt-replay] experiment=${EXPERIMENT_NAME}"
echo "[gt-replay] amass=${AMASS_ROOT}"
echo "[gt-replay] warmstart=${WARMSTART_CKPT}"
echo "[gt-replay] official_before=${OFFICIAL_CKPT}"
echo "[gt-replay] ngpu=${NGPU} envs=${NUM_ENVS} batch=${BATCH_SIZE} steps=${TRAINING_MAX_STEPS} max_per_group=${MAX_PER_GROUP}"

nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[gt-replay] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ -n "${CUDA_DRV:-}" ] && awk "BEGIN{exit !(${CUDA_DRV} < 11.4)}"; then
  echo "[gt-replay] FATAL_BAD_NODE: CUDA driver ${CUDA_DRV} < 11.4. Aborting fast for reschedule."
  exit 42
fi
echo "[gt-replay] driver gate OK (>=11.4)"

if [[ "${RUN_NODE_SETUP:-1}" == "1" ]]; then
  PHYSFLOW_NODE_SETUP_ONLY=1 bash scripts/embodied/cursor_physflow_taiji_node_setup.sh || true
fi

ln -sfn "${ENVDIR}/isaacgym" /root/isaacgym
ln -sfn "${ENVDIR}/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -3 || echo "[gt-replay] WARN dnf python38-devel install failed"
PY38RT="${ENVDIR}/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[gt-replay] restoring base python3.8 from ${PY38RT}"
  cp -a "${PY38RT}/bin/python3.8" /usr/bin/python3.8
  cp -a "${PY38RT}/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "${PY38RT}/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
if [[ ! -f /usr/include/python3.8/Python.h && -d "${PY38RT}/include/python3.8" ]]; then
  echo "[gt-replay] restoring python3.8 headers from ${PY38RT}/include/python3.8"
  mkdir -p /usr/include
  rsync -a "${PY38RT}/include/python3.8" /usr/include/
fi
if [[ ! -f /usr/include/python3.8/Python.h ]]; then
  echo "[gt-replay] ERROR: missing /usr/include/python3.8/Python.h; gymtorch cannot build" >&2
  exit 43
fi

if [[ -n "${PHYSFLOW_TRACKER_PYTHON_CMD:-}" ]]; then
  read -r -a TRACKER_PY <<< "${PHYSFLOW_TRACKER_PYTHON_CMD}"
elif [[ -x /root/physflow_isaacgym_py38_cu118/bin/python ]]; then
  TRACKER_PY=(/root/physflow_isaacgym_py38_cu118/bin/python)
else
  TRACKER_PY=(python3)
fi

echo "[gt-replay] tracker_python=${TRACKER_PY[*]}"
"${TRACKER_PY[@]}" - <<'PY'
import importlib.util, sys
print("python", sys.version)
for name in ("torch", "isaacgym", "mujoco", "lightning", "tensordict"):
    print(f"import_check {name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
PY

if [[ -z "${TRAIN_MOTION_DIR}" ]]; then
  if [[ -d "${CACHED_MOTION_DIR}" ]] && find -L "${CACHED_MOTION_DIR}" -type f -name '*.motion' -print -quit | grep -q .; then
    TRAIN_MOTION_DIR="${CACHED_MOTION_DIR}"
    echo "[gt-replay] using cached GT motion dir=${TRAIN_MOTION_DIR}"
  else
    TRAIN_MOTION_DIR="${OUT_ROOT}/train_motion_shard_${TRAIN_CONVERT_RANK}_of_${TRAIN_CONVERT_NUM_RANK}"
  fi
fi

if ! find -L "${TRAIN_MOTION_DIR}" -type f -name '*.motion' -print -quit 2>/dev/null | grep -q .; then
  if [[ ! -d "${AMASS_ROOT}" ]]; then
    echo "[gt-replay] ERROR: missing AMASS root: ${AMASS_ROOT}" >&2
    exit 2
  fi
  echo "[gt-replay] converting AMASS-G1 -> ProtoMotions motions: ${TRAIN_MOTION_DIR}"
  mkdir -p "${TRAIN_MOTION_DIR}"
  (
    cd "${PROTO_ROOT}"
    export PYTHONPATH="${PWD}:${PROJECT_ROOT}:${PYTHONPATH:-}"
    "${TRACKER_PY[@]}" data/scripts/convert_amass_g1_npz_to_proto.py \
      --input-dir "${AMASS_ROOT}" \
      --output-dir "${TRAIN_MOTION_DIR}" \
      --output-fps "${OUTPUT_FPS}" \
      --quat-order "${QUAT_ORDER}" \
      --num-rank "${TRAIN_CONVERT_NUM_RANK}" \
      --slurm-rank "${TRAIN_CONVERT_RANK}"
  )
fi

motion_count=$(find -L "${TRAIN_MOTION_DIR}" -type f -name '*.motion' | wc -l)
echo "[gt-replay] train_motion_dir=${TRAIN_MOTION_DIR} motion_count=${motion_count}"
if [[ "${motion_count}" -lt 1 ]]; then
  echo "[gt-replay] ERROR: no GT .motion files available for training" >&2
  exit 3
fi

MANIFEST="${MANIFEST:-${OUT_ROOT}/gt_replay_manifest.yaml}"
python3 scripts/embodied/build_weighted_motion_manifest.py \
  --group "gt::${TRAIN_MOTION_DIR}::1.0" \
  --max-per-group "${MAX_PER_GROUP}" \
  --seed "${MANIFEST_SEED}" \
  --output "${MANIFEST}"

cd "${PROTO_ROOT}"
export PYTHONPATH="${PWD}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/.cache/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-8}"
if [[ "${REBUILD_GYMTORCH:-1}" == "1" ]]; then
  rm -rf "${TORCH_EXTENSIONS_DIR}/gymtorch"
fi

for version in 14 13 12 11 10 9; do
  gcc_root="/opt/rh/gcc-toolset-${version}/root/usr"
  if [[ -d "${gcc_root}/bin" ]]; then
    export PATH="${gcc_root}/bin:${PATH}"
    export CC="${gcc_root}/bin/gcc"
    export CXX="${gcc_root}/bin/g++"
    export LD_LIBRARY_PATH="${gcc_root}/lib64:${LD_LIBRARY_PATH:-}"
    echo "[gt-replay] using gcc-toolset-${version}: CC=${CC}"
    break
  fi
done

echo "[gt-replay] warm IsaacGym gymtorch"
"${TRACKER_PY[@]}" - <<'PY'
import isaacgym  # noqa: F401
from isaacgym import gymtorch  # noqa: F401
print("gymtorch warmup OK")
PY

"${TRACKER_PY[@]}" - "${WARMSTART_CKPT}" "${TRAINING_MAX_STEPS}" <<'PY'
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
    "[gt-replay] checkpoint_meta "
    f"epoch={epoch} step_count={step_count} "
    f"skip_optimizer_load={state.get('skip_optimizer_load')} "
    f"best_evaluated_score={state.get('best_evaluated_score')}"
)
if os.environ.get("ALLOW_NONZERO_WARMSTART") != "1" and (epoch != 0 or step_count != 0):
    raise SystemExit("[gt-replay] ERROR: warmstart checkpoint must be epoch=0 step_count=0")
if step_count >= training_max_steps:
    raise SystemExit("[gt-replay] ERROR: checkpoint step_count already >= training_max_steps")
PY

overrides=(
  "agent.evaluator.eval_metrics_every=${EVAL_METRICS_EVERY}"
  "agent.save_last_checkpoint_every=${SAVE_EVERY}"
  "agent.save_epoch_checkpoint_every=50"
  "agent.num_steps=${NUM_STEPS}"
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
  --nodes 1 \
  --skip-initial-eval \
  --headless True \
  --overrides "${overrides[@]}"

RESULT_DIR="${PROTO_ROOT}/results/${EXPERIMENT_NAME}"
TRAINED_CKPT="${RESULT_DIR}/last.ckpt"
SNAPSHOT_DIR="${OUT_ROOT}/checkpoints"
mkdir -p "${SNAPSHOT_DIR}"
if [[ ! -s "${TRAINED_CKPT}" ]]; then
  echo "[gt-replay] ERROR: trained checkpoint missing: ${TRAINED_CKPT}" >&2
  exit 4
fi
cp -f "${TRAINED_CKPT}" "${SNAPSHOT_DIR}/last.ckpt"
for meta in config.yaml experiment_config.py resolved_configs.pt resolved_configs.yaml resolved_configs_inference.pt resolved_configs_inference.yaml; do
  if [[ -e "${RESULT_DIR}/${meta}" ]]; then
    cp -f "${RESULT_DIR}/${meta}" "${SNAPSHOT_DIR}/${meta}"
  fi
done
echo "[gt-replay] trained_checkpoint=${TRAINED_CKPT}"
echo "[gt-replay] checkpoint_snapshot=${SNAPSHOT_DIR}/last.ckpt"

if [[ "${RUN_EVAL}" == "1" ]]; then
  cd "${PROJECT_ROOT}"
  EVAL_OUT="${OUT_ROOT}/eval_before_after"
  echo "[gt-replay] running before/after eval -> ${EVAL_OUT}"
  RUN_NODE_SETUP=0 \
  OUT_ROOT="${EVAL_OUT}" \
  NUM_SHARDS="${EVAL_NUM_SHARDS}" \
  NUM_ENVS="${EVAL_NUM_ENVS}" \
  MAX_EVAL_STEPS="${EVAL_MAX_EVAL_STEPS}" \
  FORCE_EVAL="${EVAL_FORCE}" \
  SAVE_PREDICTED_MOTION_LIB="${SAVE_PREDICTED_MOTION_LIB}" \
  CHECKPOINT_SPECS="pretrained_g1_bones=${OFFICIAL_CKPT},gt_replay_after=${TRAINED_CKPT}" \
  bash scripts/embodied/run_amass_g1_proto_baseline_eval.sh
fi

echo "[gt-replay] done $(date)"
