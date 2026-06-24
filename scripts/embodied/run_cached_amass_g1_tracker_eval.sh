#!/usr/bin/env bash
# Evaluate tracker checkpoints on an existing packed AMASS-G1 ProtoMotions shard.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
ENVDIR="${ENVDIR:-/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env}"
CACHED_MOTION_BASE="${CACHED_MOTION_BASE:-${PROJECT_ROOT}/output/amass_g1_proto_baseline_eval/debug2_20260604_1904_wxyz_4gpu/motion_shards}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/gt_replay_tracker_train_eval/cached_eval_$(date +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${NUM_SHARDS:-1}"
NUM_ENVS="${NUM_ENVS:-256}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-600}"
GPU_OFFSET="${GPU_OFFSET:-0}"
SAVE_PREDICTED_MOTION_LIB_EVERY="${SAVE_PREDICTED_MOTION_LIB_EVERY:-None}"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:?set CHECKPOINT_SPECS as name=/path/to.ckpt[,name2=/path2]}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[cached-amass-eval] start $(date)"
echo "[cached-amass-eval] host=$(hostname)"
echo "[cached-amass-eval] out=${OUT_ROOT}"
echo "[cached-amass-eval] motion_base=${CACHED_MOTION_BASE}"
echo "[cached-amass-eval] num_shards=${NUM_SHARDS} num_envs=${NUM_ENVS} max_eval_steps=${MAX_EVAL_STEPS}"
echo "[cached-amass-eval] save_predicted_motion_lib_every=${SAVE_PREDICTED_MOTION_LIB_EVERY}"
echo "[cached-amass-eval] checkpoints=${CHECKPOINT_SPECS}"

ln -sfn "${ENVDIR}/isaacgym" /root/isaacgym
ln -sfn "${ENVDIR}/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -3 \
  || echo "[cached-amass-eval] WARN dnf python38-devel install failed"
PY38RT="${ENVDIR}/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[cached-amass-eval] restoring base python3.8 from ${PY38RT}"
  cp -a "${PY38RT}/bin/python3.8" /usr/bin/python3.8
  cp -a "${PY38RT}/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "${PY38RT}/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
if [[ ! -f /usr/include/python3.8/Python.h && -d "${PY38RT}/include/python3.8" ]]; then
  echo "[cached-amass-eval] restoring python3.8 headers from ${PY38RT}/include/python3.8"
  mkdir -p /usr/include
  rsync -a "${PY38RT}/include/python3.8" /usr/include/
fi
if [[ ! -f /usr/include/python3.8/Python.h ]]; then
  echo "[cached-amass-eval] ERROR: missing /usr/include/python3.8/Python.h; gymtorch cannot build" >&2
  exit 43
fi

if [[ -n "${PHYSFLOW_TRACKER_PYTHON_CMD:-}" ]]; then
  read -r -a TRACKER_PY <<< "${PHYSFLOW_TRACKER_PYTHON_CMD}"
elif [[ -x /root/physflow_isaacgym_py38_cu118/bin/python ]]; then
  TRACKER_PY=(/root/physflow_isaacgym_py38_cu118/bin/python)
else
  TRACKER_PY=(python3)
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
    echo "[cached-amass-eval] using gcc-toolset-${version}: CC=${CC}"
    break
  fi
done

"${TRACKER_PY[@]}" - <<'PY'
import importlib.util, sys
print("python", sys.version)
for name in ("torch", "isaacgym", "mujoco", "lightning", "tensordict"):
    print(f"import_check {name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
import isaacgym  # noqa: F401
from isaacgym import gymtorch  # noqa: F401
print("gymtorch OK")
PY

IFS=',' read -r -a CKPT_ARRAY <<< "${CHECKPOINT_SPECS}"
for spec in "${CKPT_ARRAY[@]}"; do
  name="${spec%%=*}"
  ckpt="${spec#*=}"
  if [[ "${ckpt}" != /* ]]; then
    ckpt="${PROJECT_ROOT}/${ckpt#./}"
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "[cached-amass-eval] ERROR: checkpoint missing for ${name}: ${ckpt}" >&2
    exit 4
  fi

  eval_dir="${OUT_ROOT}/eval_${name}"
  mkdir -p "${eval_dir}"
  echo "[cached-amass-eval] evaluating ${name}: ${ckpt}"
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_pt="${CACHED_MOTION_BASE}/amass_g1_full_shard_${shard}.pt"
    if [[ ! -s "${shard_pt}" ]]; then
      echo "[cached-amass-eval] ERROR: missing shard ${shard_pt}" >&2
      exit 5
    fi
    (
      export CUDA_VISIBLE_DEVICES="$((GPU_OFFSET + shard))"
      "${TRACKER_PY[@]}" protomotions/inference_agent.py \
        --checkpoint "${ckpt}" \
        --motion-file "${shard_pt}" \
        --simulator isaacgym \
        --num-envs "${NUM_ENVS}" \
        --headless \
        --full-eval \
        --root-dir "${eval_dir}" \
        --overrides \
          "agent.evaluator.max_eval_steps=${MAX_EVAL_STEPS}" \
          "agent.evaluator.save_predicted_motion_lib_every=${SAVE_PREDICTED_MOTION_LIB_EVERY}" \
        > "${eval_dir}/shard_${shard}.log" 2>&1
    ) &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[cached-amass-eval] ERROR: eval failed for ${name}" >&2
    exit 6
  fi
done

cd "${PROJECT_ROOT}"
python3 scripts/embodied/aggregate_proto_eval_logs.py \
  --eval-root "${OUT_ROOT}" \
  --motion-base "${CACHED_MOTION_BASE}" \
  --num-shards "${NUM_SHARDS}"

echo "[cached-amass-eval] done $(date)"
