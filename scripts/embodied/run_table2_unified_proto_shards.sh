#!/usr/bin/env bash
# Run ProtoMotions G1 tracker on the unified Table-2 fixed-window protocol.
#
# Launch one copy per Taiji host, e.g. on a 4-host slice:
#   SHARD_START=0  LOCAL_SHARDS=8 TOTAL_SHARDS=32 bash ...
#   SHARD_START=8  LOCAL_SHARDS=8 TOTAL_SHARDS=32 bash ...
#   SHARD_START=16 LOCAL_SHARDS=8 TOTAL_SHARDS=32 bash ...
#   SHARD_START=24 LOCAL_SHARDS=8 TOTAL_SHARDS=32 bash ...
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
PROTO_ROOT="${PROTO_ROOT:-${PROJECT_ROOT}/hftrainer/models/motion/physflow/trackers/protomotions/vendor}"
ENVDIR="${ENVDIR:-/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1}"
SPLITS="${SPLITS:-amass_test_fixed600 lafan1_fixed600 wild_clean_fixed600}"
SPLITS="${SPLITS//,/ }"
TOTAL_SHARDS="${TOTAL_SHARDS:-32}"
LOCAL_SHARDS="${LOCAL_SHARDS:-8}"
SHARD_START="${SHARD_START:-0}"
REFERENCE_FPS="${REFERENCE_FPS:-30}"
TRACKER_CONTROL_FPS="${TRACKER_CONTROL_FPS:-50}"
MAX_REFERENCE_FRAMES="${MAX_REFERENCE_FRAMES:-600}"
OUTPUT_FPS="${OUTPUT_FPS:-${TRACKER_CONTROL_FPS}}"
if [[ -z "${MAX_EVAL_STEPS+x}" ]]; then
  MAX_EVAL_STEPS=$(( (MAX_REFERENCE_FRAMES * TRACKER_CONTROL_FPS + REFERENCE_FPS - 1) / REFERENCE_FPS ))
fi
PROTO_SIMULATOR="${PROTO_SIMULATOR:-isaacgym}"
if [[ -z "${NUM_ENVS+x}" ]]; then
  if [[ "${PROTO_SIMULATOR}" == "mujoco" ]]; then
    NUM_ENVS=1
  else
    NUM_ENVS=128
  fi
fi
MAX_EVAL_JOBS="${MAX_EVAL_JOBS:-1}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
FORCE_PACK="${FORCE_PACK:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
REQUIRE_CUDA_DRIVER_MIN="${REQUIRE_CUDA_DRIVER_MIN:-470}"
REQUIRE_NVIDIA_CUDA_VERSION="${REQUIRE_NVIDIA_CUDA_VERSION:-}"
SAVE_PREDICTED_MOTION_LIB="${SAVE_PREDICTED_MOTION_LIB:-1}"
PREDICTED_MOTION_LIB_EVERY="${PREDICTED_MOTION_LIB_EVERY:-1}"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:-protomotions_g1_bones=data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROTOCOL_ROOT}/logs"
HOST_TAG="$(hostname)_proto_s${SHARD_START}_n${LOCAL_SHARDS}"
exec > >(tee -a "${PROTOCOL_ROOT}/logs/run_${HOST_TAG}.log") 2>&1

echo "[unified-proto] start $(date)"
echo "[unified-proto] host=$(hostname) shard_start=${SHARD_START} local_shards=${LOCAL_SHARDS} total_shards=${TOTAL_SHARDS}"
echo "[unified-proto] splits=${SPLITS}"
echo "[unified-proto] simulator=${PROTO_SIMULATOR} num_envs=${NUM_ENVS} max_eval_steps=${MAX_EVAL_STEPS} output_fps=${OUTPUT_FPS} reference_fps=${REFERENCE_FPS} tracker_control_fps=${TRACKER_CONTROL_FPS} max_reference_frames=${MAX_REFERENCE_FRAMES} max_eval_jobs=${MAX_EVAL_JOBS}"
echo "[unified-proto] checkpoints=${CHECKPOINT_SPECS}"

wait_for_background() {
  local stage="$1"
  shift || true
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      echo "[unified-proto] ERROR: ${stage} subprocess ${pid} failed" >&2
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    exit 6
  fi
}

check_gpu_driver_preflight() {
  if [[ -z "${REQUIRE_CUDA_DRIVER_MIN}" && -z "${REQUIRE_NVIDIA_CUDA_VERSION}" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[unified-proto] ERROR: nvidia-smi missing; cannot run GPU tracker" >&2
    exit 44
  fi
  local smi
  smi="$(nvidia-smi)"
  echo "${smi}" | sed 's/^/[unified-proto] nvidia-smi /'
  SMI_TEXT="${smi}" REQUIRE_CUDA_DRIVER_MIN="${REQUIRE_CUDA_DRIVER_MIN}" REQUIRE_NVIDIA_CUDA_VERSION="${REQUIRE_NVIDIA_CUDA_VERSION}" python3 - <<'PY'
import os, re, sys
smi = os.environ.get("SMI_TEXT", "")
driver_min = os.environ.get("REQUIRE_CUDA_DRIVER_MIN", "")
cuda_required = os.environ.get("REQUIRE_NVIDIA_CUDA_VERSION", "")
driver_match = re.search(r"Driver Version:\s*([0-9]+)(?:\.([0-9]+))?", smi)
cuda_match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", smi)
if driver_min:
    if not driver_match:
        print("[unified-proto] ERROR: cannot parse nvidia driver version", file=sys.stderr)
        raise SystemExit(44)
    major = int(driver_match.group(1))
    if major < int(driver_min):
        print(
            f"[unified-proto] ERROR: nvidia driver {major} < required {driver_min}; "
            "rejecting this host before eval",
            file=sys.stderr,
        )
        raise SystemExit(44)
if cuda_required:
    cuda_seen = cuda_match.group(1) if cuda_match else "unknown"
    if cuda_seen != cuda_required:
        print(
            f"[unified-proto] ERROR: nvidia-smi CUDA Version {cuda_seen} != required {cuda_required}; "
            "rejecting this host before eval",
            file=sys.stderr,
        )
        raise SystemExit(44)
print("[unified-proto] gpu driver preflight OK")
PY
}

setup_tracker_runtime() {
  if [[ "${RUN_NODE_SETUP:-1}" == "1" ]]; then
    PHYSFLOW_NODE_SETUP_ONLY=1 bash scripts/embodied/cursor_physflow_taiji_node_setup.sh
  fi

  local py38rt="${ENVDIR}/py38_runtime"
  if [[ ! -f /usr/include/python3.8/Python.h && -d "${py38rt}/include/python3.8" ]]; then
    echo "[unified-proto] restoring python3.8 headers from ${py38rt}/include/python3.8"
    mkdir -p /usr/include
    rsync -a "${py38rt}/include/python3.8" /usr/include/
  fi
  if [[ ! -f /usr/include/python3.8/Python.h ]]; then
    echo "[unified-proto] ERROR: missing /usr/include/python3.8/Python.h; gymtorch cannot build" >&2
    exit 43
  fi

  if [[ -n "${PHYSFLOW_TRACKER_PYTHON_CMD:-}" ]]; then
    read -r -a TRACKER_PY <<< "${PHYSFLOW_TRACKER_PYTHON_CMD}"
  elif [[ -x /root/physflow_isaacgym_py38_cu118/bin/python ]]; then
    TRACKER_PY=(/root/physflow_isaacgym_py38_cu118/bin/python)
  else
    TRACKER_PY=(python3)
  fi

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

  for version in 14 13 12 11 10 9; do
    local gcc_root="/opt/rh/gcc-toolset-${version}/root/usr"
    if [[ -d "${gcc_root}/bin" ]]; then
      export PATH="${gcc_root}/bin:${PATH}"
      export CC="${gcc_root}/bin/gcc"
      export CXX="${gcc_root}/bin/g++"
      export LD_LIBRARY_PATH="${gcc_root}/lib64:${LD_LIBRARY_PATH:-}"
      echo "[unified-proto] using gcc-toolset-${version}: CC=${CC}"
      break
    fi
  done

  echo "[unified-proto] tracker_python=${TRACKER_PY[*]}"
  "${TRACKER_PY[@]}" - <<'PY'
import importlib.util, sys
print("python", sys.version)
for name in ("torch", "isaacgym", "mujoco", "lightning", "tensordict"):
    print(f"import_check {name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
PY

  if [[ "${PROTO_SIMULATOR}" == "isaacgym" ]]; then
    echo "[unified-proto] warm IsaacGym gymtorch"
    if [[ "${REBUILD_GYMTORCH:-0}" == "1" ]]; then
      rm -rf "${TORCH_EXTENSIONS_DIR}/gymtorch"
    fi
    "${TRACKER_PY[@]}" - <<'PY'
import isaacgym  # noqa: F401
from isaacgym import gymtorch  # noqa: F401
print("gymtorch warmup OK")
PY
  fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/[unified-proto] gpu /' || true
}

convert_and_pack_split() {
  local split="$1"
  local input_dir="${PROTOCOL_ROOT}/inputs/${split}/npz"
  local motion_base="${PROTOCOL_ROOT}/proto_motions/${split}"
  mkdir -p "${motion_base}"
  if [[ ! -d "${input_dir}" ]]; then
    echo "[unified-proto] ERROR missing input split ${input_dir}" >&2
    exit 2
  fi

  echo "[unified-proto] converting split=${split}"
  local pids=()
  for shard in $(seq "${SHARD_START}" $((SHARD_START + LOCAL_SHARDS - 1))); do
    local shard_dir="${motion_base}/shard_${shard}"
    mkdir -p "${shard_dir}"
    if [[ "${FORCE_CONVERT}" == "1" ]]; then
      rm -rf "${shard_dir}"
      mkdir -p "${shard_dir}"
    fi
    if find "${shard_dir}" -type f -name '*.motion' | grep -q .; then
      echo "[unified-proto] ${split} shard ${shard}: conversion exists"
      continue
    fi
    (
      cd "${PROTO_ROOT}"
      "${TRACKER_PY[@]}" data/scripts/convert_g1_qpos_npz_to_proto.py \
        --input-dir "${input_dir}" \
        --output-dir "${shard_dir}" \
        --output-fps "${OUTPUT_FPS}" \
        --num-rank "${TOTAL_SHARDS}" \
        --slurm-rank "${shard}" \
        --manifest "${PROTOCOL_ROOT}/manifests/${split}/shard_${shard}.json" \
        > "${motion_base}/convert_shard_${shard}.log" 2>&1
    ) &
    pids+=("$!")
  done
  wait_for_background "${split} conversion" "${pids[@]}"

  echo "[unified-proto] packing split=${split}"
  for shard in $(seq "${SHARD_START}" $((SHARD_START + LOCAL_SHARDS - 1))); do
    local shard_dir="${motion_base}/shard_${shard}"
    local shard_pt="${motion_base}/${split}_g1_shard_${shard}.pt"
    local motion_count
    motion_count=$(find "${shard_dir}" -type f -name '*.motion' | wc -l)
    echo "[unified-proto] ${split} shard ${shard}: motion_count=${motion_count}"
    if [[ "${motion_count}" == "0" ]]; then
      continue
    fi
    if [[ "${FORCE_PACK}" == "1" || ! -s "${shard_pt}" ]]; then
      cd "${PROTO_ROOT}"
      "${TRACKER_PY[@]}" protomotions/components/motion_lib.py \
        --motion-path "${shard_dir}" \
        --output-file "${shard_pt}" \
        --device cpu \
        > "${motion_base}/pack_shard_${shard}.log" 2>&1
    fi
  done
}

eval_split() {
  local split="$1"
  local motion_base="${PROTOCOL_ROOT}/proto_motions/${split}"
  IFS=',' read -r -a CKPT_ARRAY <<< "${CHECKPOINT_SPECS}"
  for spec in "${CKPT_ARRAY[@]}"; do
    local name="${spec%%=*}"
    local ckpt="${spec#*=}"
    ckpt="${ckpt#./}"
    if [[ "${ckpt}" != /* ]]; then
      ckpt="${PROTO_ROOT}/${ckpt}"
    fi
    if [[ ! -f "${ckpt}" ]]; then
      echo "[unified-proto] ERROR checkpoint missing for ${name}: ${ckpt}" >&2
      exit 4
    fi

    local eval_dir="${PROTOCOL_ROOT}/runs/protomotions/${split}/eval_${name}"
    mkdir -p "${eval_dir}"
    echo "[unified-proto] evaluating split=${split} method=${name}"
    local pids=()
    for shard in $(seq "${SHARD_START}" $((SHARD_START + LOCAL_SHARDS - 1))); do
      local shard_pt="${motion_base}/${split}_g1_shard_${shard}.pt"
      if [[ ! -s "${shard_pt}" ]]; then
        echo "[unified-proto] ${split} ${name} shard ${shard}: empty"
        continue
      fi
      local log="${eval_dir}/shard_${shard}.log"
      local pred_root="${eval_dir}/predicted_shard_${shard}"
      if [[ "${SAVE_PREDICTED_MOTION_LIB}" == "1" ]]; then
        mkdir -p "${pred_root}"
      fi
      if [[ "${FORCE_EVAL}" != "1" && -s "${log}" ]] && grep -q "EVALUATION RESULTS" "${log}"; then
        echo "[unified-proto] ${split} ${name} shard ${shard}: already done"
        continue
      fi
      (
        export CUDA_VISIBLE_DEVICES="$(( (shard - SHARD_START) % 8 ))"
        local save_override="agent.evaluator.save_predicted_motion_lib_every=None"
        local root_args=()
        if [[ "${SAVE_PREDICTED_MOTION_LIB}" == "1" ]]; then
          save_override="agent.evaluator.save_predicted_motion_lib_every=${PREDICTED_MOTION_LIB_EVERY}"
          root_args=(--root-dir "${pred_root}")
        fi
        cd "${PROTO_ROOT}"
        "${TRACKER_PY[@]}" protomotions/inference_agent.py \
          --checkpoint "${ckpt}" \
          --motion-file "${shard_pt}" \
          --simulator "${PROTO_SIMULATOR}" \
          --num-envs "${NUM_ENVS}" \
          --headless \
          --full-eval \
          "${root_args[@]}" \
          --overrides \
            "agent.evaluator.max_eval_steps=${MAX_EVAL_STEPS}" \
            "${save_override}" \
          > "${log}" 2>&1
      ) &
      pids+=("$!")
      if [[ "${#pids[@]}" -ge "${MAX_EVAL_JOBS}" ]]; then
        wait_for_background "${split} ${name} eval" "${pids[@]}"
        pids=()
      fi
    done
    wait_for_background "${split} ${name} eval" "${pids[@]}"
  done
}

check_gpu_driver_preflight
setup_tracker_runtime
for split in ${SPLITS}; do
  convert_and_pack_split "${split}"
  eval_split "${split}"
done

echo "[unified-proto] done $(date)"
