#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
TAG="${TAG:-physflow_adv_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/output/opentrack_physflow_adversarial/${TAG}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/train.log") 2>&1

echo "[train] root=${ROOT}"
echo "[train] tag=${TAG}"
date
nvidia-smi || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
MIN_CUDA_DRV="${MIN_CUDA_DRV:-11.4}"
MAX_CUDA_DRV="${MAX_CUDA_DRV:-}"
echo "[train] host CUDA driver version: ${CUDA_DRV:-unknown}; required >= ${MIN_CUDA_DRV}${MAX_CUDA_DRV:+ and <= ${MAX_CUDA_DRV}}"
if [ -n "${CUDA_DRV:-}" ] && awk "BEGIN{exit !(${CUDA_DRV} < ${MIN_CUDA_DRV})}"; then
  echo "[train] FATAL_BAD_NODE: CUDA driver ${CUDA_DRV} < ${MIN_CUDA_DRV}; use a CUDA11.4 V100 node for OpenTrack." >&2
  exit 86
fi
if [ -n "${CUDA_DRV:-}" ] && [ -n "${MAX_CUDA_DRV:-}" ] && awk "BEGIN{exit !(${CUDA_DRV} > ${MAX_CUDA_DRV})}"; then
  echo "[train] FATAL_BAD_NODE: CUDA driver ${CUDA_DRV} > ${MAX_CUDA_DRV}; use a CUDA11.4 V100 node for OpenTrack." >&2
  exit 87
fi

cd "${ROOT}/ref_repo/OpenTrack"
export GLI_PATH="${GLI_PATH:-${PWD}}"
CUDA_LIB_DIRS=()
for d in /usr/local/cuda*/extras/CUPTI/lib64 /usr/local/cuda*/lib64; do
  [[ -d "${d}" ]] && CUDA_LIB_DIRS+=("${d}")
done
if [[ "${#CUDA_LIB_DIRS[@]}" -gt 0 ]]; then
  CUDA_LD_PATH="$(IFS=:; echo "${CUDA_LIB_DIRS[*]}")"
  export LD_LIBRARY_PATH="${CUDA_LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "[train] LD_LIBRARY_PATH prepended with ${CUDA_LD_PATH}"
fi
if command -v uv >/dev/null 2>&1; then
  UV_CMD=(uv)
else
  python3 -m pip install --user -q uv
  UV_CMD=(python3 -m uv)
fi

OPEN_VENV_DIR="${OPENTRACK_VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
if [[ "${OPEN_VENV_DIR}" == /* ]]; then
  OPEN_VENV_PATH="${OPEN_VENV_DIR}"
else
  OPEN_VENV_PATH="${PWD}/${OPEN_VENV_DIR}"
fi
export UV_PROJECT_ENVIRONMENT="${OPEN_VENV_PATH}"
OPENTRACK_CUDA_FLAVOR="${OPENTRACK_CUDA_FLAVOR:-cuda11}"

if [[ "${SKIP_UV_SYNC:-0}" == "1" ]]; then
  echo "[train] SKIP_UV_SYNC=1; reusing ${OPEN_VENV_PATH}"
elif [[ -x "${OPEN_VENV_PATH}/bin/python" ]] && "${OPEN_VENV_PATH}/bin/python" - <<'PY' >/dev/null 2>&1
import track_mj, torch, mujoco, jax  # noqa: F401

assert hasattr(jax.config, "define_bool_state")
assert hasattr(jax.random, "KeyArray")
PY
then
  echo "[train] OpenTrack venv import check passed; skip env sync"
elif [[ "${OPENTRACK_CUDA_FLAVOR}" == "cuda11" ]]; then
  echo "[train] preparing OpenTrack CUDA11 env at ${OPEN_VENV_PATH}"
  PROJECT_ROOT="${ROOT}" OPENTRACK_ROOT="${PWD}" \
    bash "${ROOT}/scripts/embodied/prepare_opentrack_cuda11_env.sh" "${OPEN_VENV_PATH}"
else
  if [[ "${OPEN_VENV_PATH}" == "${PWD}/.venv" ]]; then
    LOCK_FILE="${ROOT}/work_dirs/opentrack_uv_sync.lock"
  else
    LOCK_FILE="${OPEN_VENV_PATH}.sync.lock"
  fi
  mkdir -p "$(dirname "${LOCK_FILE}")"
  echo "[train] acquiring uv sync lock: ${LOCK_FILE}"
  echo "[train] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
  flock "${LOCK_FILE}" "${UV_CMD[@]}" sync -i https://pypi.org/simple
fi
"${OPEN_VENV_PATH}/bin/python" "${ROOT}/scripts/embodied/patch_opentrack_brax_uint64.py"
source "${OPEN_VENV_PATH}/bin/activate"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export GLI_PATH="${GLI_PATH:-${PWD}}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
if [[ "${MUJOCO_GL}" == "egl" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
else
  unset PYOPENGL_PLATFORM
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

cd "${ROOT}"
TAG="${TAG}" \
ADV_SOURCE_DIR="${ADV_SOURCE_DIR:-${ROOT}/output/opentrack_amass_g1/debug2_20260604_1915_wait_proto_wxyz/UnitreeG1}" \
ADV_KEYWORDS="${ADV_KEYWORDS-jump,fall,getup,run,sprint}" \
ADV_MAX_FILES="${ADV_MAX_FILES:-96}" \
ADV_PROB="${ADV_PROB:-0.25}" \
NUM_GPUS="${NUM_GPUS:-8}" \
NUM_TIMESTEPS="${NUM_TIMESTEPS:-2000000000}" \
bash scripts/embodied/run_opentrack_physflow_adversarial.sh

echo "[train] completed"
date
