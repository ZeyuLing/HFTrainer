#!/usr/bin/env bash
# Run one formal PhysFlow tracker-reward experiment on a Taiji V100 node.
#
# The base vermo image does not always include MuJoCo/ONNX judge deps, but the
# online reward builds those judge modules inside tools/train.py. Install/check
# the light py3.10 judge deps before entering the long formal training run.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
ENVDIR="${ENVDIR:-/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env}"
CONFIG="${CONFIG:?CONFIG is required}"

cd "${REPO}"
echo "[formal-trreward] start $(date) host=$(hostname)"
echo "[formal-trreward] config=${CONFIG}"
nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
if [[ -n "${MIN_CUDA_DRIVER_MAJOR:-}" ]]; then
  DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)"
  echo "[formal-trreward] driver_major=${DRIVER_MAJOR:-unknown} required>=${MIN_CUDA_DRIVER_MAJOR}${MAX_CUDA_DRIVER_MAJOR:+ and <=${MAX_CUDA_DRIVER_MAJOR}}"
  if [[ -z "${DRIVER_MAJOR}" || "${DRIVER_MAJOR}" -lt "${MIN_CUDA_DRIVER_MAJOR}" ]]; then
    echo "[formal-trreward] FATAL_BAD_NODE: driver ${DRIVER_MAJOR:-unknown} < ${MIN_CUDA_DRIVER_MAJOR}" >&2
    exit 42
  fi
  if [[ -n "${MAX_CUDA_DRIVER_MAJOR:-}" && "${DRIVER_MAJOR}" -gt "${MAX_CUDA_DRIVER_MAJOR}" ]]; then
    echo "[formal-trreward] FATAL_BAD_NODE: driver ${DRIVER_MAJOR:-unknown} > ${MAX_CUDA_DRIVER_MAJOR}" >&2
    exit 44
  fi
fi

# ProtoMotions reward converts KIMODO/G1 CSV into ProtoMotions .motion files in
# the IsaacGym py3.8 runtime. Without this, convert failures are mapped to the
# reward penalty and look like a real "all samples fall" signal.
ln -sfn "${ENVDIR}/isaacgym" /root/isaacgym
ln -sfn "${ENVDIR}/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -1 || true
PY38RT="${ENVDIR}/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[formal-trreward] restoring base python3.8 from ${PY38RT}"
  cp -a "${PY38RT}/bin/python3.8" /usr/bin/python3.8
  cp -a "${PY38RT}/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "${PY38RT}/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
/root/physflow_isaacgym_py38_cu118/bin/python - <<'PY'
import isaacgym
import torch
print("[formal-trreward] py38 converter OK", "torch", torch.__version__)
PY

if [[ -z "${TRAIN_PYTHON:-}" ]]; then
  for cand in /usr/local/bin/python3.10 /usr/local/bin/python3 python3.10 python3; do
    if command -v "${cand}" >/dev/null 2>&1 && "${cand}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)
PY
    then
      TRAIN_PYTHON="$(command -v "${cand}")"
      break
    fi
  done
fi
if [[ -z "${TRAIN_PYTHON:-}" ]]; then
  echo "[formal-trreward] FATAL_NO_TRAIN_PYTHON: no usable Python >=3.10" >&2
  exit 43
fi
"${TRAIN_PYTHON}" - <<'PY'
import sys
print("[formal-trreward] train python", sys.executable, sys.version.split()[0])
PY

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
timeout 600 "${TRAIN_PYTHON}" -m pip install --quiet \
  -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -3 \
  || echo "[formal-trreward] WARN judge dep install partial; trying import check anyway"

"${TRAIN_PYTHON}" - <<'PY'
import mujoco
import onnxruntime
import dm_control
import typer
print("[formal-trreward] imports OK", "mujoco", mujoco.__version__, "onnxruntime", onnxruntime.__version__)
PY

export PATH=/usr/local/bin:${PATH}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PHYSFLOW_CONVERT_PYTHON="${PHYSFLOW_CONVERT_PYTHON:-/root/physflow_isaacgym_py38_cu118/bin/python}"

TRAIN_ARGS=("${CONFIG}" "--auto-resume")
if [[ -n "${WORK_DIR_OVERRIDE:-}" ]]; then
  TRAIN_ARGS+=("--work-dir" "${WORK_DIR_OVERRIDE}")
fi
if [[ -n "${TRAIN_CFG_OPTIONS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_CFG_OPTIONS=(${TRAIN_CFG_OPTIONS})
  TRAIN_ARGS+=("--cfg-options" "${EXTRA_CFG_OPTIONS[@]}")
fi

if [[ "${ACCELERATE_NUM_PROCESSES:-1}" -gt 1 ]]; then
  echo "[formal-trreward] accelerate launch num_processes=${ACCELERATE_NUM_PROCESSES} port=${ACCELERATE_PORT:-29555}"
  "${TRAIN_PYTHON}" -m accelerate.commands.launch \
    --num_processes="${ACCELERATE_NUM_PROCESSES}" \
    --num_machines="${ACCELERATE_NUM_MACHINES:-1}" \
    --mixed_precision="${ACCELERATE_MIXED_PRECISION:-no}" \
    --main_process_port="${ACCELERATE_PORT:-29555}" \
    tools/train.py "${TRAIN_ARGS[@]}"
else
  "${TRAIN_PYTHON}" tools/train.py "${TRAIN_ARGS[@]}"
fi
echo "[formal-trreward] done $(date)"
