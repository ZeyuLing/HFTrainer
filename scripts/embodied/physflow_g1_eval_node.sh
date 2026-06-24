#!/usr/bin/env bash
# Prepare the ProtoMotions/MuJoCo runtime on a Taiji node, then run a G1 eval
# command with the repo and converter environment set.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
ENVDIR="${ENVDIR:-/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env}"

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <python-script> [args...]" >&2
  exit 2
fi

cd "${REPO}"
echo "[g1-eval-node] start $(date) host=$(hostname)"
nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true

ln -sfn "${ENVDIR}/isaacgym" /root/isaacgym
ln -sfn "${ENVDIR}/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -1 || true

PY38RT="${ENVDIR}/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[g1-eval-node] restoring base python3.8 from ${PY38RT}"
  cp -a "${PY38RT}/bin/python3.8" /usr/bin/python3.8
  cp -a "${PY38RT}/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "${PY38RT}/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
/root/physflow_isaacgym_py38_cu118/bin/python - <<'PY'
import isaacgym
import torch
print("[g1-eval-node] py38 converter OK", "torch", torch.__version__)
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
  echo "[g1-eval-node] FATAL_NO_TRAIN_PYTHON: no usable Python >=3.10" >&2
  exit 43
fi

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
timeout 600 "${TRAIN_PYTHON}" -m pip install --quiet \
  -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -3 \
  || echo "[g1-eval-node] WARN judge dep install partial; trying import check anyway"

"${TRAIN_PYTHON}" - <<'PY'
import mujoco
import onnxruntime
import dm_control
import typer
print("[g1-eval-node] imports OK", "mujoco", mujoco.__version__, "onnxruntime", onnxruntime.__version__)
PY

export PATH=/usr/local/bin:${PATH}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PHYSFLOW_CONVERT_PYTHON="${PHYSFLOW_CONVERT_PYTHON:-/root/physflow_isaacgym_py38_cu118/bin/python}"

"${TRAIN_PYTHON}" "$@"
echo "[g1-eval-node] done $(date)"
