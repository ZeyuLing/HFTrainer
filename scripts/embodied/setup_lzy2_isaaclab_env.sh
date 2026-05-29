#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
VENV="${PHYSFLOW_ISAACLAB_VENV:-/root/physflow_isaaclab_py311}"
LOG_DIR="${PROJECT_ROOT}/output/physflow_kimodo_g1/lzy2_isaaclab_env"
LOG_FILE="${LOG_DIR}/setup.log"
LOCK_FILE="${LOG_DIR}/setup.lock"

mkdir -p "${LOG_DIR}"

if [[ -e "${LOCK_FILE}" ]] && kill -0 "$(cat "${LOCK_FILE}")" 2>/dev/null; then
    echo "[setup] already running pid=$(cat "${LOCK_FILE}")"
    exit 0
fi
echo "$$" > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

{
    echo "[setup] start $(date)"
    echo "[setup] host=$(hostname)"
    echo "[setup] project=${PROJECT_ROOT}"
    echo "[setup] venv=${VENV}"
    /usr/bin/python3.11 --version

    if [[ ! -x "${VENV}/bin/python" ]]; then
        /usr/bin/python3.11 -m venv "${VENV}"
    fi

    "${VENV}/bin/python" -m ensurepip --upgrade || true
    "${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel

    "${VENV}/bin/python" -m pip install \
        torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128

    "${VENV}/bin/python" -m pip install \
        'isaaclab[isaacsim,all]==2.3.0' \
        --extra-index-url https://pypi.nvidia.com

    cd "${PROJECT_ROOT}/ref_repo/ProtoMotions"
    "${VENV}/bin/python" -m pip install -e .
    "${VENV}/bin/python" -m pip install -r requirements_isaaclab.txt

    "${VENV}/bin/python" - <<'PY'
import importlib.util
import sys

print("python", sys.version)
for name in ("torch", "isaaclab", "isaacsim", "lightning", "tensordict"):
    spec = importlib.util.find_spec(name)
    print(f"import_check {name}: {'OK' if spec else 'MISSING'}")
PY

    echo "[setup] done $(date)"
} >> "${LOG_FILE}" 2>&1
