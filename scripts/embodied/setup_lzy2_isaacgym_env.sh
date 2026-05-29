#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
VENV="${PHYSFLOW_ISAACGYM_VENV:-/root/physflow_isaacgym_py38_cu118}"
ISAAC_ROOT="${PHYSFLOW_ISAACGYM_ROOT:-/root/isaacgym}"
LOG_DIR="${PROJECT_ROOT}/output/physflow_kimodo_g1/lzy2_isaacgym_env"
LOG_FILE="${LOG_DIR}/setup.log"
LOCK_FILE="${LOG_DIR}/setup.lock"
ISAAC_URL="${PHYSFLOW_ISAACGYM_URL:-https://developer.nvidia.com/isaac-gym-preview-4}"

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
    echo "[setup] isaac_root=${ISAAC_ROOT}"

    if ! command -v python3.8 >/dev/null 2>&1; then
        echo "[setup] installing python38/python38-pip via dnf"
        dnf install -y python38 python38-pip python38-devel
    fi

    python3.8 --version

    if [[ ! -x "${VENV}/bin/python" ]]; then
        python3.8 -m venv "${VENV}"
    fi

    "${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel

    if [[ ! -d "${ISAAC_ROOT}/python/isaacgym" ]]; then
        mkdir -p "$(dirname "${ISAAC_ROOT}")"
        tmp_tar="/tmp/isaacgym_preview4.tar.gz"
        echo "[setup] downloading IsaacGym Preview4"
        wget -O "${tmp_tar}" "${ISAAC_URL}"
        rm -rf "${ISAAC_ROOT}"
        mkdir -p "${ISAAC_ROOT}.extract"
        tar -xzf "${tmp_tar}" -C "${ISAAC_ROOT}.extract"
        extracted="$(find "${ISAAC_ROOT}.extract" -maxdepth 2 -type d -name python | head -1 | xargs -r dirname)"
        if [[ -z "${extracted}" ]]; then
            echo "[setup] could not locate extracted IsaacGym root" >&2
            find "${ISAAC_ROOT}.extract" -maxdepth 3 -type d | head -50 >&2
            exit 2
        fi
        mv "${extracted}" "${ISAAC_ROOT}"
        rm -rf "${ISAAC_ROOT}.extract" "${tmp_tar}"
    fi

    "${VENV}/bin/python" -m pip install \
        'torch==2.4.1+cu118' 'torchvision==0.19.1+cu118' \
        --index-url https://download.pytorch.org/whl/cu118

    "${VENV}/bin/python" -m pip install --no-deps -e "${ISAAC_ROOT}/python"

    cd "${PROJECT_ROOT}/ref_repo/ProtoMotions"
    "${VENV}/bin/python" -m pip install -e .
    tmp_req="/tmp/physflow_requirements_isaacgym_no_torch.txt"
    grep -vE '^torch([<>= ]|$)' requirements_isaacgym.txt > "${tmp_req}"
    "${VENV}/bin/python" -m pip install --no-cache-dir -r "${tmp_req}"

    "${VENV}/bin/python" - <<'PY'
import importlib.util
import sys

print("python", sys.version)
for name in ("torch", "isaacgym", "lightning", "lightning_fabric", "tensordict", "hydra", "omegaconf"):
    spec = importlib.util.find_spec(name)
    print(f"import_check {name}: {'OK' if spec else 'MISSING'}")

import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
PY

    echo "[setup] done $(date)"
} >> "${LOG_FILE}" 2>&1
