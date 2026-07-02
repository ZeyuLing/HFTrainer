#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy7/share_305994131/home/zeyuling/hf_trainer}"
SONIC_REPO="${SONIC_REPO:-${PROJECT_ROOT}/ref_repo/GR00T-WholeBodyControl}"
ISAACLAB_REPO="${ISAACLAB_REPO:-${PROJECT_ROOT}/ref_repo/IsaacLab-2.3.2}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3.10}"
VENV="${PHYSFLOW_SONIC_ISAACLAB_VENV:-/root/physflow_sonic_isaaclab_py310}"
LOG_DIR="${PROJECT_ROOT}/output/physflow_sonic/isaaclab_env"
LOG_FILE="${LOG_DIR}/setup.log"
LOCK_FILE="${LOG_DIR}/setup.lock"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-4.5.0.0}"
GLIBC_LD="${GLIBC_LD:-}"
BASE_LIB="${BASE_LIB:-}"

mkdir -p "${LOG_DIR}"

if [[ -e "${LOCK_FILE}" ]] && kill -0 "$(cat "${LOCK_FILE}")" 2>/dev/null; then
  echo "[setup-sonic] already running pid=$(cat "${LOCK_FILE}")"
  exit 0
fi

echo "$$" > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

{
  echo "[setup-sonic] start $(date)"
  echo "[setup-sonic] host=$(hostname)"
  echo "[setup-sonic] project=${PROJECT_ROOT}"
  echo "[setup-sonic] sonic_repo=${SONIC_REPO}"
  echo "[setup-sonic] isaaclab_repo=${ISAACLAB_REPO}"
  echo "[setup-sonic] isaacsim_version=${ISAACSIM_VERSION}"
  echo "[setup-sonic] python_bin=${PYTHON_BIN}"
  echo "[setup-sonic] venv=${VENV}"

  "${PYTHON_BIN}" --version
  if [[ ! -x "${VENV}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV}"
  fi

  run_vpy() {
    if [[ -n "${GLIBC_LD}" && -x "${GLIBC_LD}" ]]; then
      "${GLIBC_LD}" --library-path "${BASE_LIB:-/lib64:/usr/lib64}:${LD_LIBRARY_PATH:-}" "${VENV}/bin/python" "$@"
    else
      "${VENV}/bin/python" "$@"
    fi
  }

  PIP_CONSTRAINT_FILE="${LOG_DIR}/pip_constraints.txt"
  cat > "${PIP_CONSTRAINT_FILE}" <<'EOF'
setuptools<81
EOF
  export PIP_CONSTRAINT="${PIP_CONSTRAINT_FILE}"
  export PIP_BUILD_CONSTRAINT="${PIP_CONSTRAINT_FILE}"

  run_vpy -m ensurepip --upgrade || true
  run_vpy -m pip install --upgrade pip "setuptools<81" wheel

  run_vpy -m pip install \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

  run_vpy -m pip install \
    "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

  # Taiji H20 containers do not provide apt-get. Avoid IsaacLab's wrapper-level
  # system dependency installer and install the Python extensions directly.
  run_vpy -m pip install "cmake>=3.27,<4"
  PIP_BUILD_CONSTRAINT_SAVED="${PIP_BUILD_CONSTRAINT:-}"
  unset PIP_BUILD_CONSTRAINT
  run_vpy -m pip install --no-build-isolation "flatdict==4.0.1"
  export PIP_BUILD_CONSTRAINT="${PIP_BUILD_CONSTRAINT_SAVED}"
  export PATH="${VENV}/bin:${PATH}"

  (
    cd "${ISAACLAB_REPO}"
    export VIRTUAL_ENV="${VENV}"
    export ISAACLAB_PATH="${ISAACLAB_REPO}"
    for ext_dir in source/*; do
      [[ -d "${ext_dir}" ]] || continue
      run_vpy -m pip install -e "${ext_dir}"
    done
    run_vpy -m pip install -e "source/isaaclab_rl[all]" -e "source/isaaclab_mimic[all]"
  )

  cd "${SONIC_REPO}"
  run_vpy -m pip install -e "gear_sonic/[training]"

  run_vpy check_environment.py --training || true
  run_vpy - <<'PY'
import importlib.util
import sys

print("python", sys.version)
for name in ("torch", "isaaclab", "isaacsim", "omni", "hydra", "trl", "gear_sonic"):
    spec = importlib.util.find_spec(name)
    print(f"import_check {name}: {'OK' if spec else 'MISSING'}")
PY

  echo "[setup-sonic] done $(date)"
} >> "${LOG_FILE}" 2>&1
