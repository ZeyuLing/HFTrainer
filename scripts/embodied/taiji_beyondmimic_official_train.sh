#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
BM_ROOT="${BM_ROOT:-${ROOT}/ref_repo/BeyondMimic}"
MODE="${BEYONDMIMIC_MODE:-preflight}"
TAG="${BEYONDMIMIC_TAG:-beyondmimic_${MODE}_$(date +%Y%m%d_%H%M%S)}"
MOTION_NAME="${BEYONDMIMIC_MOTION_NAME:-dance1_subject1}"
MAX_ITERATIONS="${BEYONDMIMIC_MAX_ITERATIONS:-20}"
NUM_ENVS="${BEYONDMIMIC_NUM_ENVS:-512}"
PYTHON_BIN="${BEYONDMIMIC_SYSTEM_PYTHON:-/usr/bin/python3.11}"
VENV="${BEYONDMIMIC_VENV:-/root/beyondmimic_isaacsim_py311}"
DATA_ROOT="${BEYONDMIMIC_DATA_ROOT:-${ROOT}/data/BeyondMimic_LAFAN1_Retargeting_Dataset}"
LOG_DIR="${ROOT}/output/beyondmimic_official/${TAG}"
MOTION_CSV="${DATA_ROOT}/g1/${MOTION_NAME}.csv"
MOTION_NPZ="${LOG_DIR}/motions/${MOTION_NAME}.npz"
RUN_NAME="${TAG}_${MOTION_NAME}"

mkdir -p "${LOG_DIR}/motions"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1

echo "[beyondmimic] start $(date)"
echo "[beyondmimic] host=$(hostname)"
echo "[beyondmimic] root=${ROOT}"
echo "[beyondmimic] mode=${MODE} tag=${TAG}"
echo "[beyondmimic] motion=${MOTION_NAME}"
echo "[beyondmimic] max_iterations=${MAX_ITERATIONS} num_envs=${NUM_ENVS}"

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACSIM_ACCEPT_EULA=YES
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"
export PIP_CACHE_DIR="${BEYONDMIMIC_PIP_CACHE_DIR:-${ROOT}/output/beyondmimic_official/pip_cache}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
mkdir -p "${PIP_CACHE_DIR}"

PIP_CONSTRAINT_FILE="${LOG_DIR}/pip_constraints.txt"
cat > "${PIP_CONSTRAINT_FILE}" <<'EOF'
setuptools<81
EOF
export PIP_CONSTRAINT="${PIP_CONSTRAINT_FILE}"
export PIP_BUILD_CONSTRAINT="${PIP_CONSTRAINT_FILE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
"${PYTHON_BIN}" --version

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
PY="${VENV}/bin/python"
"${PY}" -m ensurepip --upgrade || true
"${PY}" -m pip install --upgrade pip "setuptools<81" wheel

REQ_FILTERED="${LOG_DIR}/requirements.filtered.txt"
python3 - "${BM_ROOT}/requirements.txt" "${REQ_FILTERED}" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
skip = (
    "git.midea.com/robotics/future-appliance",
    "github.com/HybridRobotics/whole_body_tracking",
)
lines = []
for line in src.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if any(token in stripped for token in skip):
        print(f"[beyondmimic] skip private requirement: {stripped}")
        continue
    lines.append(line)
dst.write_text("\n".join(lines) + "\n")
PY

READY_MARK="${VENV}/.beyondmimic_requirements_ready"
if [[ ! -e "${READY_MARK}" ]]; then
  "${PY}" -m pip install -r "${REQ_FILTERED}" --extra-index-url https://pypi.nvidia.com --retries "${PIP_RETRIES}" --timeout "${PIP_DEFAULT_TIMEOUT}"
  touch "${READY_MARK}"
else
  echo "[beyondmimic] requirements already installed: ${READY_MARK}"
fi

cd "${BM_ROOT}"
"${PY}" -m pip install -e source/whole_body_tracking

ASSET_DIR="${BM_ROOT}/source/whole_body_tracking/whole_body_tracking/assets"
if [[ ! -d "${ASSET_DIR}/unitree_description" ]]; then
  echo "[beyondmimic] downloading Unitree robot descriptions"
  curl -L -o "${LOG_DIR}/unitree_description.tar.gz" \
    https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
  tar -xzf "${LOG_DIR}/unitree_description.tar.gz" -C "${ASSET_DIR}/"
fi

if [[ ! -f "${MOTION_CSV}" ]]; then
  echo "[beyondmimic] downloading HF LAFAN-G1 csv: ${MOTION_NAME}"
  "${PY}" - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="lvhaidong/LAFAN1_Retargeting_Dataset",
    repo_type="dataset",
    filename="g1/${MOTION_NAME}.csv",
    local_dir="${DATA_ROOT}",
    local_dir_use_symlinks=False,
)
PY
fi

"${PY}" - <<'PY'
import importlib.util
import sys
print("python", sys.version)
for name in ("torch", "isaacsim", "isaaclab", "isaaclab_rl", "isaaclab_tasks", "rsl_rl", "whole_body_tracking"):
    spec = importlib.util.find_spec(name)
    print(f"import_check {name}: {'OK' if spec else 'MISSING'}")
    if spec is None:
        raise SystemExit(2)
PY

echo "[beyondmimic] preprocessing csv to BeyondMimic npz"
"${PY}" scripts/csv_to_npz.py \
  --input_file "${MOTION_CSV}" \
  --input_fps 30 \
  --output_name "${MOTION_NAME}" \
  --output_file "${MOTION_NPZ}" \
  --output_fps 50 \
  --headless \
  --no_wandb

echo "[beyondmimic] training"
"${PY}" scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --motion_file "${MOTION_NPZ}" \
  --headless \
  --logger tensorboard \
  --run_name "${RUN_NAME}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --seed 1

RUN_DIR="$(find "${BM_ROOT}/logs/rsl_rl/g1_flat" -maxdepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -1)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "[beyondmimic] ERROR: could not find run dir for ${RUN_NAME}" >&2
  exit 3
fi
echo "${RUN_DIR}" > "${LOG_DIR}/run_dir.txt"
echo "[beyondmimic] run_dir=${RUN_DIR}"

CKPT="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -1)"
if [[ -z "${CKPT}" ]]; then
  echo "[beyondmimic] ERROR: no checkpoint found in ${RUN_DIR}" >&2
  exit 4
fi
echo "${CKPT}" > "${LOG_DIR}/checkpoint.txt"
echo "[beyondmimic] checkpoint=${CKPT}"

PLAY_VIDEO_LENGTH="${BEYONDMIMIC_PLAY_VIDEO_LENGTH:-300}"
LOAD_RUN="$(basename "${RUN_DIR}")"
LOAD_CKPT="$(basename "${CKPT}")"

echo "[beyondmimic] export/eval via play.py"
"${PY}" scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-v0 \
  --motion_file "${MOTION_NPZ}" \
  --headless \
  --video \
  --video_length "${PLAY_VIDEO_LENGTH}" \
  --num_envs 2 \
  --load_run "${LOAD_RUN}" \
  --checkpoint "${LOAD_CKPT}"

EXPORT_DIR="${RUN_DIR}/exported"
echo "${EXPORT_DIR}" > "${LOG_DIR}/export_dir.txt"
find "${RUN_DIR}" -maxdepth 4 -type f | sort > "${LOG_DIR}/artifacts.txt"
echo "[beyondmimic] completed $(date)"
