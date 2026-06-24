#!/usr/bin/env bash
set -euo pipefail

# =========================
# taiji_retarget_g1.sh — Batch SMPL/AMASS -> G1 retargeting (GMR) on Taiji.
#
# CPU-only sharded job. Each Taiji host processes items[INDEX::NUM_HOSTS] of the
# quality list and writes IsaacLab-AMP npz to --out-dir (default data/g1).
#
# Usage (inside Taiji container, called via taiji_submit --start-cmd):
#   bash tools/taiji_retarget_g1.sh [OUT_DIR] [QUALITY_LIST] [WORKERS]
# =========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

OUT_DIR="${1:-data/g1}"
QUALITY_LIST="${2:-data/hymotion_m2m_refine_data/data_quality_list/high_quality.json}"
WORKERS="${3:-$(nproc)}"

export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Sharding. Explicit WORLD_SIZE/RANK env take priority (used when launching one
# independent single-host job per shard). Otherwise derive from the Taiji
# multi-host env (NODE_LIST = comma-sep node IPs, INDEX = this host's rank).
if [[ -n "${WORLD_SIZE:-}" ]]; then
  WORLD_SIZE="${WORLD_SIZE}"
  RANK="${RANK:-${INDEX:-0}}"
elif [[ -n "${NODE_LIST:-}" ]]; then
  WORLD_SIZE="$(python3 -c 'import sys; print(len(sys.argv[1].split(",")))' "${NODE_LIST}")"
  RANK="${INDEX:-0}"
else
  WORLD_SIZE=1
  RANK="${RANK:-0}"
fi

echo "[retarget_g1] PROJ_ROOT=${PROJ_ROOT}"
echo "[retarget_g1] OUT_DIR=${OUT_DIR} QUALITY_LIST=${QUALITY_LIST}"
echo "[retarget_g1] WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} WORKERS=${WORKERS} nproc=$(nproc)"

# GMR (mink IK) deps may be absent in the Taiji image; install idempotently.
# The scipy<1.14 compat patches live in ref_repo/GMR (shared CephFS), so they
# apply regardless of the image's scipy version.
if ! python3 -c "import mink, daqp, loop_rate_limiters, smplx" 2>/dev/null; then
  echo "[retarget_g1] installing GMR deps (mink/daqp/loop_rate_limiters/smplx)..."
  python3 -m pip install --quiet mink daqp loop_rate_limiters smplx || true
fi
python3 -c "import mink, mujoco, smplx; print('[retarget_g1] deps OK')"

# OVERWRITE=1 forces re-retargeting even if the output npz already exists
# (used to regenerate stale/buggy outputs with fixed code).
OW_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OW_FLAG="--overwrite"
  echo "[retarget_g1] OVERWRITE=1 -> re-retargeting all items (ignoring existing)"
fi

python3 scripts/embodied/batch_retarget_g1_gmr.py \
  --quality-list "${QUALITY_LIST}" \
  --out-dir "${OUT_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --rank "${RANK}" \
  --workers "${WORKERS}" \
  ${OW_FLAG}

echo "[retarget_g1] rank ${RANK} finished."
