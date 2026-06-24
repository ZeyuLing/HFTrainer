#!/usr/bin/env bash
# Wait for the formal replay pool, then train Any2Track/OpenTrack on it.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
POOL="${POOL:-${REPO}/output/generator_tracker_replay/physflow_hg1_formal2k_20260617}"
WAIT_SLEEP_SEC="${WAIT_SLEEP_SEC:-300}"
WAIT_MAX_LOOPS="${WAIT_MAX_LOOPS:-1728}"

cd "${REPO}"
echo "[formal-replay-any2track] start $(date)"
echo "[formal-replay-any2track] pool=${POOL}"

ready=0
for i in $(seq 1 "${WAIT_MAX_LOOPS}"); do
  if [[ -s "${POOL}/manifest.json" ]] && find "${POOL}/qpos_npz" -type f -name '*.npz' -print -quit | grep -q .; then
    ready=1
    break
  fi
  if [[ "${i}" -le 3 || "$((i % 10))" -eq 0 ]]; then
    echo "[formal-replay-any2track] wait loop=${i}/${WAIT_MAX_LOOPS}; pool not ready"
  fi
  sleep "${WAIT_SLEEP_SEC}"
done

if [[ "${ready}" != "1" ]]; then
  echo "[formal-replay-any2track] ERROR: timed out waiting for replay pool" >&2
  exit 2
fi

TAG="${TAG:-physflow_gen_any2track_formal2k_20260617}" \
OPENTRACK_VENV_DIR="${OPENTRACK_VENV_DIR:-.venv_physflow_any2track_formal2k}" \
ADV_SOURCE_DIR="${POOL}/qpos_npz" \
ADV_KEYWORDS="" \
ADV_MAX_FILES=0 \
ADV_PROB="${ADV_PROB:-0.35}" \
NUM_GPUS="${NUM_GPUS:-8}" \
NUM_TIMESTEPS="${NUM_TIMESTEPS:-2000000000}" \
bash scripts/embodied/taiji_opentrack_physflow_adversarial_train.sh

echo "[formal-replay-any2track] done $(date)"
