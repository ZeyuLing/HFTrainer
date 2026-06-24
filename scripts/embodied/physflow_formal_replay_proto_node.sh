#!/usr/bin/env bash
# Wait for the formal replay pool, then train ProtoMotions tracker on it.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
POOL="${POOL:-${REPO}/output/generator_tracker_replay/physflow_hg1_formal2k_20260617}"
WAIT_SLEEP_SEC="${WAIT_SLEEP_SEC:-300}"
WAIT_MAX_LOOPS="${WAIT_MAX_LOOPS:-1728}"

cd "${REPO}"
echo "[formal-replay-proto] start $(date)"
echo "[formal-replay-proto] pool=${POOL}"

ready=0
for i in $(seq 1 "${WAIT_MAX_LOOPS}"); do
  if [[ -s "${POOL}/manifest.json" ]] && find "${POOL}/proto" -type f -name '*.motion' -print -quit | grep -q .; then
    ready=1
    break
  fi
  if [[ "${i}" -le 3 || "$((i % 10))" -eq 0 ]]; then
    echo "[formal-replay-proto] wait loop=${i}/${WAIT_MAX_LOOPS}; pool not ready"
  fi
  sleep "${WAIT_SLEEP_SEC}"
done

if [[ "${ready}" != "1" ]]; then
  echo "[formal-replay-proto] ERROR: timed out waiting for replay pool" >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-genproto_formal2k_20260617}" \
OUT_ROOT="${OUT_ROOT:-${REPO}/output/gt_replay_tracker_train_eval/genproto_formal2k_20260617}" \
TRAIN_MOTION_DIR="${POOL}/proto" \
TRAINING_MAX_STEPS="${TRAINING_MAX_STEPS:-1000000}" \
MAX_PER_GROUP="${MAX_PER_GROUP:-8192}" \
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-4}" \
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-512}" \
NUM_ENVS="${NUM_ENVS:-512}" \
BATCH_SIZE="${BATCH_SIZE:-4096}" \
NGPU="${NGPU:-1}" \
bash scripts/embodied/run_gt_replay_tracker_train_eval.sh

echo "[formal-replay-proto] done $(date)"
