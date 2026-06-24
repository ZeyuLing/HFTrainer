#!/usr/bin/env bash
# Wait for formal 2k frozen eval outputs, then build the formal generator replay pool.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
EVAL_ROOT="${EVAL_ROOT:-${REPO}/output/physflow_verify_hymotion_g1_130k_safe}"
OUT="${OUT:-${REPO}/output/generator_tracker_replay/physflow_hg1_formal2k_20260617}"
WAIT_SLEEP_SEC="${WAIT_SLEEP_SEC:-300}"
WAIT_MAX_LOOPS="${WAIT_MAX_LOOPS:-1728}"
REQUIRED_ITERS="${REQUIRED_ITERS:-2000}"
REQUIRED_METHODS="${REQUIRED_METHODS:-proto any2track humanoidgpt}"

cd "${REPO}"
echo "[formal-replay-pool] start $(date)"
echo "[formal-replay-pool] eval_root=${EVAL_ROOT}"
echo "[formal-replay-pool] out=${OUT}"

ready=0
for i in $(seq 1 "${WAIT_MAX_LOOPS}"); do
  missing=()
  for method in ${REQUIRED_METHODS}; do
    summary="${EVAL_ROOT}/${method}_iter${REQUIRED_ITERS}_frozen_eval/summary.json"
    if [[ ! -s "${summary}" ]]; then
      missing+=("${method}:${summary}")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    ready=1
    break
  fi
  if [[ "${i}" -le 3 || "$((i % 10))" -eq 0 ]]; then
    echo "[formal-replay-pool] wait loop=${i}/${WAIT_MAX_LOOPS}; missing=${missing[*]}"
  fi
  sleep "${WAIT_SLEEP_SEC}"
done

if [[ "${ready}" != "1" ]]; then
  echo "[formal-replay-pool] ERROR: timed out waiting for formal eval outputs" >&2
  exit 2
fi

python3 scripts/embodied/build_generator_tracker_replay_pool.py \
  --source-root "${EVAL_ROOT}" \
  --out "${OUT}" \
  --mode copy \
  --force

python3 - "${OUT}/manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
print("[formal-replay-pool] count", manifest["count"], "proto", manifest["proto_count"])
if manifest["count"] <= 0 or manifest["proto_count"] <= 0:
    raise SystemExit("formal replay pool is empty")
PY

echo "[formal-replay-pool] done $(date)"
