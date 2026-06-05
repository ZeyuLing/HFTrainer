#!/usr/bin/env bash
# Periodically record PhysFlow Taiji status, training logs, and checkpoint state.
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${PROJECT_ROOT}"

HY_TASK="${HY_TASK:-physflow_hymotion_real_mn4_20260604_070003}"
JUMP_TASK="${JUMP_TASK:-physflow_jump_overfit_hml3d40_20260604_073941}"

HY_LOG="${HY_LOG:-work_dirs/physflow_online_adv_mn_hymotion_real/20260604_073041/train.log}"
JUMP_LOG="${JUMP_LOG:-output/physflow_kimodo_g1/jump_hml3d40_noscene_mn1500_reset_env256_40m_noproj/launch.log}"
JUMP_CKPT="${JUMP_CKPT:-ref_repo/ProtoMotions/results/physflow_g1_jump_overfit_jump_hml3d40_noscene_mn1500_reset_env256_40m_noproj/last.ckpt}"

INTERVAL="${MONITOR_INTERVAL:-600}"
MAX_ROUNDS="${MONITOR_MAX_ROUNDS:-36}"

if [[ -z "${TOKEN:-}" && -r /root/.codex/skills/taiji/.token ]]; then
  export TOKEN
  TOKEN="$(cat /root/.codex/skills/taiji/.token)"
fi

if [[ -z "${TOKEN:-}" ]]; then
  echo "ERROR: TOKEN is not set and /root/.codex/skills/taiji/.token is not readable." >&2
  exit 1
fi

for round in $(seq 1 "${MAX_ROUNDS}"); do
  echo "========== $(date '+%Y-%m-%d %H:%M:%S') round ${round}/${MAX_ROUNDS} =========="

  echo "--- HY Taiji status ---"
  taiji_client il "${HY_TASK}" || true
  echo "--- HY latest step ---"
  tail -5 "${HY_LOG}" 2>/dev/null || echo "HY_LOG_NOT_FOUND ${HY_LOG}"
  echo "--- HY checkpoints ---"
  find work_dirs/physflow_online_adv_mn_hymotion_real -maxdepth 2 -type d -name 'checkpoint-*' \
    -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -5 || true

  echo "--- Jump Taiji status ---"
  taiji_client il "${JUMP_TASK}" || true
  echo "--- Jump latest epochs/errors ---"
  if [[ -f "${JUMP_LOG}" ]]; then
    rg -n "Epoch [0-9]+|Saved checkpoint|Traceback|RuntimeError|ERROR|illegal memory|Time Report|done" "${JUMP_LOG}" \
      | tail -40 || true
  else
    echo "JUMP_LOG_NOT_FOUND ${JUMP_LOG}"
  fi

  echo "--- Jump checkpoint meta ---"
  if [[ -f "${JUMP_CKPT}" ]]; then
    python3 - "${JUMP_CKPT}" <<'PY'
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
for key in ("epoch", "step_count", "best_evaluated_score"):
    print(f"{key}: {ckpt.get(key)}")
PY
  else
    echo "JUMP_CKPT_NOT_FOUND ${JUMP_CKPT}"
  fi

  if [[ "${round}" -lt "${MAX_ROUNDS}" ]]; then
    echo "--- sleeping ${INTERVAL}s ---"
    sleep "${INTERVAL}"
  fi
done
