#!/usr/bin/env bash
# Framework-native MotionGPT HumanML3D official-test inference.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

BASE="${BASE:-outputs/evaluation/t2m/humanml3d_official_test}"
HML_DIR="${HML_DIR:-${BASE}/hml263/motiongpt}"
ARTIFACT_DIR="${ARTIFACT_DIR:-checkpoints/baselines/motiongpt}"
RUN_TAG="${RUN_TAG:-motiongpt_framework_native_20260626}"
RUN_ROOT="${RUN_ROOT:-${BASE}/_runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
METRIC_DIR="${METRIC_DIR:-${RUN_ROOT}/metrics}"

TOTAL_SHARDS="${TOTAL_SHARDS:-${NUM_GPUS:-4}}"
LOCAL_SHARDS="${LOCAL_SHARDS:-${TOTAL_SHARDS}}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
PROMPT_MODE="${PROMPT_MODE:-official_nolen}"
DEVICE_IDS="${DEVICE_IDS:-}"

mkdir -p "${HML_DIR}" "${LOG_DIR}" "${METRIC_DIR}"

if [ -z "${DEVICE_IDS}" ]; then
  DEVICE_IDS="$(python3 - <<'PY'
import torch
n = torch.cuda.device_count()
print(",".join(str(i) for i in range(n)))
PY
)"
fi
IFS=',' read -r -a GPUS <<< "${DEVICE_IDS}"
if [ "${#GPUS[@]}" -lt "${LOCAL_SHARDS}" ]; then
  echo "[error] LOCAL_SHARDS=${LOCAL_SHARDS} but DEVICE_IDS=${DEVICE_IDS}" >&2
  exit 1
fi

cat > "${RUN_ROOT}/run_config.json" <<JSON
{
  "method": "motiongpt",
  "model_bundle": "hftrainer.models.motion.motiongpt.MotionGPTBundle",
  "pipeline": "hftrainer.pipelines.motiongpt.MotionGPTPipeline",
  "artifact_dir": "${ARTIFACT_DIR}",
  "out_dir": "${HML_DIR}",
  "caption_protocol": "humanml3d_official_corrected_caption",
  "annotation": "${BASE}/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json",
  "prompt_mode": "${PROMPT_MODE}",
  "seed": ${SEED},
  "total_shards": ${TOTAL_SHARDS},
  "local_shards": ${LOCAL_SHARDS},
  "shard_offset": ${SHARD_OFFSET},
  "batch_size": ${BATCH_SIZE}
}
JSON

echo "[start] MotionGPT HML263 inference $(date -Is)" | tee "${LOG_DIR}/motiongpt.log"
echo "[paths] hml=${HML_DIR} artifact=${ARTIFACT_DIR}" | tee -a "${LOG_DIR}/motiongpt.log"
echo "[shards] total=${TOTAL_SHARDS} local=${LOCAL_SHARDS} offset=${SHARD_OFFSET} gpus=${DEVICE_IDS}" | tee -a "${LOG_DIR}/motiongpt.log"

pids=()
for local_idx in $(seq 0 $((LOCAL_SHARDS - 1))); do
  shard=$((SHARD_OFFSET + local_idx))
  gpu="${GPUS[$local_idx]}"
  log="${LOG_DIR}/infer_s${shard}.log"
  echo "[launch] shard=${shard}/${TOTAL_SHARDS} gpu=${gpu} log=${log}" | tee -a "${LOG_DIR}/motiongpt.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/framework_t2m_hml263_infer.py \
    --method motiongpt \
    --artifact-dir "${ARTIFACT_DIR}" \
    --out-dir "${HML_DIR}" \
    --num-shards "${TOTAL_SHARDS}" \
    --shard-index "${shard}" \
    --batch-size "${BATCH_SIZE}" \
    --skip-existing \
    --seed "${SEED}" \
    --motiongpt-local-files-only \
    --motiongpt-prompt-mode "${PROMPT_MODE}" \
    > "${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  echo "[fail] one or more MotionGPT shards failed $(date -Is)" | tee -a "${LOG_DIR}/motiongpt.log"
  exit 1
fi

RUN_ROOT="${RUN_ROOT}" HML_DIR="${HML_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

hml = Path(os.environ["HML_DIR"])
files = sorted(hml.glob("*.npy"))
summary = {
    "method": "motiongpt",
    "representation": "hml263",
    "out_dir": str(hml),
    "count": len(files),
    "expected": 4042,
    "complete": len(files) == 4042,
}
out = Path(os.environ["RUN_ROOT"]) / "metrics" / "hml263_coverage.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
if not summary["complete"]:
    raise SystemExit(1)
PY

echo "[done] MotionGPT HML263 inference $(date -Is)" | tee -a "${LOG_DIR}/motiongpt.log"
