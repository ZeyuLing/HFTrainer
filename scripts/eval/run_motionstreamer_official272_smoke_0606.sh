#!/usr/bin/env bash
# MotionStreamer official HumanML3D-272 smoke checks.
#
# This is a debugging runner, not a paper-table runner. It keeps upstream
# HumanML3D ids so MotionStreamer's official 272 evaluator can align samples.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/motionstreamer_official272_smoke_0606}
NUM_GPUS=${NUM_GPUS:-8}
MAX_SAMPLES_PER_SHARD=${MAX_SAMPLES_PER_SHARD:-16}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-128}
COND=${COND:-5}
PREFIX_LATENT_SOURCE=${PREFIX_LATENT_SOURCE:-sample}
SAMPLING_METHOD=${SAMPLING_METHOD:-new_demo}
MS_CFG=${MS_CFG:-4.5}
MS_TEMPERATURE=${MS_TEMPERATURE:-1.0}
MS_ROOT=${MS_ROOT:-ref_repo/MotionStreamer/MotionStreamer}

mkdir -p "${OUT}/logs" "${OUT}/metrics"
echo "[start] out=${OUT} num_gpus=${NUM_GPUS} max_per_shard=${MAX_SAMPLES_PER_SHARD} eval_max=${EVAL_MAX_SAMPLES}"

T2M_DIR="${OUT}/t2m_official_eval"
mkdir -p "${T2M_DIR}" "${OUT}/logs/t2m"
echo "[t2m-generate] $(date)"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="${i}" "$PY" scripts/eval/gen_motionstreamer_smpl_npz.py \
    --dataset humanml3d \
    --out-dir "${T2M_DIR}" \
    --num-shards "${NUM_GPUS}" \
    --shard-index "${i}" \
    --max-samples "${MAX_SAMPLES_PER_SHARD}" \
    --caption-protocol original \
    --humanml3d-protocol official_eval \
    --skip-existing \
    > "${OUT}/logs/t2m/gen_s${i}.log" 2>&1 &
done
wait
echo "[t2m-generate-done] npz=$(find "${T2M_DIR}" -maxdepth 1 -name '*.npz' | wc -l)"

echo "[t2m-eval272] $(date)"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "${T2M_DIR}" \
  --tag motionstreamer_t2m_official_smoke \
  --max-samples "${EVAL_MAX_SAMPLES}" \
  --out-json "${OUT}/metrics/t2m_official272.json" \
  > "${OUT}/logs/t2m/eval272.log" 2>&1

TP2M_ROOT="${OUT}/tp2m_official_cond${COND}"
TP2M_DIR="${TP2M_ROOT}/cond${COND}_latent_prefix"
mkdir -p "${TP2M_ROOT}" "${OUT}/logs/tp2m_cond${COND}"
echo "[tp2m-generate] $(date)"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="${i}" "$PY" scripts/eval/gen_motionstreamer_tp2m_smpl_npz.py \
    --dataset humanml3d \
    --out-dir "${TP2M_ROOT}" \
    --gt-272-dir "${MS_ROOT}/humanml3d_272/motion_data" \
    --condition-num-frames "${COND}" \
    --num-shards "${NUM_GPUS}" \
    --shard-index "${i}" \
    --max-samples "${MAX_SAMPLES_PER_SHARD}" \
    --caption-protocol original \
    --humanml3d-min-motion-length 60 \
    --prefix-latent-source "${PREFIX_LATENT_SOURCE}" \
    --sampling-method "${SAMPLING_METHOD}" \
    --cfg "${MS_CFG}" \
    --temperature "${MS_TEMPERATURE}" \
    --skip-existing \
    > "${OUT}/logs/tp2m_cond${COND}/gen_s${i}.log" 2>&1 &
done
wait
echo "[tp2m-generate-done] npz=$(find "${TP2M_DIR}" -maxdepth 1 -name '*.npz' | wc -l)"

echo "[tp2m-eval272] $(date)"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "${TP2M_DIR}" \
  --tag "motionstreamer_tp2m_cond${COND}_official_smoke" \
  --max-samples "${EVAL_MAX_SAMPLES}" \
  --out-json "${OUT}/metrics/tp2m_cond${COND}_official272.json" \
  > "${OUT}/logs/tp2m_cond${COND}/eval272.log" 2>&1

"$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("OUT", "outputs/evaluation/motionstreamer_official272_smoke_0606")) / "metrics"
for path in sorted(root.glob("*.json")):
    d = json.loads(path.read_text())
    pred = d.get("pred", {})
    gt = d.get("gt_real", {})
    print(path.name, {
        "ids": d.get("ids_with_required_files"),
        "gt_r3": (gt.get("r_precision") or [None, None, None])[2],
        "pred_r3": (pred.get("r_precision") or [None, None, None])[2],
        "fid": pred.get("fid_vs_gt_native"),
        "mm": pred.get("matching_score"),
        "div": pred.get("diversity"),
    })
PY
echo "[done] $(date)"
