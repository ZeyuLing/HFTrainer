#!/usr/bin/env bash
# Eval-only rerun for HY-Motion-Lite T2M after 135D convention/root alignment.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python3 -m pip install -q mmengine smplx torchgeometry einops

OUT_ROOT="${OUT_ROOT:-outputs/evaluation/hylite_t2m_rerun0607_rootalign}"
LOGDIR="${OUT_ROOT}/logs"
METRICS="${OUT_ROOT}/metrics"
mkdir -p "${LOGDIR}" "${METRICS}"

EVAL_CKPT="${EVAL_CKPT:-checkpoints/motion_clip/motionclip_base_1p_aug_hq}"
DATA_DIR="${DATA_DIR:-data/motionhub}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
N_REPEATS="${N_REPEATS:-20}"
SEED="${SEED:-42}"

run_eval() {
  local label="$1"
  local gpu="$2"
  local anno="$3"
  local pred="$4"
  local out_json="$5"
  local rewrite="${6:-}"
  local rewrite_args=()
  if [ -n "${rewrite}" ]; then
    rewrite_args=(--rewritten_caption_file "${rewrite}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt "${EVAL_CKPT}" \
    --anno_file "${anno}" \
    --data_dir "${DATA_DIR}" \
    --pred_dir "${pred}" \
    "${rewrite_args[@]}" \
    --forward_batch_size "${FORWARD_BATCH_SIZE}" \
    --chunk_size "${CHUNK_SIZE}" \
    --out_json "${out_json}" \
    --n_repeats "${N_REPEATS}" \
    --seed "${SEED}" \
    > "${LOGDIR}/evalonly_${label}.log" 2>&1
}

run_eval h3d 0 \
  data/annotation/test_hml3d.json \
  "${OUT_ROOT}/h3d_row2col_yaw" \
  "${METRICS}/hylite_h3d_row2col_yaw_orig_c64.json" &
pid_h3d=$!

run_eval mh_orig 1 \
  data/annotation/test_motionhub_t2m.json \
  "${OUT_ROOT}/mh_orig_row2col_yaw" \
  "${METRICS}/hylite_mh_origgen_row2col_yaw_rw_c64.json" \
  data/annotation/test_motionhub_t2m_rewritten.json &
pid_mh_orig=$!

wait "${pid_h3d}" "${pid_mh_orig}"

run_eval mh_rw 0 \
  data/annotation/test_motionhub_t2m.json \
  "${OUT_ROOT}/mh_rw_row2col_yaw" \
  "${METRICS}/hylite_mh_rwgen_row2col_yaw_rw_c64.json" \
  data/annotation/test_motionhub_t2m_rewritten.json

OUT_ROOT="${OUT_ROOT}" python3 - <<'PY' | tee "${OUT_ROOT}/summary_evalonly.txt"
import json
import math
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"]) / "metrics"
items = [
    ("h3d", root / "hylite_h3d_row2col_yaw_orig_c64.json"),
    ("mh_origgen", root / "hylite_mh_origgen_row2col_yaw_rw_c64.json"),
    ("mh_rwgen", root / "hylite_mh_rwgen_row2col_yaw_rw_c64.json"),
]
for label, path in items:
    d = json.load(open(path))
    def f(key):
        value = d.get(key, float("nan"))
        return "nan" if value is None or math.isnan(float(value)) else f"{float(value):.4f}"
    print(
        label,
        "samples", d.get("samples"),
        "R1", f("r_precision_pred_top1_mean"),
        "R3", f("r_precision_pred_top3_mean"),
        "FID", f("fid_mean"),
        "MM", f("mm_dist_pred_mean"),
        "Div", f("diversity_pred_mean"),
    )
PY

touch "${OUT_ROOT}/_DONE_EVALONLY"
