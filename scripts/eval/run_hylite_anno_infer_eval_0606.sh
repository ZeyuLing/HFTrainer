#!/usr/bin/env bash
# HY-Motion-Lite annotation-key inference/evaluation for MotionHub-style test splits.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
ANNO_FILE="${ANNO_FILE:?ANNO_FILE is required}"
PRED_DIR="${OUT_ROOT}/motionclip135"
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${PRED_DIR}" "${LOGDIR}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-1,2,3,4}"
NUM_SHARDS="${NUM_SHARDS:-4}"
CAPTION_FILE="${CAPTION_FILE:-}"
CAPTION_ARGS=()
if [ -n "${CAPTION_FILE}" ]; then
  CAPTION_ARGS=(--caption-file "${CAPTION_FILE}")
fi

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hylite_t2m_anno_infer.py \
    --anno-file "${ANNO_FILE}" \
    "${CAPTION_ARGS[@]}" \
    --data-dir data/motionhub \
    --out-dir "${PRED_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${i}" \
    --gpu "${gpu}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --num-steps "${NUM_STEPS:-50}" \
    --cfg-scale "${CFG_SCALE:-5.0}" \
    --skip-existing \
    > "${LOGDIR}/infer_s${i}_gpu${gpu}.log" 2>&1 &
done
wait

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file "${ANNO_FILE}" \
  --data_dir data/motionhub \
  --pred_dir "${PRED_DIR}" \
  --chunk_size 64 \
  --out_json "${OUT_ROOT}/${OUT_NAME:-hylite_orig_c64.json}" \
  --n_repeats 20 \
  --seed 42 \
  > "${LOGDIR}/eval.log" 2>&1

OUT_ROOT="${OUT_ROOT}" OUT_NAME="${OUT_NAME:-hylite_orig_c64.json}" LABEL="${LABEL:-hylite}" python3 - <<'PY' | tee "${OUT_ROOT}/summary.txt"
import json
import os
from pathlib import Path

p = Path(os.environ["OUT_ROOT"]) / os.environ["OUT_NAME"]
d = json.load(open(p))
print(
    os.environ["LABEL"],
    "samples", d.get("samples"),
    "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
    "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
    "FID", f"{d.get('fid_mean', float('nan')):.4f}",
    "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
    "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
)
PY

touch "${OUT_ROOT}/_DONE"
