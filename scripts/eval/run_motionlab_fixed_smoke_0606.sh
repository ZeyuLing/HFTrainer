#!/usr/bin/env bash
# Smoke-test MotionLab after aligning the wrapper with the released checkpoint.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/motionlab_fixed_smoke_0606}
PRED_DIR="${OUT_ROOT}/pred"
LOGDIR="${OUT_ROOT}/logs"
GPU=${GPU:-0}
MAX_SAMPLES=${MAX_SAMPLES:-256}
NUM_REPEATS=${NUM_REPEATS:-3}
EXTRA_INFER_ARGS=${EXTRA_INFER_ARGS:-}

mkdir -p "${PRED_DIR}" "${LOGDIR}"
echo "[infer] $(date) gpu=${GPU} max=${MAX_SAMPLES}" | tee "${LOGDIR}/run.log"
CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/motionlab_infer_hml3d263.py \
  --out-dir "${PRED_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size 32 \
  --stage eval \
  --skip-existing \
  ${EXTRA_INFER_ARGS} \
  > "${LOGDIR}/infer.log" 2>&1

echo "[native-eval] $(date)" | tee -a "${LOGDIR}/run.log"
CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/eval_momask_native_h3d263.py \
  --recon_root work_dirs/h3d263_eval/h3d263_test_recon_fk \
  --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
  --momask_root ref_repo/Momask/momask-codes \
  --mode pred \
  --pred_dir "${PRED_DIR}" \
  --num_repeats "${NUM_REPEATS}" \
  --drop_mirrored \
  --caption_selection first \
  --max_samples "${MAX_SAMPLES}" \
  --output "${OUT_ROOT}/native_rep${NUM_REPEATS}.json" \
  > "${LOGDIR}/native_eval.log" 2>&1

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
p = Path("${OUT_ROOT}/native_rep${NUM_REPEATS}.json")
d = json.load(open(p))
print(
    "samples", d.get("n_samples"),
    "R1", f"{d['r_precision']['mean'][0]:.4f}",
    "R3", f"{d['r_precision']['mean'][2]:.4f}",
    "FID", f"{d['fid']['mean']:.4f}",
    "MM", f"{d['matching_score']['mean']:.4f}",
    "Div", f"{d['diversity']['mean']:.4f}",
)
PY
touch "${OUT_ROOT}/_DONE"
