#!/usr/bin/env bash
# GT HML263 -> SMPLify3D smoke control.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/gt_hml263_control_smplify3d_smoke_0606}
SRC=${SRC:-outputs/evaluation/humanml3d/gt_smpl135_to_hml263/humanml3d}
LIMIT=${LIMIT:-16}
SMPLIFY_ITERS=${SMPLIFY_ITERS:-10}
GPU=${GPU:-2}
LOGDIR="${OUT_ROOT}/logs"

mkdir -p "${LOGDIR}" "${OUT_ROOT}/smpl_npz" "${OUT_ROOT}/motionclip135"
exec > >(tee -a "${LOGDIR}/run.log") 2>&1

echo "[start] $(date) out=${OUT_ROOT} limit=${LIMIT} iters=${SMPLIFY_ITERS} gpu=${GPU}"

CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/hml263_to_smpl_smplify3d.py \
  --in-dir "${SRC}" \
  --out-dir "${OUT_ROOT}/smpl_npz" \
  --limit "${LIMIT}" \
  --source-fps 20 \
  --target-fps 30 \
  --floor-align \
  --num-smplify-iters "${SMPLIFY_ITERS}" \
  --rot6d-convention column \
  --skip-existing \
  > "${LOGDIR}/smplify.log" 2>&1

python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --src-dir "${OUT_ROOT}/smpl_npz" \
  --out-dir "${OUT_ROOT}/motionclip135" \
  --include-mirrors \
  --key-fallback \
  --align-to-gt-root \
  --overwrite \
  --workers 4 \
  > "${LOGDIR}/remap.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file data/annotation/test_hml3d.json \
  --data_dir data/motionhub \
  --pred_dir "${OUT_ROOT}/motionclip135" \
  --out_json "${OUT_ROOT}/h3d_smplify3d_smoke_c16_rep5.json" \
  --n_repeats 5 \
  --chunk_size 16 \
  --forward_batch_size 16 \
  > "${LOGDIR}/eval.log" 2>&1

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for p in [root / "smpl_npz/_retarget_summary.json", root / "h3d_smplify3d_smoke_c16_rep5.json"]:
    print(p)
    if p.exists():
        d = json.load(open(p))
        for k in ["count", "failed", "mean_mpjpe_mm", "median_mpjpe_mm", "samples", "r_precision_pred_top1_mean", "r_precision_pred_top3_mean", "fid_mean", "mm_dist_pred_mean", "diversity_pred_mean"]:
            if k in d:
                print(k, d[k])
PY

touch "${OUT_ROOT}/_DONE"
echo "[done] $(date)"
