#!/usr/bin/env bash
# MotionHub Real(HML3D->SMPL) control with GT-root alignment.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/aligned_gt_hml263_mh_control_0605}
LOGDIR="${OUT_ROOT}/logs"
SRC=${SRC:-outputs/evaluation/gt_hml263_control_row_smpl135_0605/mh}
MC135="${OUT_ROOT}/motionclip135"
mkdir -p "${LOGDIR}" "${MC135}"

echo "[remap] $(date)" | tee "${LOGDIR}/run.log"
python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file data/annotation/test_motionhub_t2m.json \
  --data-dir data/motionhub \
  --src-dir "${SRC}" \
  --out-dir "${MC135}" \
  --include-mirrors \
  --key-fallback \
  --align-to-gt-root \
  --overwrite \
  --workers "${REMAP_WORKERS:-4}" \
  > "${LOGDIR}/remap.log" 2>&1

echo "[eval] $(date)" | tee -a "${LOGDIR}/run.log"
CUDA_VISIBLE_DEVICES="${GPU:-3}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file data/annotation/test_motionhub_t2m.json \
  --data_dir data/motionhub \
  --pred_dir "${MC135}" \
  --rewritten_caption_file data/annotation/test_motionhub_t2m_rewritten.json \
  --chunk_size 64 \
  --out_json "${OUT_ROOT}/motionclip_c64.json" \
  --n_repeats 20 \
  --seed 42 \
  > "${LOGDIR}/eval.log" 2>&1

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
d=json.load(open("${OUT_ROOT}/motionclip_c64.json"))
print(
    "samples", d.get("samples"),
    "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
    "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
    "FID", f"{d.get('fid_mean', float('nan')):.4f}",
    "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
    "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
)
PY
touch "${OUT_ROOT}/_DONE"
echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
