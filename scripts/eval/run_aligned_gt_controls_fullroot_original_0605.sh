#!/usr/bin/env bash
# Re-map the GT HML263->SMPL control with full first-frame root alignment and
# evaluate with original captions. This isolates root-frame residuals from the
# IK fitting stage.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/aligned_gt_controls_fullroot_original_eval_0605}
H3D_SMPL=${H3D_SMPL:-outputs/evaluation/gt_hml263_control_row_smpl135_0605/h3d}
MH_SMPL=${MH_SMPL:-outputs/evaluation/gt_hml263_control_row_smpl135_0605/mh}
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${LOGDIR}" "${OUT_ROOT}/h3d/motionclip135" "${OUT_ROOT}/mh/motionclip135"

echo "[remap-h3d-fullroot] $(date)" | tee "${LOGDIR}/run.log"
python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --src-dir "${H3D_SMPL}" \
  --out-dir "${OUT_ROOT}/h3d/motionclip135" \
  --include-mirrors \
  --key-fallback \
  --align-to-gt-root \
  --align-root-mode full \
  --overwrite \
  --workers "${REMAP_WORKERS:-16}" \
  > "${LOGDIR}/remap_h3d.log" 2>&1

echo "[remap-mh-fullroot] $(date)" | tee -a "${LOGDIR}/run.log"
python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file data/annotation/test_motionhub_t2m.json \
  --data-dir data/motionhub \
  --src-dir "${MH_SMPL}" \
  --out-dir "${OUT_ROOT}/mh/motionclip135" \
  --include-mirrors \
  --key-fallback \
  --align-to-gt-root \
  --align-root-mode full \
  --overwrite \
  --workers "${REMAP_WORKERS:-16}" \
  > "${LOGDIR}/remap_mh.log" 2>&1

echo "[eval-original] $(date)" | tee -a "${LOGDIR}/run.log"
CUDA_VISIBLE_DEVICES="${H3D_GPU:-0}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file data/annotation/test_hml3d.json \
  --data_dir data/motionhub \
  --pred_dir "${OUT_ROOT}/h3d/motionclip135" \
  --chunk_size 64 \
  --out_json "${OUT_ROOT}/h3d/gt_hml263_fullroot_orig_c64.json" \
  --n_repeats 20 \
  --seed 42 \
  > "${LOGDIR}/eval_h3d.log" 2>&1 &

CUDA_VISIBLE_DEVICES="${MH_GPU:-1}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file data/annotation/test_motionhub_t2m.json \
  --data_dir data/motionhub \
  --pred_dir "${OUT_ROOT}/mh/motionclip135" \
  --chunk_size 64 \
  --out_json "${OUT_ROOT}/mh/gt_hml263_fullroot_orig_c64.json" \
  --n_repeats 20 \
  --seed 42 \
  > "${LOGDIR}/eval_mh.log" 2>&1 &
wait

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for split in ("h3d", "mh"):
    print(f"[{split}]")
    for p in sorted((root / split).glob("*_orig_c64.json")):
        d = json.load(open(p))
        print(
            p.name,
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
