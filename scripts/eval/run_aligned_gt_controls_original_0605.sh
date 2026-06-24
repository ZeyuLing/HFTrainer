#!/usr/bin/env bash
# Re-evaluate aligned GT controls with the original captions. This isolates
# the representation/retargeting loss from rewritten-caption retrieval effects.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SRC_H3D_ROOT=${SRC_H3D_ROOT:-outputs/evaluation/aligned_gt_controls_0605}
SRC_MH_ROOT=${SRC_MH_ROOT:-outputs/evaluation/aligned_gt_hml263_mh_control_0605}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/aligned_gt_controls_original_eval_0605}
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${LOGDIR}" "${OUT_ROOT}/h3d" "${OUT_ROOT}/mh"

run_eval() {
  local tag="$1"
  local anno="$2"
  local pred="$3"
  local out_json="$4"
  local gpu="$5"
  echo "[eval-original] ${tag} gpu=${gpu} $(date)" | tee -a "${LOGDIR}/run.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${pred}" \
    --chunk_size 64 \
    --out_json "${out_json}" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/eval_${tag}.log" 2>&1
}

echo "[start] $(date)" | tee "${LOGDIR}/run.log"
run_eval h3d_gt_ms272 data/annotation/test_hml3d.json \
  "${SRC_H3D_ROOT}/gt_ms272_aligned/motionclip135" \
  "${OUT_ROOT}/h3d/gt_ms272_aligned_orig_c64.json" 0 &
run_eval h3d_gt_hml263 data/annotation/test_hml3d.json \
  "${SRC_H3D_ROOT}/gt_hml263_aligned/motionclip135" \
  "${OUT_ROOT}/h3d/gt_hml263_aligned_orig_c64.json" 1 &
run_eval mh_gt_hml263 data/annotation/test_motionhub_t2m.json \
  "${SRC_MH_ROOT}/motionclip135" \
  "${OUT_ROOT}/mh/gt_hml263_aligned_orig_c64.json" 2 &
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
