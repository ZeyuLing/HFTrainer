#!/usr/bin/env bash
# Diagnose where Real(SMPL)->HML263->SMPL loses MotionCLIP retrieval quality.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/hml263_bridge_hybrid_diag_0606}
PRED_SMPL=${PRED_SMPL:-outputs/evaluation/gt_hml263_control_row_smpl135_0605/h3d}
if [ ! -d "${PRED_SMPL}" ]; then
  PRED_SMPL=outputs/evaluation/gt_hml263_control_hmlrot_original_c64_0606/smpl_npz/h3d
fi
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${LOGDIR}"

echo "[start] $(date) pred_smpl=${PRED_SMPL}" | tee "${LOGDIR}/run.log"

python3 scripts/eval/make_motionclip135_hybrid_controls.py \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --pred-smpl-dir "${PRED_SMPL}" \
  --out-root "${OUT_ROOT}/motionclip135" \
  --include-mirrors \
  --key-fallback \
  --align-mode yaw \
  > "${LOGDIR}/build_hybrid.log" 2>&1

MODES=(pred_trans_gt_rot gt_trans_pred_rot pred_root_gt_body gt_root_pred_body)
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "GPUS must contain at least one id" >&2
  exit 1
fi

for i in "${!MODES[@]}"; do
  mode="${MODES[$i]}"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  echo "[eval] ${mode} gpu=${gpu} $(date)" | tee -a "${LOGDIR}/run.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub \
    --pred_dir "${OUT_ROOT}/motionclip135/${mode}" \
    --chunk_size 64 \
    --out_json "${OUT_ROOT}/${mode}_h3d_orig_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/eval_${mode}.log" 2>&1 &
done
wait

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for p in sorted(root.glob("*_h3d_orig_c64.json")):
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
