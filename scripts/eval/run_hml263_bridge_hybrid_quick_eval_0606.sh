#!/usr/bin/env bash
# Fast 512-sample MotionCLIP eval for HML263 bridge hybrid controls.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/hml263_bridge_hybrid_diag5_0606}
MAX_PAIRS=${MAX_PAIRS:-512}
N_REPEATS=${N_REPEATS:-20}
LOGDIR="${OUT_ROOT}/logs_quick"
mkdir -p "${LOGDIR}"

MODES=(pred_trans_gt_rot gt_trans_pred_rot pred_root_gt_body gt_root_pred_body)
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3}"

for i in "${!MODES[@]}"; do
  mode="${MODES[$i]}"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub \
    --pred_dir "${OUT_ROOT}/motionclip135/${mode}" \
    --chunk_size 64 \
    --max_pairs "${MAX_PAIRS}" \
    --out_json "${OUT_ROOT}/${mode}_h3d_orig_c64_m${MAX_PAIRS}_r${N_REPEATS}.json" \
    --n_repeats "${N_REPEATS}" \
    --seed 42 \
    > "${LOGDIR}/eval_${mode}_m${MAX_PAIRS}.log" 2>&1 &
done
wait

python3 - <<PY | tee "${OUT_ROOT}/summary_quick_m${MAX_PAIRS}_r${N_REPEATS}.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for p in sorted(root.glob("*_h3d_orig_c64_m${MAX_PAIRS}_r${N_REPEATS}.json")):
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
