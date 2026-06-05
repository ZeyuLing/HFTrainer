#!/usr/bin/env bash
# Validate whether restoring the arbitrary GT root frame removes the HML/272 control gap.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/aligned_gt_controls_0605}
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${LOGDIR}"

GT272_SRC=${GT272_SRC:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data}
GT_HML_SMPL_SRC=${GT_HML_SMPL_SRC:-outputs/evaluation/gt_hml263_control_row_smpl135_0605/h3d}

echo "[start] $(date)" | tee "${LOGDIR}/run.log"

echo "[convert gt_ms272_aligned] $(date)" | tee -a "${LOGDIR}/run.log"
python3 scripts/eval/convert_ms272_dir_for_t2m_eval.py \
  --src-dir "${GT272_SRC}" \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --motionclip-dir "${OUT_ROOT}/gt_ms272_aligned/motionclip135" \
  --only-mapped \
  --align-to-gt-root \
  --workers "${CONVERT_WORKERS:-16}" \
  --overwrite \
  > "${LOGDIR}/convert_gt_ms272_aligned.log" 2>&1

echo "[remap gt_hml263_aligned] $(date)" | tee -a "${LOGDIR}/run.log"
python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --src-dir "${GT_HML_SMPL_SRC}" \
  --out-dir "${OUT_ROOT}/gt_hml263_aligned/motionclip135" \
  --include-mirrors \
  --key-fallback \
  --align-to-gt-root \
  --overwrite \
  --workers "${CONVERT_WORKERS:-16}" \
  > "${LOGDIR}/remap_gt_hml263_aligned.log" 2>&1

run_mc_eval() {
  local tag="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub \
    --pred_dir "${OUT_ROOT}/${tag}/motionclip135" \
    --rewritten_caption_file data/annotation/test_hml3d_rewritten.json \
    --chunk_size 64 \
    --out_json "${OUT_ROOT}/${tag}/motionclip_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/eval_${tag}.log" 2>&1
}

echo "[eval] $(date)" | tee -a "${LOGDIR}/run.log"
run_mc_eval gt_ms272_aligned 0 &
run_mc_eval gt_hml263_aligned 1 &
wait

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for tag in ["gt_ms272_aligned", "gt_hml263_aligned"]:
    p = root / tag / "motionclip_c64.json"
    print("\\n" + tag)
    if not p.exists():
        print("missing", p)
        continue
    d = json.load(open(p))
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
