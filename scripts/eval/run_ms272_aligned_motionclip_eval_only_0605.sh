#!/usr/bin/env bash
# Sequential MotionCLIP evaluation for already-converted aligned 272 baselines.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/third_party:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/ms272_t2m_eval0605_aligned}
LOGDIR="${OUT_ROOT}/logs_mc_seq"
mkdir -p "${LOGDIR}"

run_mc_eval() {
  local tag="$1"
  local gpu="$2"
  echo "[mc-eval] ${tag} gpu=${gpu} $(date)" | tee -a "${LOGDIR}/run.log"
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
    > "${LOGDIR}/mc_${tag}.log" 2>&1
}

echo "[start] $(date)" | tee "${LOGDIR}/run.log"
run_mc_eval gt_ms272 0
run_mc_eval flowmdm_smpl272 1
run_mc_eval motionlab_demo201_smpl272 2

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for tag in ["gt_ms272", "flowmdm_smpl272", "motionlab_demo201_smpl272"]:
    print("\\n" + tag)
    mc = root / tag / "motionclip_c64.json"
    if mc.exists():
        d = json.load(open(mc))
        print(
            " MotionCLIP",
            "samples", d.get("samples"),
            "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
            "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
            "FID", f"{d.get('fid_mean', float('nan')):.4f}",
            "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
            "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
        )
    ms = root / tag / "motionstreamer272.json"
    if ms.exists():
        d = json.load(open(ms))
        pred = d.get("pred", {})
        print(
            " MS272",
            "ids", d.get("ids_with_required_files"),
            "R1", f"{pred.get('r_precision', [float('nan')])[0]:.4f}",
            "R3", f"{pred.get('r_precision', [0, 0, float('nan')])[2]:.4f}",
            "FID", f"{pred.get('fid_vs_gt_native', float('nan')):.4f}",
            "MM", f"{pred.get('matching_score', float('nan')):.4f}",
            "Div", f"{pred.get('diversity', float('nan')):.4f}",
        )
PY
touch "${OUT_ROOT}/_DONE"
echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
