#!/usr/bin/env bash
# Quick MotionGPT prompt smoke eval from already generated predictions.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BASE=${BASE:-outputs/evaluation/motiongpt_prompt_ab_fixed2_0605}
MAX_PAIRS=${MAX_PAIRS:-128}
N_REPEATS=${N_REPEATS:-3}
mkdir -p "${BASE}/quick_logs"

run_eval() {
  local mode="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub \
    --pred_dir "${BASE}/${mode}/motionclip135" \
    --rewritten_caption_file data/annotation/test_hml3d_rewritten.json \
    --chunk_size 64 \
    --max_pairs "${MAX_PAIRS}" \
    --out_json "${BASE}/${mode}/motionclip_h3d_c64_quick.json" \
    --n_repeats "${N_REPEATS}" \
    --seed 43 \
    > "${BASE}/quick_logs/${mode}.log" 2>&1
}

run_eval official_nolen 5 &
run_eval official_len 6 &
run_eval instruction 7 &
run_eval direct 0 &
wait

BASE_PATH="${BASE}" python3 - <<'PY' | tee "${BASE}/quick_summary.txt"
import json
import os
from pathlib import Path

base = Path(os.environ["BASE_PATH"])
for mode in ("official_nolen", "official_len", "instruction", "direct"):
    p = base / mode / "motionclip_h3d_c64_quick.json"
    if not p.exists():
        print(mode, "missing")
        continue
    d = json.load(open(p))
    print(
        mode,
        "samples", d.get("samples"),
        "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
        "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
        "FID", f"{d.get('fid_mean', float('nan')):.4f}",
        "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
        "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
    )
PY
