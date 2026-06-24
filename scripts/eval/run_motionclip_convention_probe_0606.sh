#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

OUT_ROOT="${OUT_ROOT:-outputs/evaluation/motionclip_convention_probe_0606}"
MAX_PAIRS="${MAX_PAIRS:-512}"
N_REPEATS="${N_REPEATS:-3}"
mkdir -p "${OUT_ROOT}"

run_one() {
  local convention="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub \
    --gt_only \
    --max_pairs "${MAX_PAIRS}" \
    --chunk_size 64 \
    --n_repeats "${N_REPEATS}" \
    --seed 42 \
    --rot6d_convention "${convention}" \
    --out_json "${OUT_ROOT}/gt_${convention}_${MAX_PAIRS}.json" \
    > "${OUT_ROOT}/gt_${convention}_${MAX_PAIRS}.log" 2>&1
}

run_one row 0 &
run_one column 1 &
wait

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("OUT_ROOT", "outputs/evaluation/motionclip_convention_probe_0606"))
max_pairs = os.environ.get("MAX_PAIRS", "512")
for convention in ("row", "column"):
    path = root / f"gt_{convention}_{max_pairs}.json"
    if not path.exists():
        print(f"{convention}: missing {path}")
        continue
    d = json.load(open(path))
    print(
        convention,
        "samples", d.get("samples"),
        "R1", f"{d.get('r_precision_real_top1_mean', float('nan')):.4f}",
        "R3", f"{d.get('r_precision_real_top3_mean', float('nan')):.4f}",
        "FID", f"{d.get('fid_mean', float('nan')):.4f}",
        "MM", f"{d.get('mm_dist_real_mean', float('nan')):.4f}",
        "Div", f"{d.get('diversity_real_mean', float('nan')):.4f}",
    )
PY
