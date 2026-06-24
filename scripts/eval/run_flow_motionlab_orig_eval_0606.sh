#!/usr/bin/env bash
# Evaluate FlowMDM and MotionLab predictions with the original annotations.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

LOGDIR="outputs/evaluation/flow_motionlab_orig_eval0606/logs"
mkdir -p \
  "${LOGDIR}" \
  outputs/evaluation/flowmdm_officialstats_0606/motionclip_eval_orig_c64/h3d \
  outputs/evaluation/flowmdm_officialstats_0606/motionclip_eval_orig_c64/mh \
  outputs/evaluation/motionlab_fixed0606/motionclip_eval_orig_c64/h3d \
  outputs/evaluation/motionlab_fixed0606/motionclip_eval_orig_c64/mh

run_eval() {
  local gpu="$1"
  local anno="$2"
  local pred="$3"
  local out="$4"
  local log="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${pred}" \
    --chunk_size 64 \
    --out_json "${out}" \
    --n_repeats 20 \
    --seed 42 \
    > "${log}" 2>&1
}

run_eval 0 data/annotation/test_hml3d.json \
  outputs/evaluation/flowmdm_officialstats_0606/motionclip135_rw_c64/h3d \
  outputs/evaluation/flowmdm_officialstats_0606/motionclip_eval_orig_c64/h3d/flowmdm_officialstats_orig_c64.json \
  "${LOGDIR}/flow_h3d.log" &

run_eval 1 data/annotation/test_motionhub_t2m.json \
  outputs/evaluation/flowmdm_officialstats_0606/motionclip135_rw_c64/mh \
  outputs/evaluation/flowmdm_officialstats_0606/motionclip_eval_orig_c64/mh/flowmdm_officialstats_orig_c64.json \
  "${LOGDIR}/flow_mh.log" &

run_eval 3 data/annotation/test_hml3d.json \
  outputs/evaluation/motionlab_fixed0606/motionclip135_rw_c64/h3d \
  outputs/evaluation/motionlab_fixed0606/motionclip_eval_orig_c64/h3d/motionlab_fixed0606_orig_c64.json \
  "${LOGDIR}/motionlab_h3d.log" &

run_eval 4 data/annotation/test_motionhub_t2m.json \
  outputs/evaluation/motionlab_fixed0606/motionclip135_rw_c64/mh \
  outputs/evaluation/motionlab_fixed0606/motionclip_eval_orig_c64/mh/motionlab_fixed0606_orig_c64.json \
  "${LOGDIR}/motionlab_mh.log" &

wait

python3 - <<'PY' | tee outputs/evaluation/flow_motionlab_orig_eval0606/summary.txt
import json
from pathlib import Path

items = [
    ("flow_h3d", "outputs/evaluation/flowmdm_officialstats_0606/motionclip_eval_orig_c64/h3d/flowmdm_officialstats_orig_c64.json"),
    ("flow_mh", "outputs/evaluation/flowmdm_officialstats_0606/motionclip_eval_orig_c64/mh/flowmdm_officialstats_orig_c64.json"),
    ("motionlab_h3d", "outputs/evaluation/motionlab_fixed0606/motionclip_eval_orig_c64/h3d/motionlab_fixed0606_orig_c64.json"),
    ("motionlab_mh", "outputs/evaluation/motionlab_fixed0606/motionclip_eval_orig_c64/mh/motionlab_fixed0606_orig_c64.json"),
]
for name, path in items:
    p = Path(path)
    if not p.exists():
        print(name, "MISSING", path)
        continue
    d = json.load(open(p))
    print(
        name,
        "samples", d.get("samples"),
        "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
        "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
        "FID", f"{d.get('fid_mean', float('nan')):.4f}",
        "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
        "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
    )
PY

touch outputs/evaluation/flow_motionlab_orig_eval0606/_DONE
