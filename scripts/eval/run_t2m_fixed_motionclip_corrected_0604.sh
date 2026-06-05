#!/bin/bash
# Correct MotionCLIP evaluation for fixed HML263 baselines.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/t2m_fixed_eval0604_motionclip_corrected}
SMPL_ROOT=${SMPL_ROOT:-outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604}
MC135_ROOT=${MC135_ROOT:-outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604_motionclip135}
LOGDIR=$EVAL_ROOT/logs
OUT=$EVAL_ROOT/motionclip_metrics
REMAP_WORKERS=${REMAP_WORKERS:-8}
mkdir -p "$LOGDIR" "$OUT" "$MC135_ROOT"

echo "[setup] start $(date) smpl=$SMPL_ROOT mc135=$MC135_ROOT" | tee -a "$LOGDIR/run.log"
run_remap () {
    local method=$1
    python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
        --src-dir "$SMPL_ROOT/$method" \
        --out-dir "$MC135_ROOT/$method" \
        --overwrite \
        --workers "$REMAP_WORKERS" \
        > "$LOGDIR/remap_${method}.log" 2>&1
}

for method in momask mdm_fixed motiongpt3_fixed mld_v1_rootfix; do
    run_remap "$method" &
done
wait

run_mc_eval () {
    local method=$1 gpu=$2
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/eval_with_motionclip_evaluator.py \
        --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
        --anno_file data/annotation/test_hml3d.json \
        --data_dir data/motionhub \
        --pred_dir "$MC135_ROOT/$method" \
        --chunk_size 64 \
        --out_json "$OUT/${method}_orig_c64.json" \
        --n_repeats 20 \
        --seed 42 \
        > "$LOGDIR/mc_${method}_orig_c64.log" 2>&1
}

echo "[eval] launch $(date)" | tee -a "$LOGDIR/run.log"
run_mc_eval momask 0 &
run_mc_eval mdm_fixed 1 &
run_mc_eval motiongpt3_fixed 2 &
run_mc_eval mld_v1_rootfix 3 &
wait

EVAL_ROOT="$EVAL_ROOT" python3 - <<'PY' | tee "$EVAL_ROOT/summary.txt"
import json
import os
from pathlib import Path
root = Path(os.environ["EVAL_ROOT"])
print("[summary]")
for p in sorted((root / "motionclip_metrics").glob("*.json")):
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
touch "$EVAL_ROOT/_DONE"
echo "[done] $(date)" | tee -a "$LOGDIR/run.log"
