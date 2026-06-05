#!/bin/bash
# Evaluate PRISM epoch-7 HumanML3D outputs from the real output directory
# (no merged symlinks) with MotionCLIP and MotionStreamer evaluators.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

ROOT=${ROOT:-outputs/evaluation/prism_kt_spectral_epoch7_rw/h3d}
MODE=depth_driven
PRED_DIR="$ROOT/$MODE"
LOGDIR="$ROOT/_metrics_logs_actual0604"
MS_OUT=${MS_OUT:-outputs/evaluation/motionstreamer_272_epoch7_actual0604}
MS_REPACK="$MS_OUT/prism_kt_spectral_epoch7_h3d_depth_driven"
MS_JSON="$MS_OUT/metrics/prism_kt_spectral_epoch7_h3d_depth_driven.json"

mkdir -p "$LOGDIR" "$MS_OUT/metrics"
echo "[start] $(date) root=$ROOT npz=$(find "$PRED_DIR" -maxdepth 1 -name '*.npz' | wc -l)" | tee "$LOGDIR/run.log"

(
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/compute_kafs_metrics.py \
        --kafs-dir "$ROOT" \
        --modes "$MODE" \
        --anno-file data/annotation/test_hml3d.json \
        --rewritten-caption-file data/annotation/test_hml3d_rewritten.json \
        --data-dir data/motionhub \
        --evaluator-ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
        --chunk-size 64 \
        --n-repeats 20 \
        --workers 32 \
        --gpu 0 \
        > "$LOGDIR/motionclip_h3d.log" 2>&1
) &
pids=($!)

(
    bash scripts/eval/_cache_272_data.sh > "$LOGDIR/ms_cache.log" 2>&1
    python3 scripts/eval/repack_pred_to_272ids.py \
        --npz-dir "$PRED_DIR" \
        --anno-file data/annotation/test_hml3d.json \
        --out-dir "$MS_REPACK" \
        --workers 32 \
        > "$LOGDIR/ms_repack.log" 2>&1
    CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/eval_motionstreamer_272.py \
        --pred-dir "$MS_REPACK" \
        --tag prism_kt_spectral_epoch7_h3d_depth_driven \
        --out-json "$MS_JSON" \
        --device cuda \
        > "$LOGDIR/motionstreamer_h3d.log" 2>&1
) &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

python3 - <<'PY' | tee "$LOGDIR/summary.txt"
import json
from pathlib import Path
mc = Path("outputs/evaluation/prism_kt_spectral_epoch7_rw/h3d/metrics_depth_driven.json")
ms = Path("outputs/evaluation/motionstreamer_272_epoch7_actual0604/metrics/prism_kt_spectral_epoch7_h3d_depth_driven.json")
if mc.exists():
    d = json.load(open(mc))
    print("[motionclip]", {k: d.get(k) for k in [
        "samples", "r_precision_pred_top1_mean", "r_precision_pred_top3_mean",
        "fid_mean", "mm_dist_pred_mean", "diversity_pred_mean"]})
if ms.exists():
    d = json.load(open(ms))
    pred = d.get("pred", {})
    print("[motionstreamer]", {
        "r_precision": pred.get("r_precision"),
        "fid_vs_gt_native": pred.get("fid_vs_gt_native"),
        "matching_score": pred.get("matching_score"),
        "diversity": pred.get("diversity"),
        "nb": pred.get("nb"),
    })
PY
touch "$LOGDIR/_DONE"
echo "[done] $(date)" | tee -a "$LOGDIR/run.log"
