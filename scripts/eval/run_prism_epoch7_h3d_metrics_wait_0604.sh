#!/bin/bash
# Wait for PRISM epoch-7 HumanML3D generation to reach the expected evaluation
# count, then run MotionCLIP and MotionStreamer evaluators in parallel.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

OLD_DIR=${OLD_DIR:-outputs/evaluation/prism_kt_spectral_epoch7_rw/h3d/depth_driven}
FAST_DIR=${FAST_DIR:-outputs/evaluation/prism_kt_spectral_epoch7_rw_fast0604/h3d/depth_driven}
MERGED_ROOT=${MERGED_ROOT:-outputs/evaluation/prism_kt_spectral_epoch7_rw_merged0604/h3d}
MODE=depth_driven
MERGED_DIR="$MERGED_ROOT/$MODE"
LOGDIR="$MERGED_ROOT/_metrics_logs"
MS_OUT=${MS_OUT:-outputs/evaluation/motionstreamer_272_epoch7fast}
MS_REPACK="$MS_OUT/prism_kt_spectral_epoch7_h3d_depth_driven"
MS_JSON="$MS_OUT/metrics/prism_kt_spectral_epoch7_h3d_depth_driven.json"
TARGET=${TARGET:-4269}
MAX_WAIT=${MAX_WAIT:-10800}
SLEEP_SEC=${SLEEP_SEC:-60}

mkdir -p "$MERGED_DIR" "$LOGDIR" "$MS_OUT/metrics"

merge_once () {
    python3 - <<'PY'
from pathlib import Path
import os
old_dir = Path(os.environ["OLD_DIR"])
fast_dir = Path(os.environ["FAST_DIR"])
merged = Path(os.environ["MERGED_DIR"])
merged.mkdir(parents=True, exist_ok=True)
linked = 0
for srcdir in (old_dir, fast_dir):
    if not srcdir.exists():
        continue
    for src in srcdir.glob("*.npz"):
        dst = merged / src.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(os.path.relpath(src, dst.parent), dst)
        linked += 1
print(f"linked={linked} merged={len(list(merged.glob('*.npz')))}", flush=True)
PY
}

start_ts=$(date +%s)
echo "[wait] start $(date) target=$TARGET merged=$MERGED_DIR" | tee -a "$LOGDIR/wait.log"
while true; do
    export OLD_DIR FAST_DIR MERGED_DIR
    merge_once | tee -a "$LOGDIR/wait.log"
    count=$(find "$MERGED_DIR" -maxdepth 1 \( -type l -o -type f \) -name '*.npz' | wc -l)
    old_count=$(find "$OLD_DIR" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
    fast_count=$(find "$FAST_DIR" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
    echo "[wait] $(date) old=$old_count fast=$fast_count merged=$count" | tee -a "$LOGDIR/wait.log"
    if [ "$count" -ge "$TARGET" ]; then
        break
    fi
    now=$(date +%s)
    if [ $((now - start_ts)) -ge "$MAX_WAIT" ]; then
        echo "[wait] timeout with merged=$count target=$TARGET" | tee -a "$LOGDIR/wait.log"
        exit 2
    fi
    sleep "$SLEEP_SEC"
done

echo "[eval] launch $(date)" | tee -a "$LOGDIR/wait.log"

(
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/compute_kafs_metrics.py \
        --kafs-dir "$MERGED_ROOT" \
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
        --npz-dir "$MERGED_DIR" \
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
mc = Path("outputs/evaluation/prism_kt_spectral_epoch7_rw_merged0604/h3d/metrics_depth_driven.json")
ms = Path("outputs/evaluation/motionstreamer_272_epoch7fast/metrics/prism_kt_spectral_epoch7_h3d_depth_driven.json")
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
echo "[done] $(date)" | tee -a "$LOGDIR/wait.log"
