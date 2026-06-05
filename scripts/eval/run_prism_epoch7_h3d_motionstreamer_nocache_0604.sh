#!/bin/bash
# MotionStreamer-272 evaluation for PRISM epoch-7 H3D, without /dev/shm cache.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

PRED_DIR=${PRED_DIR:-outputs/evaluation/prism_kt_spectral_epoch7_rw/h3d/depth_driven}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/motionstreamer_272_epoch7_actual0604_nocache}
REPACK_DIR="$OUT_ROOT/prism_kt_spectral_epoch7_h3d_depth_driven"
JSON="$OUT_ROOT/metrics/prism_kt_spectral_epoch7_h3d_depth_driven.json"
LOGDIR="$OUT_ROOT/logs"

mkdir -p "$OUT_ROOT/metrics" "$LOGDIR"
rm -rf /dev/shm/ms272_data

echo "[start] $(date) pred_npz=$(find "$PRED_DIR" -maxdepth 1 -name '*.npz' | wc -l)" | tee "$LOGDIR/run.log"
python3 scripts/eval/repack_pred_to_272ids.py \
    --npz-dir "$PRED_DIR" \
    --anno-file data/annotation/test_hml3d.json \
    --out-dir "$REPACK_DIR" \
    --workers 32 \
    > "$LOGDIR/repack.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$REPACK_DIR" \
    --tag prism_kt_spectral_epoch7_h3d_depth_driven \
    --out-json "$JSON" \
    --device cuda \
    > "$LOGDIR/eval.log" 2>&1

python3 - <<'PY' | tee "$LOGDIR/summary.txt"
import json
from pathlib import Path
p = Path("outputs/evaluation/motionstreamer_272_epoch7_actual0604_nocache/metrics/prism_kt_spectral_epoch7_h3d_depth_driven.json")
d = json.load(open(p))
print({"gt_real": d.get("gt_real"), "pred": d.get("pred")})
PY
touch "$LOGDIR/_DONE"
echo "[done] $(date)" | tee -a "$LOGDIR/run.log"
