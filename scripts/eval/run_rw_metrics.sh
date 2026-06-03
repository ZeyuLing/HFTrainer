#!/bin/bash
# Compute metrics under the CONSISTENT rewritten protocol:
#   generate-on-rewritten (already done) + evaluate-on-rewritten captions.
# Tests the hypothesis that the paper's high H3D R-Precision came from rw+rw.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:${PYTHONPATH:-}

GPU=${GPU:-7}
LOG=outputs/evaluation/_rw_metrics_logs
mkdir -p "$LOG"

run_ds () {
    local ds=$1 kafs=$2 anno=$3 rw=$4 data=$5
    echo "[rw-metrics] >>> $ds start $(date)" | tee -a "$LOG/run.log"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval/compute_kafs_metrics.py \
        --kafs-dir "$kafs" \
        --modes depth_driven \
        --anno-file "$anno" \
        --rewritten-caption-file "$rw" \
        --data-dir "$data" \
        --n-repeats 20 \
        --gpu 0 \
        >"$LOG/${ds}.log" 2>&1
    echo "[rw-metrics] <<< $ds done rc=$? $(date)" | tee -a "$LOG/run.log"
}

run_ds h3d \
    outputs/evaluation/prism_kt_spectral_epoch4_rw/h3d \
    data/annotation/test_hml3d.json \
    data/annotation/test_hml3d_rewritten.json \
    data/motionhub

run_ds mh \
    outputs/evaluation/prism_kt_spectral_epoch4_rw/mh \
    data/annotation/test_motionhub_t2m.json \
    data/annotation/test_motionhub_t2m_rewritten.json \
    data/motionhub

touch "$LOG/_DONE"
echo "[rw-metrics] ALL DONE $(date)" | tee -a "$LOG/run.log"
