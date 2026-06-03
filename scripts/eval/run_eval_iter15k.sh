#!/bin/bash
# Evaluate the iter_15000 (paper "ours") generations under the PAPER protocol:
#   convert NPZ->135d, then R-P/FID/MM-Dist with rewritten captions + pool 64.
# Compare to paper "ours": H3D T1/T3=0.699/0.893 FID0.027 ; MH T1/T3=0.530/0.772 FID0.055.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:${PYTHONPATH:-}
GPU=${GPU:-0}
ROOT=outputs/evaluation/prism_paper_iter15000
LOG=$ROOT/_eval_logs
mkdir -p "$LOG"

run_ds () {
    local ds=$1 anno=$2 rw=$3
    echo "[eval15k] >>> $ds start $(date)" | tee -a "$LOG/run.log"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval/compute_kafs_metrics.py \
        --kafs-dir "$ROOT/$ds" \
        --modes none \
        --anno-file "$anno" \
        --rewritten-caption-file "$rw" \
        --data-dir data/motionhub \
        --chunk-size 64 \
        --n-repeats 20 \
        --gpu 0 \
        >"$LOG/${ds}.log" 2>&1
    echo "[eval15k] <<< $ds done rc=$? $(date)" | tee -a "$LOG/run.log"
}

run_ds h3d data/annotation/test_hml3d.json         data/annotation/test_hml3d_rewritten.json
run_ds mh  data/annotation/test_motionhub_t2m.json data/annotation/test_motionhub_t2m_rewritten.json

touch "$LOG/_DONE"
echo "[eval15k] ALL DONE $(date)" | tee -a "$LOG/run.log"
