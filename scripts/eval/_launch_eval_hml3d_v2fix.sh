#!/bin/bash
# Launch PRISM HML3D evaluation with v2 first-frame fix (conditional)
# Run this script on a GPU machine with 8 V100s
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH

LOG=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_v2fix.log

echo "=== Starting eval at $(date) ===" > $LOG
echo "Fix: conditional fix_first_chunk (only first segment)" >> $LOG

python3 scripts/eval/eval_prism_t2m_hml3d.py \
    --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0 \
    --output-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --gpus 0 1 2 3 4 5 6 7 \
    >> $LOG 2>&1

echo "=== Eval finished at $(date) with exit code $? ===" >> $LOG
