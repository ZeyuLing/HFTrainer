#!/bin/bash
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUTPUT_DIR=work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten
LOG_FILE=work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten.log

python3 scripts/eval/eval_prism_t2m_hml3d.py \
    --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0 \
    --output-dir $OUTPUT_DIR \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --gpus 0 1 2 3 4 5 6 7 \
    2>&1 | tee $LOG_FILE
