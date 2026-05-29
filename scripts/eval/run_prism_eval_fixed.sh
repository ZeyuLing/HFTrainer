#!/bin/bash
# Launch PRISM T2M HML3D evaluation with first-frame fix
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH

python3 scripts/eval/eval_prism_t2m_hml3d.py \
    --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0 \
    --output-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --gpus 0 1 2 3 4 5 6 7
