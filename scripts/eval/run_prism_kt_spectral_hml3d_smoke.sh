#!/bin/bash
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
mkdir -p work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_epoch0
export CUDA_VISIBLE_DEVICES=0
python scripts/eval/eval_prism_kafs_ablation.py \
    --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0 \
    --kafs-mode none \
    --anno-file data/annotation/test_hml3d.json \
    --data-dir data/motionhub \
    --output-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_epoch0 \
    --max-samples 50 \
    --num-inference-steps 50 \
    --guidance-scale 5.0
