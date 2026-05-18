# HyMotion M2M v2 0.46B with T2M Pretrained (No Freeze)
#
# This config loads T2M pretrained weights but leaves ALL modules trainable.
# This serves as a baseline to measure whether pretraining helps over random init.
#
# Use case: Ablation study to validate the transfer learning benefit.

_base_ = '_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_046b_t2m_no_freeze'

model = dict(
    t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    t2m_freeze_strategy='none',  # No freezing - all modules remain trainable
)
