# HyMotion M2M v2 0.46B with T2M Pretrained (Full Freeze)
#
# This config loads T2M pretrained weights and freezes ALL reusable modules.
# Only the reinitialized input_encoder and final_layer are trainable.
#
# Use case: When T2M transfer learning baseline is very strong, this prevents
# catastrophic forgetting of pre-trained features while adapting only the
# VACE-specific components.

_base_ = '_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_046b_t2m_full_freeze'

model = dict(
    t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    t2m_freeze_strategy='full',  # Freeze all loaded modules (encoders, blocks, text_refiner)
)
