# HyMotion M2M 0.46B — Completion (unconditioned) with flow matching (pred_type='velocity').
#
# Unconditioned version: no text encoder, null text embeddings throughout.
# The model learns pure motion completion conditioned only on src_mask + src_motion.
#
# Flow matching: x_t = (1-t)*x0 + t*x1, predict v = x1 - x0.
# Timesteps: uniform U[0, 1].
# Loss: SmoothL1(pred_velocity, gt_velocity).
#
# Dataset: MotionHub with universal mask (M1-M6 strategies).
# Note: task_mode='auto' is used but LoadCompatibleCaption is absent from pipeline,
# so caption is never loaded. No need for caption-based task filtering.
#
# Suitable mask patterns for unconditioned training:
#   - M1 (random_cell): good — mask pattern itself provides spatial structure
#   - M2 (random_block): good
#   - M3 (temporal_contiguous): good — in-between/prediction/prefix all valid
#   - M4 (joint_contiguous): good — upper/lower body editing
#   - M5 (full_mask): good — pure motion generation from noise, this is the
#       canonical unconditioned case; weight kept low (5%) to avoid ignoring src_mask
#   - M6 (keyframe_sparse): good — spatial/temporal anchors, no text needed
#
# Excluded patterns (not applicable to unconditioned):
#   - Text-guided variants (F2 / t2m) are absent here since text_encoder=None.
#     The model treats all samples as motion-conditioned completion.
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_046b.py 8
#   bash tools/taiji_dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_046b.py

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_uncond_fm_046b'

model = dict(
    pred_type='velocity',
    uncondition_mode=True,
    text_encoder=None,
    cond_mask_prob=0.0,   # No text conditioning → no CFG dropout needed
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
    ),
)
