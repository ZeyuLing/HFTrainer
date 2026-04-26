# HyMotion M2M 0.46B — Completion (unconditioned) with JiT loss (pred_type='x1').
#
# Unconditioned version: no text encoder, null text embeddings throughout.
# The model learns pure motion completion conditioned only on src_mask + src_motion.
#
# JiT (Jump-in-Time): x_t = (1-t)*x0 + t*x1, predict x1 directly.
# Timesteps: sigmoid(z), z ~ N(-0.8, 0.8²).
# Loss: velocity reparameterized as (pred_x1 - x_t)/(1-t) + direct x1 reconstruction.
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_046b.py 8
#   bash tools/taiji_dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_046b.py

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_uncond_jit_046b'

model = dict(
    pred_type='x1',
    uncondition_mode=True,
    text_encoder=None,
    cond_mask_prob=0.0,   # No text conditioning → no CFG dropout needed
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,   # reparameterized velocity: (pred_x1 - x_t) / (1-t)
        x1_weight=1.0,         # direct x1 reconstruction
        keypoints3d_weight=0.0,
        translation_weight=0.0,
    ),
)
