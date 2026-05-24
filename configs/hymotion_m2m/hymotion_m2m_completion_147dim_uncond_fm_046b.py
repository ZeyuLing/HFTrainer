# HyMotion M2M 147-dim Completion (Unconditional, Flow Matching)
# 
# This config extends the 147-dim base with specific training settings.
# Start from _base_hymotion_m2m_147dim_046b which includes:
#  - 147-dim motion representation (135-dim + 12-dim end-effector positions)
#  - Compute147DimEndEffector transform in data pipeline
#  - Correct VACE input dims: 588 (motion + 3×motion)
#  - Output dims: 147
#  - Normalization stats: data/hymotion_m2m_data/_stats_147dim

_base_ = '_base_hymotion_m2m_147dim_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_147dim_uncond_fm_046b'

# Unconditional (no text conditioning)
model = dict(
    uncondition_mode=True,
    cond_mask_prob=0.0,
)

trainer = dict(
    type='HyMotionM2MTrainer',
    val_num_steps=10,
    mask_aware_noise=False,
)
