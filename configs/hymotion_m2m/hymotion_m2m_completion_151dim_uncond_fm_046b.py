# HyMotion M2M 0.46B training config — 151-dim unconditional (text-free)
# Motion representation: smpl_22 with rotation_6d + end-effector positions + foot contact
# Input: 151 + 3*151 = 604 dims (motion + VACE context)
# Output: 151 dims

_base_ = '_base_hymotion_m2m_151dim_046b.py'

# Work directory
work_dir = 'work_dirs/hymotion_m2m_completion_151dim_uncond_fm_046b'

# Train configuration
train_cfg = dict(
    by_epoch=True,
    max_epochs=1000,
    val_interval=10,
    max_grad_norm=1.0,
)
