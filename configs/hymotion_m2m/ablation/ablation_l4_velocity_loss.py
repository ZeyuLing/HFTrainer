# Ablation L4: Velocity Smoothness Loss
# 验证：加入 motion velocity smoothness loss 是否能减少 jitter。
# KIMODO 将 velocity 作为表示的一部分并在 loss 中加权（γ_vel=2）。
# 我方在不改表示的前提下，通过 loss 中的帧差分约束实现类似效果。
#
# 改动：motion_smoothness_weight: 0 → 0.5

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_l4_velocity_loss'

train_cfg = dict(max_epochs=20)

model = dict(
    losses_cfg=dict(
        motion_smoothness_weight=0.5,
    ),
)
