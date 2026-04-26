# Ablation L3b: Translation Weight = 10
# 验证：提高 translation 维度权重的效果（接近 KIMODO γ_pos=10）。
#
# 改动：trans_dim_weight: 5 → 10

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_l3b_trans_w10'

train_cfg = dict(max_epochs=20)

model = dict(
    losses_cfg=dict(
        trans_dim_weight=10.0,
    ),
)
