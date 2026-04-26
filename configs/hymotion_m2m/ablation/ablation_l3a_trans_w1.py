# Ablation L3a: Translation Weight = 1
# 验证：降低 translation 维度权重的效果。
# Baseline trans_dim_weight=5.0; KIMODO γ_pos=10。
#
# 改动：trans_dim_weight: 5 → 1

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_l3a_trans_w1'

train_cfg = dict(max_epochs=20)

model = dict(
    losses_cfg=dict(
        trans_dim_weight=1.0,
    ),
)
