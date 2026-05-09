_base_ = '../hymotion_m2m_v2_caption_global_phase2.py'

model = dict(
    losses_cfg=dict(
        velocity_loss_reduction='component_mean',
        trans_dim_weight=1.0,
    ),
)
