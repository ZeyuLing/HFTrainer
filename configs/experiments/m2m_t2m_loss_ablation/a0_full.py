# Ablation A0 — FULL current M2M loss recipe (baseline of this ablation).
# velocity + component_mean + translation overweighting + motion smoothness
# + KIMODO-style aux (joint_pos / joint_vel / fk_consistency) + keypoints3d.

_base_ = './_base_m2m_t2m_loss_ablation.py'

work_dir = 'work_dirs/m2m_t2m_loss_ablation/a0_full'

model = dict(
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        velocity_loss_reduction='component_mean',
        x1_weight=0.0,
        translation_weight=0.0,
        keypoints3d_weight=10.0,
        motion_smoothness_weight=0.5,
        trans_dim_weight=5.0,
        fk_consistency_weight=0.0,  # legacy path off (aux fk_consistency on)
        aux_joint_pos_weight=50.0,
        aux_joint_vel_weight=500.0,
        aux_fk_consistency_weight=1500.0,
    ),
)
