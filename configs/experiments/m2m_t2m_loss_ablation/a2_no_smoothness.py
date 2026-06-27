# Ablation A2 — a0_full MINUS motion smoothness.
# Isolates the effect of motion_smoothness_weight (suspected to suppress
# high-frequency detail that FID / diversity reward).

_base_ = './_base_m2m_t2m_loss_ablation.py'

work_dir = 'work_dirs/m2m_t2m_loss_ablation/a2_no_smoothness'

model = dict(
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        velocity_loss_reduction='component_mean',
        x1_weight=0.0,
        translation_weight=0.0,
        keypoints3d_weight=10.0,
        motion_smoothness_weight=0.0,  # <-- ablated
        trans_dim_weight=5.0,
        fk_consistency_weight=0.0,
        aux_joint_pos_weight=50.0,
        aux_joint_vel_weight=500.0,
        aux_fk_consistency_weight=1500.0,
    ),
)
