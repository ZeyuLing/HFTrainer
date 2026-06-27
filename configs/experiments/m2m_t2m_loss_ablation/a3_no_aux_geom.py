# Ablation A3 — a0_full MINUS all geometry/position-space supervision
# (joint_pos / joint_vel / fk_consistency + keypoints3d). Keeps velocity,
# smoothness and translation weighting. Isolates the FK-derived metric-space
# aux losses.

_base_ = './_base_m2m_t2m_loss_ablation.py'

work_dir = 'work_dirs/m2m_t2m_loss_ablation/a3_no_aux_geom'

model = dict(
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        velocity_loss_reduction='component_mean',
        x1_weight=0.0,
        translation_weight=0.0,
        keypoints3d_weight=0.0,        # <-- ablated
        motion_smoothness_weight=0.5,
        trans_dim_weight=5.0,
        fk_consistency_weight=0.0,
        aux_joint_pos_weight=0.0,      # <-- ablated
        aux_joint_vel_weight=0.0,      # <-- ablated
        aux_fk_consistency_weight=0.0,  # <-- ablated
    ),
)
