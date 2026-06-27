# Ablation A1 — VELOCITY ONLY (== HYMotion T2M objective, on 198-dim target).
# All auxiliary / smoothness / translation-weighting terms disabled. If this
# beats a0_full on T2M metrics at equal budget, the extra losses are negative
# optimization for pure text-to-motion.

_base_ = './_base_m2m_t2m_loss_ablation.py'

work_dir = 'work_dirs/m2m_t2m_loss_ablation/a1_velocity_only'

model = dict(
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        velocity_loss_reduction='component_mean',
        x1_weight=0.0,
        translation_weight=0.0,
        keypoints3d_weight=0.0,
        motion_smoothness_weight=0.0,
        trans_dim_weight=1.0,
        fk_consistency_weight=0.0,
        aux_joint_pos_weight=0.0,
        aux_joint_vel_weight=0.0,
        aux_fk_consistency_weight=0.0,
    ),
)
