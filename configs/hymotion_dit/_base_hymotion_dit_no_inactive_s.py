# HyMotion DiT Small — No-Inactive VACE ablation.
#
# Ablation: remove the inactive channel from VACE conditioning.
# With mask-aware noise (_man), x_t already contains clean values in known
# regions, making the inactive channel redundant. VACE becomes
# cat([reactive, mask]) = 2*D instead of cat([inactive, reactive, mask]) = 3*D.
#
# input_dim = D + 2*D = 405 (was 540 with 3*D VACE)
#
# No pretrained weights — train from scratch.

_base_ = './_base_hymotion_dit_s.py'

work_dir = 'work_dirs/hymotion_dit_no_inactive_s'

_motion_dim = 135

model = dict(
    motion_transformer=dict(
        input_dim=_motion_dim + 2 * _motion_dim,  # 405 (was 540)
    ),
    vace_condition_mode='no_inactive',
)
