# HyMotion M2M v3 (DSCF) — Debug config for local testing.
#
# Minimal configuration for verifying forward pass, loss computation,
# and gradient flow on a single GPU. Uses small batch, few steps.
#
# Launch:
#   python tools/train.py configs/hymotion_m2m_v3/hymotion_m2m_v3_debug.py
#
# Expected behavior:
#   - Should complete 1 epoch of ~10 steps without error
#   - Loss should decrease from initial value
#   - All parameter groups should have non-zero gradients

_base_ = './_base_hymotion_m2m_v3_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v3_debug'

# Override to not load pretrained (faster debug startup)
load_from = None

# Small batch for single GPU debug
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        # Use the smaller annotation for quick debug
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            dict(type='Compute198DimPosition', key='motion'),  # 135 -> 198
            dict(type='LocalToGlobalRotation', key='motion'),
            dict(
                type='RandomCropPadding',
                clip_len=120,  # Shorter for debug
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            dict(
                type='PrepareM2MUniversalMask',
                key='motion',
                strategy_weights=dict(
                    m1_random_cell=0.25,
                    m2_random_block=0.15,
                    m3_temporal_contiguous=0.25,
                    m4_joint_contiguous=0.15,
                    m5_full_mask=0.05,
                    m6_keyframe_sparse=0.15,
                ),
                min_mask_ratio=0.1,
                max_mask_ratio=0.9,
                edit_repair_prob=0.2,
                corruptor_names=['jitter', 'joint_jump'],
                max_corruptions=1,
            ),
            dict(
                type='PackInputs',
                keys=['src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length', 'edit_mode'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
    ),
)

# Faster training
train_cfg = dict(
    by_epoch=True,
    max_epochs=2,
    val_interval=1,
    max_grad_norm=1.0,
)

# Disable EMA for debug speed
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    ema=None,
)

# Simpler losses for debug
model = dict(
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        trans_dim_weight=1.0,
        motion_smoothness_weight=0.0,
        fk_consistency_weight=0.0,
    ),
    kimodo_aux_loss_cfg=dict(
        joint_pos_weight=0.0,
        joint_vel_weight=0.0,
        fk_consistency_weight=0.0,
    ),
)
