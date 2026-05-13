# HyMotion M2M v2 Validation Training Config
# Purpose: Validate loss spike fixes (Fix 1: max_grad_norm=2.0, Fix 2: spike detection)
# Duration: 10 epochs (validation smoke test)
# GPU: 1×8 V100 expected

_base_ = './_base_hymotion_m2m_v2_046b.py'

# Override for validation: small batch, short training
train_dataloader = dict(
    batch_size=28,
    num_workers=4,
    persistent_workers=False,
    shuffle=True,
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        task_mode='auto',
        num_person=1,
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            dict(type='Compute198DimPosition', key='motion'),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            dict(
                type='PrepareM2Mv2Condition',
                key='motion',
                tier2_prob=0.4,
                editing_prob=0.15,
                corruptor_names=[
                    'jitter', 'joint_jump', 'sliding',
                    'limb_candy_wrapper', 'wrist_candy_wrapper',
                ],
                max_corruptions=2,
            ),
            dict(
                type='PackInputs',
                keys=[
                    'src_motion', 'tgt_motion', 'src_mask',
                    'tgt_length', 'src_length', 'edit_mode',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
        verbose=True,
        refetch=True,
    ),
)

# Validation training: 10 epochs only
train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_interval=1,  # Validate every epoch to monitor loss curves
    max_grad_norm=2.0,  # Fix 1: Gradient clipping threshold (verified correct value)
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=10, save_last=True),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)

work_dir = 'work_dirs/hymotion_m2m_v2_uncond_local_046b_validation'

# Load from T2M pretrained weights (base config default)
# This is the standard initialization for all M2M v2 models
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)
