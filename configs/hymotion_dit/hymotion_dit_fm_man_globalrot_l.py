# HyMotion DiT Large (~383M) — FM + MAN + Global Rotation.
#
# Text-free DiT with flow matching, mask-aware noise, and global rotation.
# Train from scratch, no pretrained weights.
#
# Launch:
#   python tools/train.py configs/hymotion_dit/hymotion_dit_fm_man_globalrot_l.py
#   bash tools/dist_train.sh configs/hymotion_dit/hymotion_dit_fm_man_globalrot_l.py 8

_base_ = './hymotion_dit_fm_man_l.py'

work_dir = 'work_dirs/hymotion_dit_fm_man_globalrot_l'

model = dict(
    mean_std_dir='data/hymotion_m2m_data/_stats_global_rot',
    rotation_space='global',
)

# Must override full pipeline to insert LocalToGlobalRotation after LoadSmplx55.
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            dict(type='LocalToGlobalRotation', key='motion'),
            dict(
                type='RandomCropPadding',
                clip_len=360,
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
                min_mask_ratio=0.05,
                max_mask_ratio=0.95,
                edit_repair_prob=0.15,
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
    ),
)
