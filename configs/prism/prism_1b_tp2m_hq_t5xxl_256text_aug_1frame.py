_base_ = './prism_1b_tp2m_motionhub.py'

train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        anno_file='data/annotation/train_hq_motionhub_hymotion.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs_rel',
                smpl_type='smpl_22',
                transl_aug_prob=0.75,
                transl_aug_yaw_deg=180.0,
                transl_aug_offset_std=(1.0, 0.0, 1.0),
            ),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                allow_longer=False,
            ),
            dict(
                type='PackInputs',
                keys=['motion', 'num_frames', 'caption'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
        verbose=False,
    ),
)

model = dict(
    smpl_pose_processor=dict(
        stats_file='data/statistic/smplx55_stats_hymotion_aug.json',
    ),
)
