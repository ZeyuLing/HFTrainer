_base_ = './prism_1b_t2m_motionhub.py'

trainer = dict(
    type='PrismTrainer',
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
)

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        anno_file='data/motionhub/train_motionhub_t2m_1p.json',
        pipeline=[
            dict(type='LoadHierarchicalCaption', allow_none=False),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs_rel',
                smpl_type='smpl_22',
            ),
            dict(
                type='RandomCropPadding',
                clip_len=512,
                pad_mode='replicate',
                allow_shorter=True,
            ),
            dict(
                type='PackInputs',
                keys=['motion', 'num_frames', 'caption'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
    ),
)
