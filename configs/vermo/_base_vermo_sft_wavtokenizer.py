_base_ = './_base_vermo_pretrain_wavtokenizer.py'

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        task_bucket_mode='modality',
        verbose=True,
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(type='LoadTxt', key='speech_script', allow_none=True),
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
                type='MotionAudioRandomCrop',
                motion_key='motion',
                audio_key='audio',
                clip_duration=12.0,
                duration_diff_threshold=0.1,
                pair_only=True,
            ),
            dict(
                type='MotionAudioRandomCrop',
                motion_key='motion',
                audio_key='music',
                clip_duration=12.0,
                duration_diff_threshold=0.1,
                pair_only=True,
            ),
            dict(
                type='MotionAudioRandomCrop',
                motion_key='motion',
                audio_key=None,
                clip_duration=12.0,
                duration_diff_threshold=0.1,
                pair_only=False,
            ),
            dict(type='LoadAudio', key='audio', target_sr=24000, allow_none=True),
            dict(type='LoadAudio', key='music', target_sr=24000, allow_none=True),
            dict(
                type='SplitPrediction',
                key='motion',
                past_ratio=0.4,
                random_ratio=False,
                single_frame_prob=0.25,
                min_future_frames=17,
            ),
            dict(
                type='SplitInbetween',
                keys='motion',
                past_ratio=0.2,
                future_ratio=0.2,
                random_ratio=False,
                single_frame_pair_prob=0.25,
                min_edge_frames=4,
                min_middle_frames=4,
            ),
            dict(type='SplitMotionForAR', key='motion', single_frame_prob=1.0),
            dict(type='SplitMusicForAR', key='music'),
            dict(
                type='PackInputs',
                keys=[
                    'task',
                    'motion',
                    'past_motion',
                    'future_motion',
                    'middle_motion',
                    'num_frames',
                    'duration',
                    'audio',
                    'music',
                    'past_music',
                    'future_music',
                    'caption',
                    'speech_script',
                    'num_person',
                    'genre',
                    'per_person_num_frames',
                    'past_per_person_num_frames',
                    'future_per_person_num_frames',
                    'middle_per_person_num_frames',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)

model = dict(
    processor=dict(
        instruction_stage=True,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=2e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

# Set ``load_from = dict(path='.../checkpoint-iter_x', load_scope='model')``
# in this config or via CLI when starting SFT from a pretrained VerMo stage.
