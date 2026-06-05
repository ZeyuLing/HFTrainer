"""VerMo 16k Llama-1B all-task overfit validation config.

The annotation is train/test shared by construction:
``data/annotation/vermo_overfit_alltasks_190_20260603.json`` contains
19 VerMo tasks x 10 deterministic source samples.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_v100ampfp16'

model = dict(
    processor=dict(
        optional_input_modal_mode='all',
        task_template_mode='first',
        shuffle_modal_parts=False,
        max_seq_len=0,
    ),
    lm=dict(
        module_dtype='fp32',
    ),
)

accelerator = dict(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    shuffle=False,
    dataset=dict(
        anno_file='data/annotation/vermo_overfit_alltasks_190_20260603.json',
        refetch=False,
        verbose=True,
        task_mode='auto',
        task_bucket_mode='none',
        log_task_iter=10,
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True, select_mode='first'),
            dict(type='LoadTxt', key='speech_script', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs_rel',
                smpl_type='smpl_22',
                rot6d_convention='column',
                transl_aug_prob=0.0,
            ),
            dict(type='LoadAudio', key='audio', target_sr=24000, allow_none=True),
            dict(type='LoadAudio', key='music', target_sr=24000, allow_none=True),
            dict(type='ComposeMultiPerson', compose_prob=0.0),
            dict(
                type='MotionAudioMaxDurationFilter',
                motion_key='motion',
                audio_key='audio',
                max_duration=12.0,
                pair_only=True,
            ),
            dict(
                type='MotionAudioMaxDurationFilter',
                motion_key='motion',
                audio_key='music',
                max_duration=12.0,
                pair_only=True,
            ),
            dict(
                type='MotionAudioMaxDurationFilter',
                motion_key='motion',
                audio_key=None,
                max_duration=12.0,
                pair_only=False,
            ),
            dict(
                type='SplitPrediction',
                key='motion',
                past_ratio=0.4,
                random_ratio=False,
                single_frame_prob=0.0,
                min_future_frames=17,
            ),
            dict(
                type='SplitInbetween',
                keys='motion',
                past_ratio=0.2,
                future_ratio=0.2,
                random_ratio=False,
                single_frame_pair_prob=0.0,
                min_edge_frames=4,
                min_middle_frames=4,
            ),
            dict(
                type='SplitMotionForAR',
                key='motion',
                single_frame_prob=1.0,
                min_future_frames=8,
            ),
            dict(
                type='SplitMusicForAR',
                key='music',
                past_ratio=0.2,
                random_ratio=False,
                min_future_samples=4000,
            ),
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
                meta_keys=['motion_path', 'fps', 'overfit_source_key'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_cfg = dict(
    by_epoch=False,
    max_iters=5000,
    val_interval=10000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=250,
        max_keep_ckpts=5,
        save_last=True,
    ),
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
