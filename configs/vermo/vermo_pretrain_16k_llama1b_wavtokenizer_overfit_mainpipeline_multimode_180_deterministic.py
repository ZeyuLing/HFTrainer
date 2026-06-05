"""Deterministic full-task VerMo overfit on the main model chain.

This is the strict correctness check: train/test are the same 180 static cases,
including true and pseudo multi-person T2M/M2T/N2TM/Pred/Inbetween.  Random
augmentation and random prompt choices are disabled so an implementation bug is
not hidden behind stochastic targets.

Compared with the full main experiment, this keeps the same Llama-1B/eager
model chain, tokenizer stack, and task definitions, but makes the tiny overfit
dataset deterministic.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager/'
        'checkpoint-iter_8000'
    ),
    load_scope='full',
)

model = dict(
    processor=dict(
        optional_input_modal_mode='all',
        task_template_mode='first',
        shuffle_modal_parts=False,
        max_seq_len=2048,
    ),
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

vermo_tasks_no_pretrain = [
    't2m',
    'm2t',
    'n2tm',
    'pred',
    'inbetween',
    'm2d',
    'd2m',
    't2md',
    'g2md',
    'n2md',
    'm2d_ar',
    'd2m_ar',
    's2g',
    'g2s',
    't2sg',
    'n2sg',
    'ss2sg',
    's2g_ar',
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    shuffle=False,
    dataset=dict(
        anno_file=(
            'data/annotation/'
            'vermo_overfit_multimode_180_short2s_promptuniq_textpseudo_noaudio_20260604.json'
        ),
        refetch=False,
        verbose=True,
        task_mode='preset',
        preset_tasks=vermo_tasks_no_pretrain,
        task_bucket_mode='none',
        log_task_iter=20,
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
                    'person_captions',
                    'speech_script',
                    'num_person',
                    'genre',
                    'per_person_num_frames',
                    'past_per_person_num_frames',
                    'future_per_person_num_frames',
                    'middle_per_person_num_frames',
                ],
                meta_keys=['motion_path', 'fps', 'person_captions', 'overfit_source_key'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)
