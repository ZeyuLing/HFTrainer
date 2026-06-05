"""Paper-test VerMo evaluation config for single-person T2M and M2T.

This keeps the main 64xV100 Llama-1B VerMo model/configuration, but swaps the
training dataset for a deterministic paper-test annotation where every
MotionHub T2M test sample is evaluated once as T2M and once as M2T.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager.py'

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    shuffle=False,
    dataset=dict(
        anno_file='data/annotation/vermo_paper_test_t2m_m2t_valid_20260605.json',
        refetch=False,
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs_rel',
                smpl_type='smpl_22',
                rot6d_convention='column',
            ),
            dict(
                type='MotionAudioMaxDurationFilter',
                motion_key='motion',
                audio_key=None,
                max_duration=12.0,
                pair_only=False,
            ),
            dict(
                type='PackInputs',
                keys=[
                    'task',
                    'motion',
                    'num_frames',
                    'duration',
                    'caption',
                    'person_captions',
                    'num_person',
                ],
                meta_keys=['motion_path', 'fps', 'person_captions'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
        task_mode='auto',
        num_person=1,
        log_task_iter=100,
    ),
)
