_base_ = '../_base_/default_runtime.py'

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1000, max_keep_ckpts=5, save_last=True),
)

model = dict(
    type='VermoBundle',
    processor=dict(
        type='VermoProcessor',
        trainable=False,
        save_ckpt=False,
        module_dtype='fp32',
        pretrained_text_tokenizer=dict(
            type='AutoTokenizer',
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/Llama-3.2-1B-Instruct',
            ),
        ),
        smpl_pose_processor=dict(
            type='SMPLPoseProcessor',
            do_normalize=True,
            stats_file='data/statistic/smplx55_stats_hymotion_aug.json',
            rot_type='rotation_6d',
            transl_type='abs_rel',
            smpl_type='smpl_22',
            smpl_model=dict(
                type='SmplxLiteV437Coco17',
                model_path='checkpoints/smpl_models/smplx',
                smplx2smpl_path='checkpoints/smpl_models/smplx2smpl_sparse.pt',
                coco17_regressor_path='checkpoints/smpl_models/smpl_coco17_J_regressor.pt',
                smplx_verts437_path='checkpoints/smpl_models/smplx_verts437.pt',
                gender='neutral',
                num_betas=10,
            ),
        ),
        motion_tokenizer=dict(
            type='VQVAEWanMotion2DTK',
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/vermo_vqvae2d_16k_rescale_iter47k',
            ),
        ),
        audio_tokenizer=dict(
            type='WavTokenizer',
            pretrained='checkpoints/WavTokenizer-large-unify-40token',
        ),
        audio_codebook_size=4096,
        instruction_stage=False,
        optional_input_modal_mode='random',
        max_seq_len=0,
    ),
    lm=dict(
        type='VermoLlamaForCausalLM',
        trainable=True,
        gradient_checkpointing=True,
        module_dtype='bf16',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/Llama-3.2-1B-Instruct',
        ),
    ),
    mean_init_embeddings=False,
)

trainer = dict(
    type='VermoTrainer',
)

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hq_motionhub_hymotion.json',
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
            dict(type='LoadAudio', key='audio', target_sr=24000, allow_none=True),
            dict(type='LoadAudio', key='music', target_sr=24000, allow_none=True),
            dict(
                type='ComposeMultiPerson',
                compose_prob=0.2,
                min_persons=2,
                max_persons=3,
                placement_radius_range=(1.0, 3.0),
                yaw_range=180.0,
                collision_check=True,
                skip_with_audio=True,
            ),
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
        verbose=False,
        refetch=True,
        task_mode='auto',
        log_task_iter=1000,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

lr_scheduler = None

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=1000,
    val_interval=1000,
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
