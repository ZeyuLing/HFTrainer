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
                pretrained_model_name_or_path='checkpoints/Qwen3-0.6B',
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
        audio_tokenizer=None,
        audio_codebook_size=4096,
        instruction_stage=True,
        optional_input_modal_mode='random',
        max_seq_len=0,
    ),
    lm=dict(
        type='VermoQwen3ForCausalLM',
        trainable=True,
        gradient_checkpointing=True,
        module_dtype='bf16',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/Qwen3-0.6B',
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
    persistent_workers=False,
    shuffle=True,
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hq_motionhub_hymotion.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
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
                    'num_person',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
        verbose=False,
        refetch=True,
        task_mode='preset',
        preset_tasks=['t2m', 'm2t'],
        num_person=1,
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
