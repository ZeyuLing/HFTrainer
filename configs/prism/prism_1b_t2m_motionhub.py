_base_ = '../_base_/default_runtime.py'

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=2000, max_keep_ckpts=5, save_last=True),
)

model = dict(
    type='PrismBundle',
    transformer=dict(
        type='PrismTransformerMotionModel',
        trainable=True,
        gradient_checkpointing=True,
        module_dtype='bf16',
        patch_size=(1, 1),
        attention_head_dim=128,
        cross_attn_norm=True,
        added_kv_proj_dim=None,
        eps=1e-6,
        ffn_dim=8960,
        freq_dim=256,
        in_channels=16,
        num_attention_heads=12,
        num_layers=30,
        out_channels=16,
        qk_norm='rms_norm_across_heads',
        rope_max_seq_len=1024,
        text_dim=4096,
    ),
    vae=dict(
        type='AutoencoderKLPrism2DTK',
        trainable=False,
        save_ckpt=False,
        module_dtype='fp32',
        from_pretrained=dict(pretrained_model_name_or_path='checkpoints/vermo_vae'),
    ),
    tokenizer=dict(
        type='T5Tokenizer',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/Wan2.1-VACE-1.3B-diffusers',
            local_files_only=True,
            subfolder='tokenizer',
        ),
    ),
    text_encoder=dict(
        type='UMT5EncoderModel',
        trainable=False,
        save_ckpt=False,
        module_dtype='bf16',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/Wan2.1-VACE-1.3B-diffusers',
            local_files_only=True,
            subfolder='text_encoder',
            low_cpu_mem_usage=False,
        ),
    ),
    scheduler=dict(
        type='FlowMatchEulerDiscreteScheduler',
        num_train_timesteps=1000,
        shift=5.0,
        use_dynamic_shifting=False,
        base_shift=0.5,
        max_shift=1.15,
    ),
    smpl_pose_processor=dict(
        type='SMPLPoseProcessor',
        trainable=False,
        save_ckpt=False,
        do_normalize=True,
        stats_file='data/motionhub/statistics/smplx55_aug_stats_motionhub_motiongv.json',
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
)

trainer = dict(
    type='PrismTrainer',
    condition_num_frames=[1],
    frame_condition_rate=0.0,
    prompt_drop_rate=0.1,
    max_text_length=256,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=False,
    shuffle=True,
    dataset=dict(
        type='MotionHubSingleAgentTextDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/motionhub/train_motionhub_1p.json',
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
                pad_mode='none',
                allow_shorter=True,
            ),
            dict(
                type='PackInputs',
                keys=['motion', 'num_frames', 'caption'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
        verbose=True,
        refetch=True,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
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
    max_epochs=100,
    val_interval=1000,
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
