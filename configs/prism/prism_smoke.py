_base_ = '../_base_/default_runtime.py'

model = dict(
    type='PrismBundle',
    transformer=dict(
        type='PrismTransformerMotionModel',
        trainable=True,
        patch_size=(1, 1),
        attention_head_dim=16,
        cross_attn_norm=True,
        added_kv_proj_dim=None,
        eps=1e-6,
        ffn_dim=128,
        freq_dim=64,
        in_channels=8,
        num_attention_heads=4,
        num_layers=2,
        out_channels=8,
        qk_norm='rms_norm_across_heads',
        rope_max_seq_len=256,
        text_dim=64,
    ),
    vae=dict(
        type='AutoencoderKLPrism2DTK',
        trainable=False,
        save_ckpt=False,
        base_dim=32,
        z_dim=8,
        dim_mult=(1, 2),
        num_res_blocks=1,
        temporal_downsample=(False, True),
        in_channels=6,
        out_channels=6,
        scale_factor_temporal=2,
        latents_mean=[0.0] * 8,
        latents_std=[1.0] * 8,
        use_static=False,
    ),
    tokenizer=dict(
        type='AutoTokenizer',
        from_pretrained=dict(pretrained_model_name_or_path='tests/assets/motion/tiny_tokenizer'),
    ),
    text_encoder=dict(
        type='T5EncoderModel',
        trainable=False,
        save_ckpt=False,
        from_pretrained=dict(pretrained_model_name_or_path='tests/assets/motion/tiny_t5_encoder'),
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
        smpl_model=None,
        smooth_model=None,
        do_normalize=True,
        stats_file='tests/assets/motion/smpl_stats.json',
        rot_type='rotation_6d',
        transl_type='abs_rel',
        smpl_type='smpl_22',
    ),
)

trainer = dict(
    type='PrismTrainer',
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=32,
    val_prompts=['a person walks forward'],
    num_val_inference_steps=4,
    guidance_scale=3.0,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    shuffle=True,
    dataset=dict(
        type='RandomMotionTextDataset',
        num_samples=8,
        num_frames=17,
        num_joints=22,
        rot_dim=6,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
)

lr_scheduler = None

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

train_cfg = dict(
    by_epoch=False,
    max_iters=10,
    val_interval=100,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2, save_last=True),
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
