_base_ = '../_base_/default_runtime.py'

model = dict(
    type='VermoBundle',
    processor=dict(
        type='VermoProcessor',
        trainable=False,
        save_ckpt=False,
        pretrained_text_tokenizer=dict(
            type='AutoTokenizer',
            from_pretrained=dict(pretrained_model_name_or_path='tests/assets/motion/tiny_tokenizer'),
        ),
        smpl_pose_processor=dict(
            type='SMPLPoseProcessor',
            smpl_model=None,
            smooth_model=None,
            do_normalize=True,
            stats_file='tests/assets/motion/smpl_stats.json',
            rot_type='rotation_6d',
            transl_type='abs_rel',
            smpl_type='smpl_22',
        ),
        motion_tokenizer=dict(
            type='VQVAEWanMotion2DTK',
            base_dim=32,
            z_dim=8,
            dim_mult=(1, 2),
            num_res_blocks=1,
            temporal_downsample=(False, True),
            in_channels=6,
            out_channels=6,
            scale_factor_temporal=2,
            quantizer_cfg=dict(
                type='FSQuantizer',
                dim=8,
                levels=[2, 2, 2, 2, 2],
            ),
            use_static=False,
        ),
        audio_tokenizer=None,
        audio_codebook_size=32,
        instruction_stage=True,
        optional_input_modal_mode='random',
        max_seq_len=128,
    ),
    lm=dict(
        type='AutoModelForCausalLM',
        from_pretrained=dict(pretrained_model_name_or_path='tests/assets/motion/tiny_llama'),
    ),
    mean_init_embeddings=False,
)

trainer = dict(
    type='VermoTrainer',
)

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    shuffle=True,
    dataset=dict(
        type='VermoToyDataset',
        tasks=['t2m', 'm2t'],
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
