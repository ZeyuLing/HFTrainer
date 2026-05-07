"""MotionCLIP smoke config — tiny model, synthetic data, 5 train steps.

For pipeline / API verification only. Do not use for real training.
"""

_base_ = '../_base_/default_runtime.py'

text_config = dict(
    vocab_size=49408,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    max_position_embeddings=64,
    projection_dim=64,
    hidden_act='quick_gelu',
    layer_norm_eps=1e-5,
    attention_dropout=0.0,
    initializer_range=0.02,
    initializer_factor=1.0,
)
motion_config = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    motion_dim=135,
    max_position_embeddings=128,
    projection_dim=64,
    hidden_act='quick_gelu',
    layer_norm_eps=1e-5,
    attention_dropout=0.0,
    initializer_range=0.02,
    initializer_factor=1.0,
)

model = dict(
    type='MotionCLIPBundle',
    text_config=text_config,
    motion_config=motion_config,
    projection_dim=64,
    logit_scale_init_value=2.6592,
    tokenizer=dict(
        type='CLIPTokenizer',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/clip-vit-base-patch32',
        ),
    ),
    smpl_pose_processor=dict(
        type='SMPLPoseProcessor',
        do_normalize=True,
        stats_file='data/statistic/smplx55_stats_hymotion_aug.json',
        rot_type='rotation_6d',
        transl_type='abs',
        smpl_type='smpl_22',
        smpl_model=None,
        smooth_model=None,
    ),
    clip_pretrained=None,           # skip CLIP-init for smoke
    freeze_text_encoder=False,
)

trainer = dict(type='MotionCLIPTrainer')

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    shuffle=True,
    dataset=dict(
        type='MotionCLIPSyntheticDataset',
        num_samples=8,
        max_frame=16,
        motion_dim=135,
    ),
)

optimizer = dict(type='AdamW', lr=1e-4)
lr_scheduler = None

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

work_dir = 'work_dirs/motion_clip_smoke'
train_cfg = dict(by_epoch=False, max_iters=5, val_interval=100)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=1, save_last=True),
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
