"""MotionCLIP evaluator (SMPL-22, 135-dim) — production training config.

Mirrors versatilemotion/configs/motion_clip/motionclip_base_1p_aug_hq.py but
ported to hftrainer's bundle/trainer/runner stack.

Architecture:
  * Text encoder: CLIP ViT-B/32 text tower (init from openai/clip-vit-base-patch32),
    extended to 256 positions for long motion captions.
  * Motion encoder: 12-layer transformer, CLIP ViT-B/32-aligned, 135-dim input
    (3 abs translation + 22 joints x rot6d).
  * Both projected to 512-dim shared embedding space, contrastive CLIP loss.

Usage:
  bash tools/dist_train.sh configs/motion_clip/motion_clip_smpl22_motionhub.py 8
"""

_base_ = '../_base_/default_runtime.py'

# ============================================================================
# Architecture configs (CLIP ViT-B/32-aligned)
# ============================================================================
_clip_b32 = dict(
    vocab_size=49408,
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=8,
    projection_dim=512,
    hidden_act='quick_gelu',
    layer_norm_eps=1e-5,
    attention_dropout=0.0,
    initializer_range=0.02,
    initializer_factor=1.0,
)

motion_dim = 135
motion_max_position_embeddings = 512

text_config = dict(
    **{k: v for k, v in _clip_b32.items() if k != 'projection_dim'},
    max_position_embeddings=256,
    projection_dim=_clip_b32['projection_dim'],
)
motion_config = dict(
    hidden_size=_clip_b32['hidden_size'],
    intermediate_size=_clip_b32['intermediate_size'],
    num_hidden_layers=_clip_b32['num_hidden_layers'],
    num_attention_heads=_clip_b32['num_attention_heads'],
    motion_dim=motion_dim,
    max_position_embeddings=motion_max_position_embeddings,
    projection_dim=_clip_b32['projection_dim'],
    hidden_act=_clip_b32['hidden_act'],
    layer_norm_eps=_clip_b32['layer_norm_eps'],
    attention_dropout=_clip_b32['attention_dropout'],
    initializer_range=_clip_b32['initializer_range'],
    initializer_factor=_clip_b32['initializer_factor'],
)

# ============================================================================
# ModelBundle
# ============================================================================
model = dict(
    type='MotionCLIPBundle',
    text_config=text_config,
    motion_config=motion_config,
    projection_dim=_clip_b32['projection_dim'],
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
    clip_pretrained='checkpoints/clip-vit-base-patch32',
    freeze_text_encoder=False,  # position embeddings extended 77 -> 256
)

# ============================================================================
# Trainer
# ============================================================================
trainer = dict(type='MotionCLIPTrainer')

# ============================================================================
# Pipelines
# ============================================================================
clip_len = 360

train_pipeline = [
    dict(type='LoadCompatibleCaption', allow_none=False),
    dict(
        type='LoadSmplx55',
        key='motion',
        rot_type='rotation_6d',
        transl_type='abs',
        smpl_type='smpl_22',
        transl_aug_prob=0.75,
        transl_aug_yaw_deg=180.0,
        transl_aug_offset_std=(1.0, 0.0, 1.0),
    ),
    dict(
        type='RandomCropPadding',
        clip_len=clip_len,
        pad_mode='replicate',
        allow_shorter=True,
        allow_longer=False,
    ),
    dict(
        type='PackInputs',
        keys=['motion', 'num_frames', 'caption'],
        meta_keys=['motion_path', 'fps'],
        set_dummy_value=False,
    ),
]

val_pipeline = [
    dict(type='LoadCompatibleCaption', allow_none=False),
    dict(
        type='LoadSmplx55',
        key='motion',
        rot_type='rotation_6d',
        transl_type='abs',
        smpl_type='smpl_22',
        transl_aug_prob=0.0,
    ),
    dict(
        type='RandomCropPadding',
        clip_len=clip_len,
        pad_mode='replicate',
        allow_shorter=True,
        allow_longer=False,
    ),
    dict(
        type='PackInputs',
        keys=['motion', 'num_frames', 'caption'],
        meta_keys=['motion_path', 'fps'],
        set_dummy_value=False,
    ),
]

# ============================================================================
# DataLoaders
# ============================================================================
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    shuffle=True,
    dataset=dict(
        type='MotionHubSingleAgentTextDataset',
        motion_key='smplx',
        caption_key='hierarchical_caption',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hq_motionhub_hymotion.json',
        pipeline=train_pipeline,
        verbose=True,
        refetch=True,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    shuffle=True,
    dataset=dict(
        type='MotionHubSingleAgentTextDataset',
        motion_key='smplx',
        caption_key='hierarchical_caption',
        data_dir='data/motionhub',
        anno_file='data/annotation/test_hml3d.json',
        pipeline=val_pipeline,
        verbose=True,
        refetch=True,
    ),
)

# ============================================================================
# Optimizer / Scheduler
# ============================================================================
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
lr_scheduler = None

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

# ============================================================================
# Train Config
# ============================================================================
work_dir = 'work_dirs/motion_clip_smpl22_motionhub'

train_cfg = dict(
    by_epoch=True,
    max_epochs=1000,
    val_interval=20,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=5,
        save_last=True,
    ),
)

val_evaluator = None
val_visualizer = None
