# HyMotion T2M 201-dim 0.46B config — Text-to-Motion generation.
#
# Motion representation: 201 dims from HY-Motion-1.0-Lite original format.
# Architecture: HunyuanMotionMMDiT (0.46B) WITHOUT VACE conditioning.
# input_dim = output_dim = 201 (motion_dim only, no VACE multiplier).
#
# This config loads the original HY-Motion-1.0-Lite T2M checkpoint directly.
# Since input_dim=201 and output_dim=201 match the original checkpoint,
# ALL parameters (including input_encoder and final_layer) will be loaded
# successfully — no random initialization needed.
#
# TODO: The current data pipeline with LoadSmplx55 (smpl_type='smpl_22')
# outputs 135 dims. For 201 dims, LoadSmplx55 needs to be extended to also
# output local joint positions (22 joints × 3 dims = 66 dims), giving
# 135 + 66 = 201 dims total. For now, this config uses 135 dims with a
# matching model config override below; update once the data pipeline
# supports 201 dims.

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/hymotion_t2m_201dim_046b'

# ----- Model -----
# 0.46B model dimensions from HY-Motion-1.0-Lite/config.yml:
#   feat_dim=1024, num_layers=18, num_heads=16
#   ctxt_input_dim=4096 (qwen3), vtxt_input_dim=768 (clipl)
#   mask_mode=narrowband, apply_rope_to_single_branch=False
#   time_factor=1000.0
#
# T2M (no VACE): input_dim = output_dim = motion_dim
_motion_dim = 201

model = dict(
    type='HyMotionT2MBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=True,
        input_dim=_motion_dim,         # 201 (NO VACE multiplier)
        feat_dim=1024,
        output_dim=_motion_dim,        # 201
        ctxt_input_dim=4096,
        vtxt_input_dim=768,
        num_layers=18,
        num_heads=16,
        mlp_ratio=4.0,
        mlp_act_type='gelu_tanh',
        norm_type='layer',
        qk_norm_type='rms',
        qkv_bias=True,
        dropout=0.0,
        text_refiner_cfg=dict(num_layers=2),
        final_layer_cfg=dict(act_type='silu'),
        mask_mode='narrowband',
        apply_rope_to_single_branch=False,
        insert_start_token=False,
        with_long_skip_connection=False,
        time_factor=1000.0,
    ),
    # text_encoder: empty dict placeholder so child configs can override.
    # Set to a real config for text conditioning.
    text_encoder=dict(),
    mean_std_dir='checkpoints/HY-Motion-1.0/stats/',
    motion_type='smpl_22',
    pred_type='velocity',
    uncondition_mode=False,    # Text-conditioned, not unconditional
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
    ),
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.1,        # CFG dropout: 10% of samples drop text
    enable_special_game_feat=True,
    vtxt_input_dim=768,
    ctxt_input_dim=4096,
    body_model_path=None,
)

trainer = dict(
    type='HyMotionT2MTrainer',
    val_num_steps=10,
)

# ----- Data -----
# TODO: For 201 dims, LoadSmplx55 needs extension to output local joint
# positions (22×3=66 dims) in addition to the 135-dim rotation_6d + transl.
# For now, this config uses 135 dims. Update motion_dim and input_dim/output_dim
# once the data pipeline supports 201 dims.
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    shuffle=True,
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hq_motionhub_hymotion.json',
        task_mode='auto',
        num_person=1,
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='rel',
                smpl_type='smpl_22',
            ),
            dict(
                type='RandomCropPadding',
                clip_len=360,  # Match HY-Motion T2M 1.0 (train_frames=360)
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            dict(
                type='PackInputs',
                keys=['motion', 'tgt_length'],
                meta_keys=['motion_path', 'fps', 'caption'],
                set_dummy_value=False,
            ),
        ],
        verbose=True,
        refetch=True,
    ),
)

# ----- Optimizer -----
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

lr_scheduler = None

# ----- Accelerator -----
accelerator = dict(
    mixed_precision='bf16',
    gradient_accumulation_steps=1,
)

# ----- Train cfg -----
train_cfg = dict(
    by_epoch=True,
    max_epochs=1000,
    val_interval=10,
    max_grad_norm=1.0,
)

# ----- Hooks -----
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=2000, max_keep_ckpts=5, save_last=True),
)

# ----- Load T2M pretrained weights -----
# HY-Motion-1.0-Lite is the 0.46B T2M checkpoint (460M params, motion_dim=201).
# With input_dim=output_dim=201, all parameters should load successfully.
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
