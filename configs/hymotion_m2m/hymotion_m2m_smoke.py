_base_ = '../_base_/default_runtime.py'

# Tiny HunyuanMotionMMDiT for smoke testing.
# Uses small feat_dim, few layers, and synthetic random data.
model = dict(
    type='HyMotionM2MBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=True,
        # ----- tiny dims for smoke test -----
        input_dim=135 + 3 * 135,       # motion_dim + 3 * motion_dim (VACE context)
        feat_dim=64,
        output_dim=135,
        ctxt_input_dim=64,
        vtxt_input_dim=64,
        num_layers=3,
        num_heads=4,
        mlp_ratio=2.0,
        mlp_act_type='gelu_tanh',
        norm_type='layer',
        qk_norm_type='rms',
        qkv_bias=True,
        dropout=0.0,
        text_refiner_cfg=dict(num_layers=1),
        final_layer_cfg=dict(act_type='silu'),
        mask_mode=None,
        apply_rope_to_single_branch=True,
        insert_start_token=False,
        with_long_skip_connection=False,
    ),
    # No text encoder: use null embeddings (uncondition mode)
    text_encoder=None,
    mean_std_dir=None,
    motion_type='smpl_22',
    pred_type='velocity',
    uncondition_mode=True,
    losses_cfg=dict(
        loss_type='mse',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
    ),
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=2),
    cond_mask_prob=0.0,
    vace_condition_mode='split_reactive',
    vtxt_input_dim=64,
    ctxt_input_dim=64,
    body_model_path=None,
)

trainer = dict(
    type='HyMotionM2MTrainer',
    val_num_steps=2,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    shuffle=True,
    dataset=dict(
        type='HyMotionM2MSyntheticDataset',
        num_samples=8,
        max_frame=16,
        motion_dim=135,
        mask_ratio=0.5,
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
