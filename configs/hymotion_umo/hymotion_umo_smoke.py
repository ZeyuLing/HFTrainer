# HyMotion UMO smoke test config.
# Uses synthetic random data to verify the training pipeline works end-to-end.

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/hymotion_umo_smoke'

_motion_dim = 201
_feat_dim = 1024

model = dict(
    type='HyMotionUMOBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=True,
        input_dim=_motion_dim,
        feat_dim=_feat_dim,
        output_dim=_motion_dim,
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
    motion_dim=_motion_dim,
    feat_dim=_feat_dim,
    mean_std_dir=None,  # No normalization for smoke test
    pred_type='velocity',
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
    ),
    cond_mask_prob=0.1,
    vtxt_input_dim=768,
    ctxt_input_dim=4096,
)

trainer = dict(
    type='HyMotionUMOTrainer',
    val_num_steps=2,
    max_text_len=128,
    source_cond_mask_prob=0.1,
)

# Synthetic dataset for smoke test
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    shuffle=True,
    dataset=dict(
        type='HyMotionT2MSyntheticDataset',
        num_samples=10,
        max_frame=64,
        motion_dim=_motion_dim,
    ),
)

optimizer = dict(type='AdamW', lr=1e-3)
lr_scheduler = None

accelerator = dict(mixed_precision='no', gradient_accumulation_steps=1)

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1, max_grad_norm=1.0)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=100),
)

load_from = None
val_dataloader = None
val_evaluator = None
val_visualizer = None
