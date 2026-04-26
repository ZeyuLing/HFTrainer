# HyMotion M2M Text-Free DiT base config — Small (48.7M params).
#
# Text-free variant: removes ALL text modules (ctxt_encoder, vtxt_encoder,
# text_refiner, text branches in transformer blocks). Uses HunyuanMotionDiT
# with uniform DiTBlock layers. Train from scratch (no pretrained weights).
#
# Model size: Small
#   feat_dim=512, num_layers=12, num_heads=8 -> ~49M params
#
# Motion representation: smpl_22 with rotation_6d, 135 dims.
# Input: motion (135) + VACE context (3*135) = 540 dims.

_base_ = '../_base_/default_runtime.py'

_motion_dim = 135

model = dict(
    type='HyMotionM2MBundle',
    motion_transformer=dict(
        type='HunyuanMotionDiT',
        trainable=True,
        input_dim=_motion_dim + 3 * _motion_dim,  # 540
        feat_dim=512,
        output_dim=_motion_dim,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        mlp_act_type='gelu_tanh',
        qk_norm_type='rms',
        qkv_bias=True,
        dropout=0.0,
        final_layer_cfg=dict(act_type='silu'),
        mask_mode='narrowband',
        time_factor=1000.0,
    ),
    text_encoder=None,
    mean_std_dir='data/hymotion_m2m_data/_stats',
    motion_type='smpl_22',
    pred_type='velocity',
    uncondition_mode=True,
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        trans_dim_weight=5.0,
        motion_smoothness_weight=0.5,
    ),
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.0,
    vace_condition_mode='split_reactive',
    vtxt_input_dim=768,
    ctxt_input_dim=4096,
    body_model_path=None,
)

trainer = dict(
    type='HyMotionM2MTrainer',
    val_num_steps=10,
    mask_aware_noise=False,
)

train_dataloader = dict(
    batch_size=64,  # Smaller model -> can use larger batch
    num_workers=4,
    persistent_workers=False,
    shuffle=True,
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        task_mode='auto',
        num_person=1,
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            dict(
                type='PrepareM2MUniversalMask',
                key='motion',
                strategy_weights=dict(
                    m1_random_cell=0.25,
                    m2_random_block=0.15,
                    m3_temporal_contiguous=0.25,
                    m4_joint_contiguous=0.15,
                    m5_full_mask=0.05,
                    m6_keyframe_sparse=0.15,
                ),
                min_mask_ratio=0.05,
                max_mask_ratio=0.95,
                edit_repair_prob=0.15,
                corruptor_names=['jitter', 'joint_jump', 'sliding',
                                 'limb_candy_wrapper', 'wrist_candy_wrapper'],
                max_corruptions=2,
            ),
            dict(
                type='PackInputs',
                keys=['src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length', 'edit_mode'],
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
    lr=2e-4,  # Higher LR for smaller model
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
    val_interval=10,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_last=True),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)

# No pretrained weights — train from scratch
load_from = None

val_dataloader = None
val_evaluator = None
val_visualizer = None
