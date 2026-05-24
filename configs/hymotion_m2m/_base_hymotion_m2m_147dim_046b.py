# HyMotion M2M 0.46B base config — 147-dim with end-effector positions
#
# Motion representation: smpl_22 with rotation_6d + end-effector positions
# Layout: [3D translation, 132D rot6d (22×6), 12D end-effector positions (4 joints × 3D)]
# Total dims: 147 (3 + 132 + 12)
#
# End-effector joints (SMPL-22 indices):
#   - L_Wrist (20): dims [135:138]
#   - R_Wrist (21): dims [138:141]
#   - L_Foot (10): dims [141:144]
#   - R_Foot (11): dims [144:147]
#
# For M2M with VACE conditioning:
#   input_dim = 147 + 3*147 = 588 (motion + VACE context)
#   output_dim = 147

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/hymotion_m2m_completion_147dim_046b'

# ----- Model -----
# 0.46B model dimensions from HY-Motion-1.0-Lite/config.yml:
#   feat_dim=1024, num_layers=18, num_heads=16
#   ctxt_input_dim=4096 (qwen3), vtxt_input_dim=768 (clipl)
#   mask_mode=narrowband, apply_rope_to_single_branch=False
#   time_factor=1000.0
#
# For M2M with 147-dim: input_dim = 147 + 3*147 = 588
# output_dim = 147
_motion_dim = 147  # smpl_22 with rot6d + abs transl + end-effector pos: 3 + 132 + 12 = 147

model = dict(
    type='HyMotionM2MBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=True,
        input_dim=_motion_dim + 3 * _motion_dim,  # 588 (motion + VACE context)
        feat_dim=1024,
        output_dim=_motion_dim,                    # 147
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
    text_encoder=dict(),
    mean_std_dir='data/hymotion_m2m_data/_stats_147dim',  # 147-dim Mean/Std
    motion_type='smpl_22',
    pred_type='velocity',
    uncondition_mode=True,
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        trans_dim_weight=5.0,  # Upweight translation dims to compensate imbalance
        motion_smoothness_weight=0.5,
        fk_consistency_weight=5.0,  # FK consistency loss weight (γ_fk from roadmap)
        fk_consistency_warmup_steps=10000,  # Warmup scheduling for FK loss
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

# ----- Data -----
train_dataloader = dict(
    batch_size=32,
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
            # Compute 147-dim by appending end-effector positions
            dict(
                type='Compute147DimEndEffector',
                key='motion',
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
    mixed_precision='no',
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
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_last=True),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)

# ----- Load T2M pretrained weights -----
# Core transformer blocks will load correctly. input_encoder and final_layer
# will be randomly initialized due to shape mismatch.
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
