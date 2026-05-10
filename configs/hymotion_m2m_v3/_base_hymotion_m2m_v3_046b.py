# HyMotion M2M v3 (DSCF) 0.6B base config — Dual-Stream Condition Fusion.
#
# Architecture: HunyuanMotionMMDiTv3 (DSCF)
#   - Motion condition via cross-attention (MotionCondEncoder: 128 queries, 4 layers)
#   - Text condition via cross-attention (same as v1)
#   - TimestepAdaptiveFusionGate per block: learns text vs motion balance
#   - RoleEmbedding: KEEP/GENERATE/EDIT per-frame signal
#   - All 18 blocks are DualCondMMDiTBlocks
#   - Total ~606M params (backbone 304M + MotionCondEncoder 51M + cross-attn additions)
#
# Motion representation: smpl_22, 198-dim (3 trans + 132 rot6d + 63 position).
# Pretrained from: HY-Motion-1.0-Lite (only matching transformer blocks loaded).
#
# Key differences from v1:
#   - No VACE input concat → input_dim = motion_dim (not motion_dim + 3*motion_dim)
#   - Mask-aware noise always ON (weak positional hint in x_t for known regions)
#   - v3 transformer internally handles scalar_mask concat, MotionCondEncoder, RoleEmbedding

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/hymotion_m2m_v3_046b'

# ----- Model -----
_motion_dim = 198  # smpl_22 with rot6d + position: 3 trans + 132 rot6d + 63 position

model = dict(
    type='HyMotionM2Mv3Bundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiTv3',
        trainable=True,
        motion_dim=_motion_dim,
        feat_dim=1024,
        output_dim=_motion_dim,
        ctxt_input_dim=4096,
        vtxt_input_dim=768,
        num_layers=18,
        num_heads=16,
        mlp_ratio=4.0,
        mlp_act_type='gelu_tanh',
        qk_norm_type='rms',
        qkv_bias=True,
        dropout=0.0,
        text_refiner_cfg=dict(num_layers=2),
        final_layer_cfg=dict(act_type='silu'),
        mask_mode='narrowband',
        time_factor=1000.0,
        # ---- v3 specific ----
        cond_encoder_cfg=dict(
            num_queries=128,
            num_layers=4,
            num_heads=16,
            max_seq_len=512,
            dropout=0.0,
        ),
        role_embedding_cfg=dict(
            mode='per_frame',
            zero_init=True,
        ),
        gate_type='timestep',
        include_scalar_mask=True,
    ),
    text_encoder=dict(),
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_global_rot',
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
        fk_consistency_weight=0.1,
    ),
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.0,
    vtxt_input_dim=768,
    ctxt_input_dim=4096,
    body_model_path=None,
    rotation_space='global',
    kimodo_aux_loss_cfg=dict(
        joint_pos_weight=0.1,
        joint_vel_weight=0.05,
        fk_consistency_weight=0.0,
        joint_pos_warmup_steps=1000,
        joint_vel_warmup_steps=1000,
    ),
)

trainer = dict(
    type='HyMotionM2Mv3Trainer',
    val_num_steps=10,
    mask_aware_noise=True,
)

# ----- Data -----
train_dataloader = dict(
    batch_size=16,  # v3 uses more memory (MotionCondEncoder cross-attn + dual cross-attn per block)
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
            dict(type='Compute198DimPosition', key='motion'),  # 135 -> 198
            dict(type='LocalToGlobalRotation', key='motion'),
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

# ----- Load pretrained weights -----
# Load from v1 checkpoint and map to v3 using load_pretrained_backbone().
# The v3 transformer's load_state_dict handles the mapping internally.
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
