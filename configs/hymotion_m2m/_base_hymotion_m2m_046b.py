# HyMotion M2M 0.46B base config — Completion task.
#
# Motion representation: smpl_22 with rotation_6d, 135 dims (3 abs transl + 6*22 rot6d).
# Architecture: HunyuanMotionMMDiT (0.46B) with VACE conditioning.
#
# The original HY-Motion-1.0-Lite T2M checkpoint uses motion_dim=201 (33 joints),
# which doesn't match our smpl_22 (135 dim). Only the core transformer blocks
# (feat_dim=1024, num_heads=16, 18 layers) share the same hidden dimensions.
# The input_encoder (T2M: [1024,201] vs M2M: [1024,540]) and final_layer
# (T2M: [201,1024] vs M2M: [135,1024]) will NOT be loaded — they train from
# random init. All 18 transformer layers (double+single stream blocks),
# text encoders (ctxt_encoder, vtxt_encoder), timestep_encoder, and text_refiner
# will be successfully loaded from the T2M checkpoint.

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/hymotion_m2m_completion_046b'

# ----- Model -----
# 0.46B model dimensions from HY-Motion-1.0-Lite/config.yml:
#   feat_dim=1024, num_layers=18, num_heads=16
#   ctxt_input_dim=4096 (qwen3), vtxt_input_dim=768 (clipl)
#   mask_mode=narrowband, apply_rope_to_single_branch=False
#   time_factor=1000.0
#
# For M2M with smpl_22: input_dim = 135 + 3*135 = 540 (motion + VACE context)
# output_dim = 135
_motion_dim = 135  # smpl_22 with rot6d + abs transl: 3 (abs transl) + 6*22 (joints) = 135

model = dict(
    type='HyMotionM2MBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=True,
        input_dim=_motion_dim + 3 * _motion_dim,  # 540 (motion + VACE context)
        feat_dim=1024,
        output_dim=_motion_dim,                    # 135
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
    # text_encoder: None means null embeddings (unconditioned mode).
    # Set to a dict with llm_type/sentence_emb_type in child configs for text conditioning.
    # Use an empty dict as placeholder so child configs can override with a dict without
    # triggering MMEngine's type-mismatch error (None → dict).
    text_encoder=dict(),
    mean_std_dir='data/hymotion_m2m_data/_stats',  # Per-dim normalization (135-dim Mean/Std)
    motion_type='smpl_22',
    pred_type='velocity',  # overridden in child configs
    uncondition_mode=True,
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        trans_dim_weight=5.0,  # Upweight translation dims (first 3) to compensate 3/135 imbalance
        motion_smoothness_weight=0.5,  # Temporal smoothness: penalize jitter in denoised x1
    ),
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.0,  # No text conditioning → no CFG dropout
    vace_condition_mode='split_reactive',
    vtxt_input_dim=768,
    ctxt_input_dim=4096,
    body_model_path=None,
)

trainer = dict(
    type='HyMotionM2MTrainer',
    val_num_steps=10,
    mask_aware_noise=False,  # V4 ablation: set True to keep known regions clean in x_t
)

# ----- Data -----
# Use quality-filtered hymotion data (408k HQ samples from original 549k).
# Filtered by data/hymotion_m2m_refine_data/data_quality_list/high_quality.json
# to exclude low-quality (85k) and borderline (62k) motions that limit model
# quality ceiling. See hftrainer/models/motion/CLAUDE.md §Training Data Quality.
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
            dict(
                type='RandomCropPadding',
                clip_len=360,  # Match HY-Motion T2M 1.0 (train_frames=360)
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
                # Edit-repair: 15% of samples get online corruption (Editing mode)
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
    mixed_precision='no',  # float32 — bf16 精度对 motion 太低
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
# HY-Motion-1.0-Lite is the 0.46B T2M checkpoint (460M params, motion_dim=201).
# Our M2M model uses smpl_22 (135 dim). The core transformer blocks (all 18 layers,
# feat_dim=1024) will load correctly. input_encoder and final_layer have mismatched
# shapes and will be randomly initialized.
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
