# HyMotion UMO 201-dim 0.46B config — UMO-style temporal fusion.
#
# Motion representation: 201 dims (o6dp_1103 22-joint format).
#   [transl(3), root_rot6d(6), body_rot6d(126), ric_joints(66)]
#
# Architecture: Frozen HunyuanMotionMMDiT (0.46B T2M-Lite backbone)
#   + Trainable E_ctx (nn.Linear, ~0.2M params)
#   + Trainable meta_op_embeddings (nn.Embedding(3, 201), ~600 params)
#
# Training: flow matching with UMO fusion.
#   Only E_ctx + meta_op_embeddings are trained (~0.207M total).
#   Backbone is frozen after loading T2M pretrained weights.
#
# Data: Uses pre-processed 471-dim o6dp_1103 npy files (52-joint),
#   automatically extracts 22-joint 201-dim subset via LoadO6dp transform.
#   The LoadO6dp transform loads from results['motion_path'] which
#   should point to the .npy file. A RemapMotionPath transform
#   converts the npz path from the annotation to the o6dp npy path.

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/hymotion_umo_201dim_046b'

# ----- Model -----
_motion_dim = 201
_feat_dim = 1024

model = dict(
    type='HyMotionUMOBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=True,   # Will be frozen by trainer after weight loading
        input_dim=_motion_dim,   # 201 (T2M mode, no VACE)
        feat_dim=_feat_dim,
        output_dim=_motion_dim,  # 201
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
    mean_std_dir='/apdcephfs_cq11/share_1467498/home/zeyuling/HY-Motion-1.0/stats/',
    pred_type='velocity',
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
    ),
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.1,
    vtxt_input_dim=768,
    ctxt_input_dim=4096,
    body_model_path=None,
)

trainer = dict(
    type='HyMotionUMOTrainer',
    val_num_steps=10,
    max_text_len=128,
    source_cond_mask_prob=0.1,  # 10% source CFG dropout
)

# ----- Data -----
# The annotation uses smplx_path pointing to .npz files.
# RemapMotionPathToO6dp converts the path to o6dp_v1205 .npy format.
# LoadO6dp loads the 471-dim npy and extracts 22-joint 201-dim.
# PrepareM2MUniversalMask generates mask on 135-dim grid.
# The trainer expands 135-dim mask to 201-dim (joint_pos follows joint rotation mask).
train_dataloader = dict(
    batch_size=32,  # V100 32GB float32: bs=32 peak=18.6GB, bs=40 OOM (backward spike through frozen backbone)
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hymotion_400h.json',
        task_mode='auto',
        num_person=1,
        pipeline=[
            # Load pre-extracted Qwen3+CLIP embeddings (no text encoder needed at train time)
            dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
            # Remap smplx .npz path to o6dp_v1205 .npy path
            dict(type='RemapMotionPathToO6dp',
                 src_dir='motions',
                 dst_dir='motions_o6dp_v1205',
                 src_ext='.npz',
                 dst_ext='.npy'),
            # Load 471-dim npy, extract 22-joint 201-dim
            dict(type='LoadO6dp',
                 key='motion',
                 joints_num=22),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            # Generate frame-level mask for UMO training.
            # UMO uses frame-level meta-ops (P/G/E), NOT per-joint control.
            # Only use strategies that produce whole-body frame masks:
            #   M3 (temporal_contiguous) -> prediction / backcasting / in-betweening
            #   M5 (full_mask) -> text-to-motion (all G)
            #   M6 (keyframe_sparse) -> keyframe infilling
            # Do NOT use M1/M2/M4/M7 which produce per-joint masks (UMO
            # doesn't support joint-level control).
            dict(type='PrepareM2MUniversalMask',
                 key='motion',
                 strategy_weights=dict(
                     m3_temporal_contiguous=55,   # M3: prediction/backcasting/in-betweening
                     m5_full_mask=15,             # M5: text-to-motion (all frames G)
                     m6_keyframe_sparse=30,       # M6: keyframe infilling
                 )),
            dict(
                type='PackInputs',
                keys=['src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length',
                      'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
        verbose=True,
        refetch=True,
    ),
)

# ----- Optimizer -----
# Only ~0.207M params trained — use higher LR than full model.
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    betas=[0.9, 0.99],
    weight_decay=0.01,
)

lr_scheduler = None

# ----- Accelerator -----
# V100 32GB: float32, bs=32 -> 18.6GB peak (tested).
# bf16 not supported on V100; fp16 managed by Accelerate if needed.
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
    checkpoint=dict(type='CheckpointHook', interval=2000, max_keep_ckpts=5, save_last=True),
)

# ----- Load T2M pretrained weights -----
# HY-Motion-1.0-Lite is the 0.46B T2M checkpoint.
# input_dim=output_dim=201 matches the original, so ALL backbone params load.
# E_ctx and meta_op_embeddings are initialized from scratch (E_ctx copies from
# input_encoder at training start).
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
