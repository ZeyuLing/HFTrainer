# HyMotion M2M v2 — Overfitting experiment on 100 samples
#
# Purpose: Verify model implementation correctness by overfitting on 100
# training samples. After convergence, the model should reproduce training
# motions almost exactly under any given mask (text-only or text+frame).
#
# Strategy:
#   - Same training strategy as full caption config (all mask types, caption conditioning)
#   - 100 diverse samples with verified pre-extracted text embeddings
#   - Frequent checkpointing (every 50 epochs) for loss convergence monitoring
#   - Single node 8x V100 GPUs
#
# Launch:
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_overfit_100_caption_046b.py 8 --auto-resume
# Or via Taiji (1 node = 8 GPUs):
#   python tools/taiji_submit.py m2m_v2_overfit_100 configs/hymotion_m2m_v2/hymotion_m2m_v2_overfit_100_caption_046b.py --host_num 1

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_overfit_100_v2'

# Load from T2M pretrained (same as full caption config)
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,  # Enable text conditioning
    cond_mask_prob=0.1,       # CFG: 10% unconditional during training
    motion_cond_mask_prob=0.3,  # 30% motion condition dropout
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    # Freeze vtxt/ctxt/timestep encoders to preserve T2M text understanding
    caption_freeze_strategy='encoders',
    text_encoder=dict(),  # Use default QWEN3 + CLIP-L
    losses_cfg=dict(
        keypoints3d_weight=10.0,
        velocity_loss_reduction='component_mean',
    ),
)

train_dataloader = dict(
    batch_size=8,  # 100 samples / 8 GPUs / bs=8 ≈ ~2 steps per epoch (with remainder)
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        anno_file='data/annotation/overfit_100_caption_20260526.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),  # Require captions
            dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            dict(type='Compute198DimPosition', key='motion'),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            dict(
                type='PrepareM2Mv2Condition',
                key='motion',
                sampler_version='v3',
                editing_prob=0.15,
                corruptor_names=[
                    'jitter', 'joint_jump', 'sliding',
                    'limb_candy_wrapper', 'wrist_candy_wrapper',
                ],
                max_corruptions=2,
            ),
            # No LoadEditingSourceMotion — overfit set has no editing pairs
            dict(
                type='PackInputs',
                keys=[
                    'src_motion', 'tgt_motion', 'src_mask',
                    'tgt_length', 'src_length', 'edit_mode',
                    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)

# ----- Train cfg (overfit-specific) -----
# 100 samples / (8 GPUs × bs=8) ≈ 2 steps/epoch
# Target: train until loss converges; prev run hit 10K epochs without convergence
# due to MotionFix Z-up data bug (now fixed). 20K epochs should be enough.
train_cfg = dict(
    by_epoch=True,
    max_epochs=20000,
    val_interval=100,
    max_grad_norm=2.0,
)

# Higher LR for overfitting (2x default) — helps convergence on small dataset
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

# ----- Hooks (frequent checkpointing for overfit monitoring) -----
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),  # Log every step
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=50, save_last=True),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
