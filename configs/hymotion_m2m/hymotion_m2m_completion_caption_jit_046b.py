# HyMotion M2M 0.46B — Completion (caption-conditioned) with JiT loss (pred_type='x1').
#
# Caption version: uses pre-extracted Qwen3+CLIP embeddings from .pt files.
# cond_mask_prob=0.3: 30% CFG dropout so the model produces good results even
# without text input at inference.
#
# JiT (Jump-in-Time): x_t = (1-t)*x0 + t*x1, predict x1 directly.
# Timesteps: sigmoid(z), z ~ N(-0.8, 0.8²).
# Loss: velocity reparameterized as (pred_x1 - x_t)/(1-t) + direct x1 reconstruction.
#
# Text embedding strategy: see hymotion_m2m_completion_caption_fm_046b.py for details.
# All embeddings are pre-extracted — no text encoder loaded at training time.
#
# Data: hymotion HQ-only (train_hymotion_400h_hq.json, 408k samples).
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_046b.py 8

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_caption_jit_046b'

model = dict(
    pred_type='x1',
    uncondition_mode=False,
    # No text_encoder needed: all embeddings are pre-extracted.
    cond_mask_prob=0.3,
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,   # reparameterized velocity: (pred_x1 - x_t) / (1-t)
        x1_weight=1.0,         # direct x1 reconstruction
        keypoints3d_weight=0.0,
        translation_weight=0.0,
    ),
)

# Data: hymotion-only, 549k samples
# Caption: 128 text tokens increase attention seq to 488 (vs uncond 361).
# V100 32GB: B=24 peak ~27GB, B=32 OOM.
train_dataloader = dict(
    batch_size=24,
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        pipeline=[
            dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
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
            ),
            dict(
                type='PackInputs',
                keys=['src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length',
                      'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)
