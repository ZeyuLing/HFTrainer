# HyMotion M2M 0.46B — Completion (caption-conditioned) with flow matching (pred_type='velocity').
#
# Caption version: uses pre-extracted Qwen3+CLIP embeddings from .pt files.
# cond_mask_prob=0.3: 30% CFG dropout so the model works well without text at inference.
#
# Flow matching: x_t = (1-t)*x0 + t*x1, predict v = x1 - x0.
# Timesteps: uniform U[0, 1].
# Loss: SmoothL1(pred_velocity, gt_velocity).
#
# Text embedding strategy:
#   All samples use pre-extracted Qwen3+CLIP .pt files from sibling directories
#   (e.g. qwen3_augmented/, qwen3_human_checked_short/).
#   Training completely bypasses the Qwen3-8B model — no text encoder is loaded,
#   saving ~16GB memory per process and eliminating the 30-60s startup cost.
#
# Data: hymotion HQ-only (train_hymotion_400h_hq.json, 408k samples).
#   Subsets: academic, academicretarget, taobao, game.
#   All subsets have pre-extracted qwen3 embeddings.
#   Motionhub data excluded (lower quality, no pre-extracted embeddings).
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_046b.py 8

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_caption_fm_046b'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    # No text_encoder needed: all embeddings are pre-extracted.
    # text_encoder=dict() (inherited from base) stays falsy → bundle won't load Qwen3-8B.
    cond_mask_prob=0.3,   # 30% CFG dropout: model learns both text-conditioned and
                          # null-text paths. Higher than standard 0.1 because we need
                          # good quality output even without text input at inference.
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
    ),
)

# Data: hymotion-only, 549k samples
# Caption path: ctxt_input (B, 128, 4096) adds 128 text tokens to attention,
# making total seq 488 vs uncond's 361. Attention mem ∝ seq², so ~1.8x more.
# V100 32GB: B=24 peak ~27GB (measured), B=32 OOM.
train_dataloader = dict(
    batch_size=24,
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        pipeline=[
            # Load pre-extracted Qwen3+CLIP embeddings (fast path, no text encoder)
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
