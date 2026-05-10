# HyMotion M2M v3 (DSCF) — Caption-conditioned + Global Rotation.
#
# DSCF architecture with text conditioning via pre-extracted Qwen3+CLIP embeddings.
# cond_mask_prob=0.3: 30% CFG dropout → model works well with and without text.
#
# Architecture difference from v1 caption config:
#   - No VACE input (no 4x input dimension inflation)
#   - Motion condition via MotionCondEncoder cross-attention
#   - Text/motion fusion via learnable gates per block
#   - Despite 2x more params (~606M vs 304M), memory usage is similar because
#     VACE's 4x input inflation was the dominant memory cost
#
# Launch:
#   python tools/train.py configs/hymotion_m2m_v3/hymotion_m2m_v3_caption_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m_v3/hymotion_m2m_v3_caption_046b.py 8

_base_ = './_base_hymotion_m2m_v3_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v3_caption_046b'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.3,
)

# Caption pipeline with pre-extracted text embeddings
train_dataloader = dict(
    batch_size=16,
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
                keys=[
                    'src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length',
                    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length', 'edit_mode',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)
