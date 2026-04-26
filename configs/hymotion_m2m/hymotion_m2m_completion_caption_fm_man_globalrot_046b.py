# HyMotion M2M 0.46B — Completion (caption-conditioned) + FM + MAN + Global Rotation ablation.
#
# Caption-conditioned variant of global rotation ablation.
# Uses pre-extracted Qwen3+CLIP embeddings (no text encoder loaded).
# See uncond_fm_man_globalrot for detailed description of the global rotation mechanism.
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_man_globalrot_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_man_globalrot_046b.py 8
#   bash tools/taiji_dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_man_globalrot_046b.py

_base_ = './hymotion_m2m_completion_caption_fm_man_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_caption_fm_man_globalrot_046b'

model = dict(
    mean_std_dir='data/hymotion_m2m_data/_stats_global_rot',
    rotation_space='global',
)

# Must override full pipeline (MMEngine list merge = replace) to insert
# LocalToGlobalRotation after LoadSmplx55.
# Caption version: LoadPreExtractedTextEmbedding + text keys in PackInputs.
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
            ),
            dict(
                type='PackInputs',
                keys=[
                    'src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length',
                    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)
