# PRISM 1B text+pose-to-motion, multi-frame + KT-RoPE spectral + T5 cached
#
# Identical to prism_1b_tp2m_multiframe_kt_spectral.py but uses pre-extracted
# T5 text embeddings from disk instead of online encoding. This eliminates:
#   - T5 forward pass every training step (main speedup)
#
# Pre-extraction must be completed first:
#   python scripts/submit/submit_t5_extract.py --num-shards 64
#
# The text_encoder is still loaded in the bundle (for val_step compatibility)
# but never called during train_step. It can be removed from config entirely
# once inference is handled separately.

_base_ = './prism_1b_tp2m_multiframe_kt_spectral.py'

# ---- Trainer: add null_embedding_path for prompt dropout ----
trainer = dict(
    null_embedding_path='data/t5_feature/_null_embedding.pt',
)

# ---- Dataset: replace LoadCompatibleCaption with LoadPreExtractedT5Feature ----
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            # Load pre-extracted T5 features (replaces LoadCompatibleCaption)
            dict(
                type='LoadPreExtractedT5Feature',
                feature_dir='data/t5_feature',
                data_dir='data/motionhub',
                max_seq_length=256,
                allow_none=True,  # Triggers refetch if .pt not found
                hidden_dim=4096,
            ),
            # Motion loading (unchanged)
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs_rel',
                smpl_type='smpl_22',
                rot6d_convention='column',
                transl_aug_prob=0.75,
                transl_aug_yaw_deg=180.0,
                transl_aug_offset_std=(1.0, 0.0, 1.0),
            ),
            # Temporal crop/pad (unchanged)
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                allow_longer=False,
            ),
            # Pack inputs: add t5_text_embeds and t5_text_mask
            dict(
                type='PackInputs',
                keys=['motion', 'num_frames', 'caption', 't5_text_embeds', 't5_text_mask'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
    ),
)

# Use same work_dir with suffix to distinguish from non-cached run
work_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral_t5cached'
