# PRISM 1B text+pose-to-motion, multi-frame + KT-RoPE spectral_unified + T5 cached
#
# Combines spectral_unified (fixed KT-RoPE) with pre-extracted T5 features.
# Key differences from base spectral_unified config:
#   - T5 encoder and tokenizer are NOT loaded (saves ~11GB GPU memory)
#   - Uses LoadPreExtractedT5Feature to load pre-extracted embeddings from disk
#   - Prompt dropout uses pre-extracted null embedding (_null_embedding.pt)
#
# Pre-extraction must be completed first:
#   python scripts/submit/submit_t5_extract.py --num-shards 64
#
# This config does NOT load any text encoder — val_step with online encoding
# is disabled. For inference, use a separate config that includes text_encoder.

_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified.py'

# ---- Remove T5 encoder and tokenizer from bundle (not loaded at all) ----
model = dict(
    tokenizer=None,
    text_encoder=None,
)

# ---- Trainer: add null_embedding_path for prompt dropout ----
# Text cross-attention runs without a mask (matches the official Wan setup:
# text padded with zeros, context_lens=None) — hardcoded in PrismTrainer.
trainer = dict(
    null_embedding_path='data/t5_feature/_null_embedding.pt',
)

# ---- Dataset: replace LoadCompatibleCaption with LoadPreExtractedT5Feature ----
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            # Load pre-extracted T5 features (replaces online T5 encoding)
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
            # Pack inputs: include t5_text_embeds and t5_text_mask
            dict(
                type='PackInputs',
                keys=['motion', 'num_frames', 'caption', 't5_text_embeds', 't5_text_mask'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
    ),
)

work_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached'
