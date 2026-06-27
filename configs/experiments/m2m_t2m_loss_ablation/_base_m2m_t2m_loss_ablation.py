# M2M T2M-only LOSS ablation — shared base.
#
# Goal: test whether M2M's auxiliary / smoothness / translation-weighting
# losses are *negative optimization* for PURE text-to-motion, by holding
# EVERYTHING else fixed and varying ONLY `model.losses_cfg` across arms:
#   - init      : HY-Motion-1.0-Lite backbone (inherited load_from in _base;
#                 transformer blocks + text encoders loaded, input_encoder /
#                 final_layer reinitialized by the bundle on shape mismatch)
#   - data      : train_hq_motionhub_hymotion.json (same source as HYMotion T2M)
#   - target    : 198-dim SMPL (Scheme-D position) + _stats_198dim normalization
#   - condition : pure T2M via v3 sampler K=0 (all-generate mask, reactive=0),
#                 editing_prob=0, no corruptors
#   - text      : QWEN3 + CLIP-L pre-extracted embeddings, encoders frozen
#   - budget    : identical max_epochs / batch_size across all arms
#
# Arms (configs in this dir):
#   a0_full           — current full M2M recipe (velocity + smoothness + trans
#                       weighting + joint_pos/vel + fk_consistency + keypoints3d)
#   a1_velocity_only  — velocity only (== HYMotion T2M objective on 198-dim)
#   a2_no_smoothness  — a0 minus motion_smoothness
#   a3_no_aux_geom    — a0 minus joint_pos/vel + fk_consistency + keypoints3d
#
# NOTE: experimental / ablation config — intentionally NOT in configs/hymotion_m2m/.

_base_ = '../../hymotion_m2m/_base_hymotion_m2m_046b.py'

# Overridden per arm.
work_dir = 'work_dirs/m2m_t2m_loss_ablation/_base'

model = dict(
    uncondition_mode=False,      # text-conditioned
    cond_mask_prob=0.1,          # CFG: 10% unconditional during training
    caption_freeze_strategy='encoders',  # preserve T2M text understanding
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    text_encoder=dict(),         # default QWEN3 + CLIP-L
    # losses_cfg intentionally NOT defined here — set per arm so the ONLY
    # variable in this ablation is the loss objective.
)

train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        anno_file='data/annotation/train_hq_motionhub_hymotion.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
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
            # Pure T2M: K=0 -> lock=empty -> src_mask all-generate, reactive=0.
            dict(
                type='PrepareM2Mv2Condition',
                key='motion',
                sampler_version='v3',
                editing_prob=0.0,
                corruptor_names=[],
                max_corruptions=0,
                v3_config=dict(k_weights=[1.0, 0.0, 0.0, 0.0, 0.0], editing_prob=0.0),
            ),
            dict(type='LoadEditingSourceMotion'),  # no-op for T2M samples
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

# Save every 5 epochs so the per-epoch T2M metric trend can be tracked under an
# equal training budget across arms.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=5, max_keep_ckpts=40, save_last=True),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=200,
    val_interval=10,
    max_grad_norm=2.0,
)
