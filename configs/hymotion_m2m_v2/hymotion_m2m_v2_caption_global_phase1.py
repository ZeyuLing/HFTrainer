# HyMotion M2M v2 — Caption + Global — Phase 1: Pure T2M.
#
# Curriculum Phase 1: all samples are pure generation (mask=1 everywhere).
# Global rotation variant. Model learns text-to-motion before completion tasks.
#
# After Phase 1 converges, switch to Phase 2 config and resume.
#
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_caption_global_p1 configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase1.py --host_num 8

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_caption_global_phase1'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_global_rot',
    rotation_space='global',
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        trans_dim_weight=5.0,
        motion_smoothness_weight=0.5,
        fk_consistency_weight=0.1,
        fk_consistency_warmup_steps=2000,
    ),
)

train_dataloader = dict(
    batch_size=20,
    dataset=dict(
        pipeline=[
            dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            dict(type='Compute198DimPosition', key='motion'),
            dict(type='LocalToGlobalRotation', key='motion'),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            # Phase 1: full mask — every sample is pure T2M generation
            dict(type='PrepareM2Mv2FullMask', key='motion'),
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

trainer = dict(
    type='HyMotionM2MTrainer',
    val_num_steps=10,
    mask_aware_noise=False,
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=50,
    val_interval=10,
    max_grad_norm=10.0,
)

# Load from latest caption_global checkpoint (epoch 213)
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_global_046b/checkpoint-epoch_213/model.safetensors',
    load_scope='model',
    # B2-ext fix: intermediate checkpoints have all-zero null embeddings
    # (safetensors doesn't store bundle-level params). Patch from T2M pretrained.
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)
