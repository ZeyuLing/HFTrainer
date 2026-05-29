# HyMotion M2M v2 — Caption + Global — Phase 2: Mixed T2M + Completion.
#
# Curriculum Phase 2: introduce completion/editing with high T2M ratio.
# Resume from Phase 1 checkpoint.
#
# 2026-04-25: switched to **v3 universal mask sampler**. K=0 (pure
# generation = T2M) raised to 16 % to preserve the v2 Phase-2 T2M
# budget. See `docs/design/mask_prior_rank_k.md`. Edit-repair
# (corruptor pipeline) unchanged.
#
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_caption_global_p2 configs/hymotion_m2m/hymotion_m2m_caption_global_phase2.py --host_num 8

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_caption_global_phase2'

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
        # Disabled: KIMODO-style aux fk_consistency below replaces this.
        fk_consistency_weight=0.0,
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
                v3_config=dict(
                    k_weights=(0.16, 0.513, 0.233, 0.065, 0.029),
                ),
            ),
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
    mask_aware_noise=True,
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=10000,
    val_interval=10,
    max_grad_norm=10.0,
)

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_global_phase1/checkpoint-epoch_50/model.safetensors',
    load_scope='model',
    # B2-ext fix: intermediate checkpoints have all-zero null embeddings
    # (safetensors doesn't store bundle-level params). Patch from T2M pretrained.
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)
