# HyMotion M2M v2 — Caption + Local — Phase 2b: component_mean loss reduction.
#
# Continues from Phase 2 (epoch 3320) with two key changes:
#   1. velocity_loss_reduction = 'component_mean' — splits 198-dim loss into
#      4 semantic groups (trans/root_rot/body_rot/joint_pos), each averaged
#      independently then meta-averaged. Translation gets 25% weight instead
#      of ~1.5% under element_mean.
#   2. trans_dim_weight = 1.0 — component_mean already handles the imbalance;
#      keeping 5.0 would overcorrect to ~55%.
#
# Per-component losses (velocity_trans, velocity_root_rot, velocity_body_rot,
# velocity_joint_pos) are logged for monitoring.
#
# Launch:
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2b.py 8

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_caption_local_phase2b'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        # component_mean: each of the 4 semantic groups gets equal 25% weight.
        # trans_dim_weight=1.0 avoids overcorrection (5.0 would give trans ~55%).
        velocity_loss_reduction='component_mean',
        trans_dim_weight=1.0,
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
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            # Phase 2b: same mask sampler as Phase 2 (v3 Rank-K).
            # K=0 probability raised from 0.10 to 0.16 to match the v2
            # Phase-2 pure_gen budget. Remaining K=1..4 mass renormalised.
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

# Continue from Phase 2 latest checkpoint
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3320/model.safetensors',
    load_scope='model',
    # B2-ext fix: intermediate checkpoints have all-zero null embeddings
    # (safetensors doesn't store bundle-level params). Patch from T2M pretrained.
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)
