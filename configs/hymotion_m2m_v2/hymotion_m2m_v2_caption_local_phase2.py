# HyMotion M2M v2 — Caption + Local — Phase 2: Mixed T2M + Completion.
#
# Curriculum Phase 2: introduce completion/editing tasks while keeping
# high T2M ratio. Resume from Phase 1 checkpoint.
#
# Key differences from default config:
#   - mask_aware_noise=True (for completion tasks)
#   - load_from points to Phase 1 best checkpoint
#
# 2026-04-25: switched to **v3 universal mask sampler**.
# Pure-generation (T2M) share = K=0 probability. v2 phase-2 used
# ``tier2_prob * pure_gen = 0.4 * 0.4 = 16 %`` global pure-gen; we
# preserve that target by overriding the default K weights below
# (DEFAULT_K_WEIGHTS K=0 is only 10 %). Other K=1..4 weights are
# rescaled proportionally. See `docs/design/mask_prior_rank_k.md`.
# Edit-repair branch (corruptor pipeline) is unchanged; triggered by
# ``editing_prob`` only.
#
# Launch:
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2.py 8 \
#     --load-from work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_200/model.safetensors

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_caption_local_phase2'

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
        trans_dim_weight=5.0,
        motion_smoothness_weight=0.5,
        # Disabled: KIMODO-style aux fk_consistency below replaces this.
        fk_consistency_weight=0.0,
        fk_consistency_warmup_steps=2000,
    ),
    # KIMODO-style auxiliary losses: see base config for the rationale behind
    # the weight magnitudes (denormalised-metres regime, smooth_l1 + t²).
    kimodo_aux_loss_cfg=dict(
        joint_pos_weight=50.0,
        joint_vel_weight=500.0,
        fk_consistency_weight=1500.0,
        loss_type='smooth_l1',
        timestep_squared_weighting=True,
        fk_consistency_warmup_steps=2000,
        joint_pos_warmup_steps=2000,
        joint_vel_warmup_steps=2000,
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
            # Phase 2: mixed T2M + completion / editing.
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
                    # 0.16 + 0.55*(0.84/0.90) + 0.25*(0.84/0.90)
                    #     + 0.07*(0.84/0.90) + 0.03*(0.84/0.90) ≈ 1.0
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
    mask_aware_noise=True,  # Phase 2: MAN for completion tasks
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=10000,
    val_interval=10,
    max_grad_norm=1.0,
)

# Load Phase 1 checkpoint (override base config's T2M pretrained weights)
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_50/model.safetensors',
    load_scope='model',
)
