# HyMotion M2M v2 — PHASE 0 E4+PerMo: KIMODO Root + Caption + PerMo
#
# **Experiment E4+PerMo**: SMPL motion converted to KIMODO Root representation 
# (with online ADMM smoothing) and text caption conditioning, augmented with 
# PerMo dataset (6,542 training samples).
#
# Key features:
#   - Base: Standard E4 config (KIMODO Root + caption_local + ADMM smoothing)
#   - Dataset: 400h HQ + PerMo (414k total training samples)
#   - ADMM online smoothing: 6cm margin on XZ plane (horizontal)
#   - Keypoint supervision enabled (keypoints3d_weight=10.0)
#   - Timestep squared weighting enabled (suppresses FK spikes)
#   - Text encoding: QWEN3 + CLIP-L (caption_local variant)
#   - Classifier-Free Guidance (CFG): 10% unconditional during training
#
# KIMODO Root representation (198-dim):
#   [0:3]      ADMM smoothed pelvis translation (online smoothing)
#   [3:9]      root joint 6D rotation (continuous)
#   [9:135]    body (21 non-root joints) 6D rotations
#   [135:198]  FK-derived joint positions relative to pelvis
#
# Comparison to E4:
#   - E4 uses 400h only (407,552 samples)
#   - E4+PerMo uses 400h + PerMo (414,094 samples, +1.6% data)
#   - Same model architecture with ADMM smoothing
#   - Better consistency for embodied tasks (robot deployment)
#
# Launch (local):
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_kimodo_caption_permo_E4plus \
#     configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_permo_E4plus_editfix_from890_20260528'

# Resume from the correct KIMODO resume checkpoint, then train with the fixed
# PerMo/MotionFix editing pipeline.
# exclude_bundle_keys: ensure mean/std comes from config's mean_std_dir
# load_scope='model' resets optimizer/scheduler for clean start with fixed data.
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4/checkpoint-epoch_890/',
    load_scope='model',
    exclude_bundle_keys=['mean', 'std'],
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,  # Enable text conditioning
    cond_mask_prob=0.1,       # CFG: 10% unconditional during training
    motion_cond_mask_prob=0.3,  # 30% motion condition dropout to prevent condition shortcut
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',  # KIMODO stats
    rotation_space='local',
    caption_freeze_strategy='encoders',  # Freeze vtxt/ctxt/timestep encoders
    text_encoder=dict(),  # Use default QWEN3 + CLIP-L
    losses_cfg=dict(
        # Override base: enable keypoint supervision (E4 baseline)
        keypoints3d_weight=10.0,
        # Decompose velocity loss into components for per-component monitoring
        velocity_loss_reduction='component_mean',
    ),
)

train_dataloader = dict(
    batch_size=20,  # Reduce batch size for caption config (higher memory)
    num_workers=8,  # Keep DataLoader prefetch ahead of train_step
    persistent_workers=True,  # Avoid per-epoch worker restart overhead
    dataset=dict(
        # 400h + PerMo caption + real PerMo/MotionFix instruction editing pairs.
        anno_file='data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260527.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),  # Require captions
            dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            # Compute198DimPosition MUST come before SmplTransToKimodoRootOnline
            dict(type='Compute198DimPosition', key='motion'),
            # KEY DIFFERENCE FROM E2+PerMo: Convert SMPL Root → KIMODO Root
            # ADMM smoothing applied online during __getitem__
            dict(
                type='SmplTransToKimodoRootOnline',
                key='motion',
                admm_margin_m=0.06,  # 6cm margin on XZ plane
            ),
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
            ),
            # Override synthetic corruption with real source motion for
            # PerMo/MotionFix editing pairs. KIMODO root conversion is needed
            # because the target motion has already gone through
            # SmplTransToKimodoRootOnline above.
            dict(
                type='LoadEditingSourceMotion',
                kimodo_root_cfg=dict(admm_margin_m=0.06),
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
