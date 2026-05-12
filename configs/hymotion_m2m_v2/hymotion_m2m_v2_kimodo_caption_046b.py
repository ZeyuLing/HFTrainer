# HyMotion M2M v2 — PHASE 0 E4: KIMODO Root + Caption Conditioning
#
# **Experiment E4** from next-gen proposal: SMPL motion converted to KIMODO
# Root representation (with online ADMM smoothing) and text caption conditioning.
#
# Key overrides from base config (_base_hymotion_m2m_v2_046b.py):
#   - keypoints3d_weight: 0.0 → 10.0 (enable keypoint supervision)
#   - timestep_squared_weighting: True → False (standard loss weighting)
#   - cond_mask_prob: 0.0 → 0.1 (enable CFG during training)
#   - Data pipeline: Add SmplTransToKimodoRootOnline transform (ADMM smoothing)
#   - Text encoding: enabled (caption_local)
#   - Rotation space: local (SMPL frame)
#   - Mean/std: KIMODO Root 198-dim stats
#
# KIMODO Root representation (198-dim):
#   [0:3]      ADMM smoothed pelvis translation (online smoothing during load)
#   [3:9]      root joint 6D rotation (continuous)
#   [9:135]    body (21 non-root joints) 6D rotations
#   [135:198]  FK-derived joint positions relative to pelvis (21 × 3)
#
# ADMM smoothing: Applied online during dataset __getitem__ with margin ≤ 6cm
# on XZ plane (horizontal), Y-axis unchanged (vertical is unsmoothed).
#
# E4 combines KIMODO Root with text guidance (T2M with structured root):
#   - Text-driven generation with smoothed root trajectory
#   - Better consistency for embodied tasks (robot deployment)
#   - Position-to-root relative coordinates for text conditioning
#
# Prerequisite: KIMODO Root mean/std must be computed and placed at
#   data/hymotion_m2m_data/_stats_198dim_kimodo_root/
#
# Launch (local):
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_kimodo_caption_E4 configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_E4'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,  # Enable text conditioning
    cond_mask_prob=0.1,       # CFG: 10% unconditional during training
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',
    rotation_space='local',
    text_encoder=dict(),  # Use default QWEN3 + CLIP-L
    losses_cfg=dict(
        # Override base: enable keypoint supervision (E4 baseline)
        keypoints3d_weight=10.0,
    ),
    # Override base: disable t² weighting for standard loss weighting
    kimodo_aux_loss_cfg=dict(
        joint_pos_weight=50.0,
        joint_vel_weight=500.0,
        fk_consistency_weight=1500.0,
        loss_type='smooth_l1',
        timestep_squared_weighting=False,  # E4 baseline: standard weighting
        fk_consistency_warmup_steps=2000,
        joint_pos_warmup_steps=2000,
        joint_vel_warmup_steps=2000,
    ),
)

train_dataloader = dict(
    batch_size=20,  # Reduce batch size for caption config (higher memory)
    dataset=dict(
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),  # Require captions
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            # Compute198DimPosition MUST come before SmplTransToKimodoRootOnline:
            # LoadSmplx55 outputs 135-dim, Compute198DimPosition → 198-dim,
            # then SmplTransToKimodoRootOnline smooths translation on 198-dim.
            dict(type='Compute198DimPosition', key='motion'),
            # **KEY DIFFERENCE FROM E2/E4**: Convert SMPL Root → KIMODO Root
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
            dict(
                type='PackInputs',
                keys=[
                    'src_motion', 'tgt_motion', 'src_mask',
                    'tgt_length', 'src_length', 'edit_mode',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
    ),
)
