# HyMotion M2M v2 — PHASE 0 E1: SMPL Root + Unconditioned Baseline
#
# **Experiment E1** from next-gen proposal: baseline with SMPL root rotation
# (no KIMODO smoothing) and unconditional generation (no text).
#
# Key overrides from base config (_base_hymotion_m2m_v2_046b.py):
#   - keypoints3d_weight: 0.0 → 10.0 (enable keypoint supervision)
#   - timestep_squared_weighting: True → False (standard loss weighting)
#   - Rotation space: local (SMPL frame)
#   - Text conditioning: disabled (uncond_mode=True)
#
# This is the simplest E1 baseline for Phase 0 validation:
# - No ADMM smoothing / KIMODO Root transform
# - Standard velocity prediction on 198-dim SMPL representation
# - Focus on motion quality metrics without confounding from root representation
#
# Launch (local):
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_uncond_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_smpl_uncond_E1 configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_uncond_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_uncond_E1'

model = dict(
    pred_type='velocity',
    uncondition_mode=True,
    text_encoder=None,
    cond_mask_prob=0.0,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    losses_cfg=dict(
        # Override base: enable keypoint supervision (E1 baseline)
        keypoints3d_weight=10.0,
    ),
    # Override base: disable t² weighting for standard loss weighting
    kimodo_aux_loss_cfg=dict(
        joint_pos_weight=50.0,
        joint_vel_weight=500.0,
        fk_consistency_weight=1500.0,
        loss_type='smooth_l1',
        timestep_squared_weighting=False,  # E1 baseline: standard weighting
        fk_consistency_warmup_steps=2000,
        joint_pos_warmup_steps=2000,
        joint_vel_warmup_steps=2000,
    ),
)

train_dataloader = dict(
    batch_size=28,
    dataset=dict(
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
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
