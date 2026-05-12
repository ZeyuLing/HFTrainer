# HyMotion M2M v2 — PHASE 0 E2: SMPL Root + Caption Conditioning
#
# **Experiment E2** from next-gen proposal: SMPL root rotation baseline
# with text caption conditioning (caption_local variant).
#
# Key overrides from base config (_base_hymotion_m2m_v2_046b.py):
#   - keypoints3d_weight: 0.0 → 10.0 (enable keypoint supervision)
#   - timestep_squared_weighting: True → False (standard loss weighting)
#   - cond_mask_prob: 0.0 → 0.1 (enable CFG during training)
#   - text encoding: enabled (caption_local)
#   - Rotation space: local (SMPL frame)
#
# This validates text-to-motion generation with SMPL Root baseline:
# - Caption conditioning + classifier-free guidance
# - Standard velocity prediction on 198-dim SMPL representation
# - No ADMM smoothing / KIMODO Root transform
#
# Difference from E1: Adds text encoder + caption pipeline
#
# Launch (local):
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_smpl_caption_E2 configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_E2'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,  # Enable text conditioning
    cond_mask_prob=0.1,       # CFG: 10% unconditional during training
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    text_encoder=dict(),  # Use default QWEN3 + CLIP-L
    losses_cfg=dict(
        # Override base: enable keypoint supervision (E2 baseline)
        keypoints3d_weight=10.0,
    ),
    # Override base: disable t² weighting for standard loss weighting
    kimodo_aux_loss_cfg=dict(
        joint_pos_weight=50.0,
        joint_vel_weight=500.0,
        fk_consistency_weight=1500.0,
        loss_type='smooth_l1',
        timestep_squared_weighting=False,  # E2 baseline: standard weighting
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
