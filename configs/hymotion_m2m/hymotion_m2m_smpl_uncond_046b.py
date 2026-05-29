# HyMotion M2M v2 — PHASE 0 E1: SMPL Root + Unconditioned Baseline
#
# **Experiment E1** from next-gen proposal: baseline with SMPL root rotation
# (no KIMODO smoothing) and unconditional generation (no text).
#
# Key overrides from base config (_base_hymotion_m2m_046b.py):
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
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_smpl_uncond_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_smpl_uncond_E1 configs/hymotion_m2m/hymotion_m2m_smpl_uncond_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_uncond_E1'

# Resume from uncond_local checkpoint (epoch 2930).
# NOTE: uncond_local ckpt has ALL-ZERO null_ctxt due to historical bug.
# null_embedding_source tells the runner to patch zero frozen embeddings
# from the T2M pretrained checkpoint after loading.
# load_scope='model' resets optimizer/scheduler (loss config changed:
# keypoints3d 0→10, t² weighting off, sampler_v3).
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2930',
    load_scope='model',
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)

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
        # Decompose velocity loss into trans/root_rot/body_rot/joint_pos
        # for per-component monitoring in training logs.
        velocity_loss_reduction='component_mean',
    ),
)

train_dataloader = dict(
    batch_size=28,
    num_workers=8,  # Increase from 4 to keep DataLoader prefetch ahead of train_step
    persistent_workers=True,  # Avoid per-epoch worker restart overhead
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
