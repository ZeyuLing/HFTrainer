# HyMotion M2M v2 — PHASE 0 E2+PerMo: SMPL Root + Caption + PerMo Data
#
# **Experiment E2+PerMo**: SMPL root rotation baseline with text caption 
# conditioning, augmented with PerMo dataset (6,542 training samples).
#
# Key features:
#   - Base: Standard E2 config (SMPL Root + caption_local conditioning)
#   - Dataset: 400h HQ + PerMo (414k total training samples)
#   - Keypoint supervision enabled (keypoints3d_weight=10.0)
#   - Standard velocity prediction on 198-dim SMPL representation
#   - Text encoding: QWEN3 + CLIP-L (caption_local variant)
#   - Classifier-Free Guidance (CFG): 10% unconditional during training
#
# Comparison to E2:
#   - E2 uses 400h only (407,552 samples)
#   - E2+PerMo uses 400h + PerMo (414,094 samples, +1.6% data)
#   - Same model architecture and training setup
#   - Different dataset composition for improved generalization
#
# Launch (local):
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_smpl_caption_permo_E2plus \
#     configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_permo_E2plus'

# Resume from caption_local_phase2 checkpoint (epoch 3370).
# caption_local_phase2 has CORRECT null_ctxt values (non-zero).
# load_scope='model' resets optimizer/scheduler (loss config changed).
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,  # Enable text conditioning
    cond_mask_prob=0.1,       # CFG: 10% unconditional during training
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',  # Standard SMPL Root stats
    rotation_space='local',
    caption_freeze_strategy='encoders',  # Freeze vtxt/ctxt/timestep encoders
    text_encoder=dict(),  # Use default QWEN3 + CLIP-L
    losses_cfg=dict(
        # Override base: enable keypoint supervision (E2 baseline)
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
        # CHANGED: Use merged 400h + PerMo annotation file
        anno_file='data/annotation/train_hymotion_400h_permo_caption_20260514.json',
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
                    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)
