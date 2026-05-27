# HyMotion M2M v2 — SMPL Root + Caption Conditioning (default caption config)
#
# Standard caption-conditioned M2M training config with:
#   - Text conditioning via QWEN3 + CLIP-L pre-extracted embeddings
#   - Frozen vtxt/ctxt/timestep encoders (from T2M pretrained) to prevent
#     encoder collapse during M2M training
#   - CFG: 10% unconditional during training
#   - Keypoint supervision enabled (keypoints3d_weight=10.0)
#   - PerMo editing pairs (6,177 real Neutral→Emotional editing samples)
#   - Standard velocity prediction on 198-dim SMPL representation
#
# Loads directly from T2M pretrained (clean text encoders). Previous approach
# of loading from caption_local_phase2 was abandoned because vtxt_encoder had
# collapsed (cos>0.98 between all caption embeddings) during the Gen1 training
# chain (046b→phase1→phase2).
#
# Launch (local):
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_smpl_caption configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption'

# Resume from E2 checkpoint (epoch 870) — model already trained on M2M tasks.
# Data fix: MotionFix Z-up bug corrected in-place (2026-05-27), so continuing
# training will learn correct Y-up MotionFix data without losing prior progress.
# load_scope='model' resets optimizer/scheduler for clean start with fixed data.
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_smpl_caption_resume_E2/checkpoint-epoch_870/',
    load_scope='model',
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,  # Enable text conditioning
    cond_mask_prob=0.1,       # CFG: 10% unconditional during training
    motion_cond_mask_prob=0.3,  # 30% motion condition dropout to prevent condition shortcut
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    # Freeze vtxt/ctxt/timestep encoders to preserve T2M text understanding.
    # Without this, vtxt_encoder collapses during M2M training (cos>0.98).
    caption_freeze_strategy='encoders',
    text_encoder=dict(),  # Use default QWEN3 + CLIP-L
    losses_cfg=dict(
        keypoints3d_weight=10.0,
        velocity_loss_reduction='component_mean',
    ),
)

train_dataloader = dict(
    batch_size=20,  # Reduce batch size for caption config (higher memory)
    num_workers=8,  # Increase from 4 to keep DataLoader prefetch ahead of train_step
    persistent_workers=True,  # Avoid per-epoch worker restart overhead
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260514.json',
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
            # Override synthetic corruption with real Neutral source for
            # PerMo editing pairs (source_motion_path present in annotation).
            # Pass-through for regular T2M / completion samples.
            dict(type='LoadEditingSourceMotion'),
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
