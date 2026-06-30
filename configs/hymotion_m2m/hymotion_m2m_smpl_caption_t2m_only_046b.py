# HyMotion M2M v2 — SMPL Root + Caption, T2M-only baseline
#
# Purpose:
#   Train the M2M architecture as a pure text-to-motion model before mixing in
#   completion, trajectory, or source-edit tasks. Every sample uses mask=all_1
#   and zero source motion, so the model sees exactly the M2M inference state
#   for text-only generation.
#
# Launch (local):
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_smpl_caption_t2m_only_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_smpl_caption_t2m_only \
#     configs/hymotion_m2m/hymotion_m2m_smpl_caption_t2m_only_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_t2m_only_20260630'

# Start from the HY-Motion T2M text prior, not from a mixed-task M2M checkpoint.
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,        # CFG: 10% unconditional text dropout
    motion_cond_mask_prob=0.0, # No motion/source condition dropout in T2M-only
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    caption_freeze_strategy='encoders',
    text_encoder=dict(),       # Use default QWEN3 + CLIP-L embeddings
    losses_cfg=dict(
        keypoints3d_weight=10.0,
        velocity_loss_reduction='component_mean',
    ),
)

trainer = dict(
    mask_aware_noise=False,  # Full mask has no known region; keep phase-1 path simple.
)

train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        # Pure 400h HQ caption data only: no MotionFix/PerMo source-edit pairs.
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
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
            dict(type='PrepareM2Mv2FullMask', key='motion'),
            dict(
                type='PackInputs',
                keys=[
                    'src_motion', 'tgt_motion', 'src_mask',
                    'tgt_length', 'src_length', 'edit_mode',
                ],
                meta_keys=['motion_path', 'fps', 'caption'],
                set_dummy_value=False,
            ),
        ],
    ),
)
