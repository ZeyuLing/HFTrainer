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
# Launch (Taiji H20, 64 GPUs):
#   NNODES=8 NODE_RANK=<rank> MASTER_ADDR=<rank0-ip> MASTER_PORT=29630 \
#     bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_smpl_caption_t2m_only_046b.py 8 --auto-resume

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_t2m_only_h20x64_20260630'

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
    # H20 has ~96GB/card. On 2026-07-01, 8xH20 resume probes from epoch 150
    # showed bs=100 passes multi-step training while bs=104/112/128 OOM.
    batch_size=100,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        # Pure 400h HQ caption data only: no MotionFix/PerMo source-edit pairs.
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        pipeline=[
            # The 400h HQ annotation contains caption-less clips. Keep them in
            # the pool and let LoadPreExtractedTextEmbedding provide learned-null
            # text conditioning, matching the T2M CFG/null-text path.
            dict(type='LoadCompatibleCaption', allow_none=True),
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
                    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
                ],
                # Some HQ clips intentionally have no raw caption. Training
                # consumes the packed text embeddings, so keep metadata keys
                # uniform and avoid collate failures on optional captions.
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)

accelerator = dict(
    # T2M-only exercises a conditional M2M graph; let DDP mark unused
    # parameters explicitly instead of letting buckets wait forever.
    ddp_kwargs=dict(find_unused_parameters=True),
)
