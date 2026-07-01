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
        # T2M-only must match the HYMotion T2M objective. Replace the M2M
        # base loss dict so KIMODO-style aux losses are not inherited.
        _delete_=True,
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        trans_dim_weight=1.0,
        motion_smoothness_weight=0.0,
        fk_consistency_weight=0.0,
        velocity_loss_reduction='modality_mean',
    ),
)

trainer = dict(
    mask_aware_noise=False,  # Full mask has no known region; keep phase-1 path simple.
)

train_cfg = dict(
    # H20 bs=64 hangs inside Accelerate's distributed grad-norm clipping after
    # backward. Keep the T2M-only resume moving and rely on the smooth-L1
    # objective / existing optimizer state instead of clipping this phase.
    val_interval=100,
    max_grad_norm=None,
)

optimizer = dict(
    # weight_decay=0 makes Adam equivalent to AdamW for this phase, while
    # avoiding the H20/cu118 AdamW kernel path that stalls after several steps.
    type='Adam',
    foreach=True,
)

train_dataloader = dict(
    # H20 has ~96GB/card. With NCCL_ALGO=Ring and NCCL_PROTO=Simple the
    # 64-card resume can progress stably, so push the per-GPU batch high enough
    # to use most of the available H20 memory.
    batch_size=100,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        # Pure 400h HQ caption data only: no MotionFix/PerMo source-edit pairs.
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        task_mode='preset',
        preset_tasks=['t2m'],
        # T2M-only must train on real text supervision. The 400h HQ annotation
        # also contains caption-less motion clips for other tasks; filter them
        # up front instead of relying on per-sample refetch during training.
        require_caption=True,
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
            dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=False),
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

default_hooks = dict(
    checkpoint=dict(interval=100),
    # HYMotion T2M official training does not maintain EMA. The M2M base EMA
    # hook doubles the per-step parameter sweep and stalls large 64-card resumes.
    ema=None,
)

accelerator = dict(
    # CFG text dropout routes through trainable null text parameters only for
    # the sampled unconditional clips. Across 64 DDP ranks some ranks can skip
    # that branch on a given step, so unused-parameter detection must stay on
    # to avoid first-step gradient synchronization stalls.
    ddp_kwargs=dict(find_unused_parameters=True),
    # Keep dataloader batches on CPU. HyMotion trainers already move only the
    # needed tensors to device; Accelerate's recursive batch placement stalls
    # on large pre-extracted text/motion batches at H20 bs=64.
    dataloader_device_placement=False,
)
