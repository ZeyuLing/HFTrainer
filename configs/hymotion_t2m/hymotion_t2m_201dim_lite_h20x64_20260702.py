# HY-Motion 1.0 Lite trainer sanity experiment.
#
# Goal:
#   Start from the released HY-Motion-1.0-Lite checkpoint, train it with the
#   in-repo HyMotionT2MTrainer for 100 epochs, then re-evaluate with the same
#   MotionStreamer/HumanML3D MS272 protocol. This isolates whether the FID gap
#   comes from our trainer/data pipeline or from the later M2M architecture.

_base_ = './hymotion_t2m_201dim_046b.py'

work_dir = 'work_dirs/hymotion_t2m_201dim_lite_h20x64_20260702'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    mean_std_dir='checkpoints/HY-Motion-1.0/stats/',
    text_encoder=dict(),
    losses_cfg=dict(
        _delete_=True,
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
    ),
)

trainer = dict(
    type='HyMotionT2MTrainer',
    val_num_steps=10,
)

train_dataloader = dict(
    _delete_=True,
    batch_size=96,
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        task_mode='preset',
        preset_tasks=['t2m'],
        num_person=1,
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
            dict(type='Compute201DimO6DP', key='motion'),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            dict(
                type='PackInputs',
                keys=[
                    'motion', 'num_frames',
                    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
                ],
                meta_keys=['motion_path', 'fps', 'caption'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
        verbose=True,
        refetch=True,
    ),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=100,
    max_grad_norm=None,
)

optimizer = dict(
    type='Adam',
    lr=1e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
    foreach=True,
)

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
    ddp_kwargs=dict(find_unused_parameters=True),
    dataloader_device_placement=False,
    # Avoid multi-node Accelerate broadcasting an invalid DataLoader mt19937
    # generator state at iterator boundaries.
    rng_types=[],
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3, save_last=True),
)

load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
