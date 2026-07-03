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
    # HY-Motion-1.0-Lite already provides learned null embeddings, and the
    # T2M trainer never consumes special-game conditioning. Freezing these
    # bundle-level parameters avoids an extra non-DDP grad sync path at 64 ranks.
    train_null_embeddings=False,
    train_special_game_embeddings=False,
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
    # On the 64-card H20 Taiji job, multi-worker batches reach the trainer but
    # can hang during the first shared-memory CPU->CUDA transfer.
    num_workers=0,
    persistent_workers=False,
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
            dict(
                type='LoadPreExtractedTextEmbedding',
                key='caption',
                allow_none=False,
                text_emb_augment_dir='qwen3_augmented',
            ),
            dict(
                type='RemapMotionPathToO6dp',
                src_dir='motions',
                dst_dir='motions_o6dp_v0922',
                src_ext='.npz',
                dst_ext='.npy',
            ),
            dict(type='LoadO6dp', key='motion', joints_num=22, transl_aug_prob=0.0),
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
    # The T2M forward path uses all DDP parameters; find_unused=True adds a
    # graph traversal and can hang in the first backward sync at 64 ranks. On
    # the 8-node H20 job, use NCCL_ALGO=Ring and NCCL_PROTO=Simple at launch;
    # the default NCCL algorithm can also hang in async gradient all-reduce.
    ddp_kwargs=dict(
        find_unused_parameters=False,
        static_graph=True,
        broadcast_buffers=False,
    ),
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
