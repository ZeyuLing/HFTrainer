# Strict HY-Motion 1.0-Lite continuation under hftrainer.
#
# Alignment target:
#   /apdcephfs_cq10/share_1467498/home/rexwen/code/HunyuanMotion_T2M/
#   output/t2m/sft/t2m_v20251114/
#   sft_fm_o6dp1103_04k_qwen3_046B_specialtoken_gpus128/config.yml
#
# This config keeps the hftrainer model/trainer path, but aligns the official
# SFT data records, qwen3 raw/aug policy, text-time crop, special source token,
# fp32 training, SmoothL1 element masked mean, and iteration-based schedule.

_base_ = './hymotion_t2m_201dim_046b.py'

work_dir = 'work_dirs/hymotion_t2m_201dim_lite_official_sft_strict_h20x64_20260703'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    enable_special_game_feat=True,
    train_null_embeddings=True,
    train_special_game_embeddings=True,
    mean_std_dir='checkpoints/HY-Motion-1.0/stats/',
    text_encoder=dict(),
    losses_cfg=dict(
        _delete_=True,
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        trans_dim_weight=1.0,
        velocity_loss_reduction='official_element_mean',
        spike_downweight_enabled=False,
    ),
)

trainer = dict(
    type='HyMotionT2MTrainer',
    val_num_steps=10,
)

train_dataloader = dict(
    _delete_=True,
    # 64 H20 * 48 = 3072 global batch, matching official 128 * 24.
    batch_size=48,
    num_workers=0,
    persistent_workers=False,
    shuffle=True,
    dataset=dict(
        type='HYMotionOfficialT2MDataset',
        data_root='data/hymotion_data',
        input_record_file_dir='_input_record_files/sft_train_v1103_qwen3',
        motion_dir='motions_o6dp_v0922',
        motion_postfix='npy',
        require_motion_file=False,
        pipeline=[
            dict(
                type='LoadPreExtractedTextEmbedding',
                key='caption',
                allow_none=False,
                text_emb_augment_dir='qwen3_augmented',
                refetch_on_missing=True,
                raw_text_prob=0.5,
            ),
            dict(type='LoadO6dp', key='motion', joints_num=22, transl_aug_prob=0.0),
            dict(
                type='CropMotionByTextTime',
                keys='motion',
                fps_key='fps',
                min_frame=10,
                max_frame=360,
            ),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                allow_longer=False,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            dict(
                type='PackInputs',
                keys=[
                    'motion', 'num_frames',
                    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
                    'data_src', 'source', 'input_filename', 'text_emb_dir',
                    'text_source_type',
                ],
                meta_keys=['motion_path', 'caption_path', 'fps', 'caption'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
        refetch=True,
        max_refetch=100,
        verbose=True,
    ),
)

train_cfg = dict(
    by_epoch=False,
    # Official SFT uses 5000 optimizer steps per epoch and max_epoch=50.
    max_iters=250000,
    # External MotionStreamer evaluation should be launched from saved ckpts.
    val_interval=250000,
    max_grad_norm=10.0,
)

optimizer = dict(
    type='Adam',
    lr=1e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
    foreach=True,
)

lr_scheduler = None

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
    ddp_kwargs=dict(
        find_unused_parameters=False,
        static_graph=True,
        broadcast_buffers=False,
    ),
    dataloader_device_placement=False,
    rng_types=[],
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10, iter_interval=10, by_epoch=False),
    # 5000 steps = one official epoch.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, max_keep_ckpts=8, save_last=True),
)

load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
