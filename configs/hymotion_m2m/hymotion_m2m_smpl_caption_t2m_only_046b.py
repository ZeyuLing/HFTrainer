# HyMotion M2M v2 -- official-SFT aligned T2M-only baseline.
#
# Phase-1 objective:
#   Train the M2M architecture in pure T2M mode while matching the verified
#   HY-Motion 1.0-Lite continuation recipe as closely as the 198-dim M2M
#   representation allows.
#
# Alignment points:
#   - official HYMotion SFT input records:
#       data/hymotion_data/_input_record_files/sft_train_v1103_qwen3
#   - qwen3 raw/augmented sampling with raw_text_prob=0.5
#   - O6DP 201 loading followed by M2M-native 198-dim position recomputation
#   - CropMotionByTextTime before 360-frame padding
#   - learned null and special source token weights from HY-Motion 1.0-Lite
#   - fp32 training and official element-masked SmoothL1 reduction

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_t2m_only_official_sft_h20x64_20260704'

_pack_keys = [
    'src_motion', 'tgt_motion', 'src_mask',
    'tgt_length', 'src_length', 'edit_mode',
    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
    'data_src', 'source', 'input_filename', 'text_emb_dir',
    'text_source_type', 'motion_path', 'caption_path', 'fps', 'caption',
]

# Warm-start from the official Lite checkpoint in the M2M bundle constructor.
# The M2M loader remaps HYMotion-Lite 201-dim IO projections into M2M's
# 198-dim representation and expanded [x_t, reactive, mask] input.  mean/std
# are always excluded by the M2M loader because this config uses 198-dim stats.
load_from = None

model = dict(
    t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    t2m_freeze_strategy='none',
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    motion_cond_mask_prob=0.0,
    enable_special_game_feat=True,
    train_null_embeddings=True,
    train_special_game_embeddings=True,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    caption_freeze_strategy='encoders',
    text_encoder=dict(),
    losses_cfg=dict(
        _delete_=True,
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        trans_dim_weight=1.0,
        motion_smoothness_weight=0.0,
        fk_consistency_weight=0.0,
        velocity_loss_reduction='official_element_mean',
        spike_downweight_enabled=False,
    ),
)

trainer = dict(
    mask_aware_noise=False,
)

train_cfg = dict(
    val_interval=100,
    max_grad_norm=None,
)

optimizer = dict(
    type='Adam',
    lr=1e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
    foreach=True,
)

lr_scheduler = None

train_dataloader = dict(
    _delete_=True,
    batch_size=100,
    num_workers=8,
    persistent_workers=True,
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
            dict(type='Compute198DimPosition', key='motion'),
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
            dict(type='PrepareM2Mv2FullMask', key='motion'),
            dict(
                type='PackInputs',
                keys=_pack_keys,
                meta_keys=[],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
        refetch=True,
        max_refetch=100,
        verbose=True,
    ),
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=100, save_last=True),
    ema=None,
)

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
    ddp_kwargs=dict(find_unused_parameters=True),
    dataloader_device_placement=False,
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
