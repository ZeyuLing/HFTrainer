# HyMotion M2M v3 — CRFM Smoke Test Config.
#
# Minimal config for verifying CRFM trainer works end-to-end.
# Uses small batch, few epochs, and loads from T2M pretrained.
#
# Launch:
#   python3 tools/train.py configs/hymotion_m2m_v3/hymotion_m2m_v3_smoke.py

_base_ = '../hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v3_smoke'

model = dict(
    motion_transformer=dict(
        enable_cde=True,  # Enable Condition Density Embedding
    ),
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.15,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    text_grad_scale=0.01,  # TAP: near-freeze text attention
)

trainer = dict(
    type='HyMotionM2MCRFMTrainer',
    val_num_steps=10,
    mask_aware_noise=True,
    # TAL config
    tal_weight=0.01,
    tal_interval=2,  # Every 2 steps for smoke test
    tal_min_effect=0.005,
    tal_density_threshold=0.7,
    text_grad_scale=0.01,
)

train_dataloader = dict(
    batch_size=4,  # Small for smoke test
    num_workers=2,
    dataset=dict(
        pipeline=[
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
                corruptor_names=['jitter', 'joint_jump', 'sliding'],
                max_corruptions=2,
                v3_config=dict(k_weights=(0.16, 0.513, 0.233, 0.065, 0.029)),
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

train_cfg = dict(
    by_epoch=True,
    max_epochs=2,
    val_interval=1,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
)

# Load from T2M pretrained for smoke test
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)
