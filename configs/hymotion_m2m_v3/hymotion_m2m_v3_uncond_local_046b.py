# HyMotion M2M v3 — CRFM Uncond Local Production Config.
#
# Same architecture as v3 caption (with CDE) but cond_mask_prob=1.0
# (all text is dropped). Used as control: verifies CDE does not
# harm unconditioned performance.
#
# Launch:
#   python3 tools/taiji_submit.py m2m_v3_crfm_uncond_local \
#       configs/hymotion_m2m_v3/hymotion_m2m_v3_uncond_local_046b.py --host_num 2

_base_ = '../hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v3_uncond_local_046b'

model = dict(
    motion_transformer=dict(
        enable_cde=True,
    ),
    pred_type='velocity',
    uncondition_mode=True,
    cond_mask_prob=1.0,  # All text dropped = uncond
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    # No TAP needed for uncond (no text pathway to protect)
    text_grad_scale=1.0,
)

trainer = dict(
    type='HyMotionM2MCRFMTrainer',
    val_num_steps=10,
    mask_aware_noise=True,
    # No TAL for uncond (no text to be aware of)
    tal_weight=0.0,
    tal_interval=999999,
    text_grad_scale=1.0,
)

train_dataloader = dict(
    batch_size=28,  # No extra forward for TAL, so same as v2 uncond
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
                corruptor_names=[
                    'jitter', 'joint_jump', 'sliding',
                    'limb_candy_wrapper', 'wrist_candy_wrapper',
                ],
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
    max_epochs=10000,
    val_interval=10,
    max_grad_norm=1.0,
)

# Resume from uncond_local best checkpoint
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2730/model.safetensors',
    load_scope='model',
)
