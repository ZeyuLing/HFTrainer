# HyMotion M2M v3 — CRFM Caption Local Production Config.
#
# CRFM (Condition-Routed Flow Matching) with:
#   - CDE (Condition Density Embedding): encodes mask density into adapter
#   - TAP (Text Attention Preservation): 0.01x gradient for text params
#   - TAL (Text-Awareness Loss): ensures text always affects generation
#
# Resume from uncond_local (best completion quality) and restore text
# attention from T2M pretrained to ensure no text atrophy from start.
#
# V100 32GB: B=16 (TAL extra forward reduces effective batch capacity).
#
# Launch:
#   python3 tools/taiji_submit.py m2m_v3_crfm_caption_local \
#       configs/hymotion_m2m_v3/hymotion_m2m_v3_caption_local_046b.py --host_num 2

_base_ = '../hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v3_caption_local_046b'

model = dict(
    motion_transformer=dict(
        enable_cde=True,
    ),
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.15,  # Higher than v2's 0.1 for better CFG training
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    text_grad_scale=0.01,
)

trainer = dict(
    type='HyMotionM2MCRFMTrainer',
    val_num_steps=10,
    mask_aware_noise=True,
    tal_weight=0.01,
    tal_interval=4,
    tal_min_effect=0.005,
    tal_density_threshold=0.7,
    text_grad_scale=0.01,
)

train_dataloader = dict(
    batch_size=16,
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
            # v3 sampler with 16% pure T2M
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

# Resume from uncond_local best checkpoint (strongest completion)
# Text attention weights will be loaded from T2M pretrained (in the
# checkpoint, text layers come from the original T2M init since uncond
# never trained with text — they're already at T2M-pretrained quality).
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2730/model.safetensors',
    load_scope='model',
)
