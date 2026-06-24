# HyMotion M2M v2 — Caption + Global — Phase 3: Rebalanced caption/control.
#
# Global-rotation twin of the local phase3 config. It keeps the same caption
# and mask-prior redesign so local/global can be compared under the same
# curriculum.

_base_ = './hymotion_m2m_v2_caption_global_phase2.py'

work_dir = 'work_dirs/hymotion_m2m_v2_caption_global_phase3_rebalanced'

model = dict(
    cond_mask_prob=0.2,
)

optimizer = dict(
    lr=5e-5,
)

train_dataloader = dict(
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
            dict(type='LocalToGlobalRotation', key='motion'),
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
                editing_prob=0.05,
                corruptor_names=[
                    'jitter', 'joint_jump', 'sliding',
                    'limb_candy_wrapper', 'wrist_candy_wrapper',
                ],
                max_corruptions=2,
                v3_config=dict(
                    k_weights=(0.35, 0.32, 0.20, 0.09, 0.04),
                    temporal_weights=dict(
                        all=1.0,
                        empty=0.3,
                        interval=3.5,
                        periodic=3.0,
                        renewal=2.0,
                        markov=2.0,
                    ),
                    kind_weights=dict(
                        rot_only=0.20,
                        pos_only=0.25,
                        trans_only=0.13,
                        mixed=0.30,
                        all_dim=0.12,
                    ),
                ),
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

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_global_phase2/checkpoint-epoch_1700/model.safetensors',
    load_scope='model',
)
