# HyMotion M2M v2 — Caption + Local — Phase 3: Rebalanced caption/control.
#
# Motivation (2026-05-02):
#   The phase2 caption models underperform uncond models on E10 and collapse
#   on E1/E4. Code audit found the eval embeddings are present, so the main
#   training-side issue is an imbalanced curriculum: pure T2M is only 16%,
#   while most batches contain strong condition frames that let the network
#   ignore caption semantics. This config restores caption as a primary signal
#   while keeping enough control masks for M2M.

_base_ = './hymotion_m2m_v2_caption_local_phase2.py'

work_dir = 'work_dirs/hymotion_m2m_v2_caption_local_phase3_rebalanced'

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
                    # K=0 is pure T2M. Raising it from phase2's 0.16 to 0.35
                    # prevents completion/control batches from overwriting
                    # caption-following ability.
                    k_weights=(0.35, 0.32, 0.20, 0.09, 0.04),
                    # Reduce dense full-frame/full-dim conditions; keep sparse
                    # interval/periodic support for E2/E3/E4/E10-style tasks.
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
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_2220/model.safetensors',
    load_scope='model',
)
