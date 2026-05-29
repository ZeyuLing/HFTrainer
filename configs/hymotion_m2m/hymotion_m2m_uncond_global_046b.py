# HyMotion M2M v2 — Unconditioned + Global rotation.
#
# No text encoder, null text embeddings.
# Uses global (world-frame) rotation with 198-dim global-rot stats.
# Requires LocalToGlobalRotation in pipeline (after Compute198DimPosition).
#
# 2026-04-25: switched to **v3 universal mask sampler**
# (Rank-K Boolean Tensor Prior). See `docs/design/mask_prior_rank_k.md`.
# Edit-repair branch (corruptor pipeline) is unchanged.
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_uncond_global_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_uncond_global_046b.py 8

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_uncond_global_046b'

model = dict(
    pred_type='velocity',
    uncondition_mode=True,
    text_encoder=None,
    cond_mask_prob=0.0,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_global_rot',
    rotation_space='global',
)

# Override pipeline to insert LocalToGlobalRotation after Compute198DimPosition
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            # Compute 198-dim position (FK uses local rotation)
            dict(type='Compute198DimPosition', key='motion'),
            # Convert rotation channels to global (position channels unaffected)
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
                # Universal Rank-K Boolean Tensor Prior (defaults from
                # condition_sampler_v3.DEFAULT_*_WEIGHTS).
                sampler_version='v3',
                editing_prob=0.15,
                corruptor_names=[
                    'jitter', 'joint_jump', 'sliding',
                    'limb_candy_wrapper', 'wrist_candy_wrapper',
                ],
                max_corruptions=2,
            ),
            dict(
                type='PackInputs',
                keys=[
                    'src_motion', 'tgt_motion', 'src_mask',
                    'tgt_length', 'src_length', 'edit_mode',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
    ),
)
