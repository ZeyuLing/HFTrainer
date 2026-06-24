# HyMotion M2M v2 — Unconditioned + Local rotation.
#
# No text encoder, null text embeddings.
# Uses local (SMPL-frame) rotation with 198-dim stats.
#
# 2026-04-25: switched to **v3 universal mask sampler**
# (Rank-K Boolean Tensor Prior). The v3 prior covers any structured
# motion-completion mask — including arbitrary period / arbitrary joint
# subset / arbitrary channel subset — instead of the v2 hand-coded
# Tier-2 templates. See `docs/design/mask_prior_rank_k.md` and
# `hftrainer/models/motion/CLAUDE.md` (§Universal Rank-K prior).
# Edit-repair branch (corruptor pipeline) is unchanged; it is
# orthogonal to the sampler and triggered solely by ``editing_prob``.
#
# Launch (local):
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_uncond_local configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_uncond_local_046b'

model = dict(
    pred_type='velocity',
    uncondition_mode=True,
    text_encoder=None,
    cond_mask_prob=0.0,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
)

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
                # Universal Rank-K Boolean Tensor Prior. K, temporal and
                # dimensional priors live in v3_config (default values
                # from condition_sampler_v3.DEFAULT_*_WEIGHTS are kept).
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
