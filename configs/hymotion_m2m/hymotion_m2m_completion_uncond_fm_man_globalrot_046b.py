# HyMotion M2M 0.46B — Completion (unconditioned) + FM + MAN + Global Rotation ablation.
#
# Global rotation ablation: training data uses world-frame global rotation instead
# of SMPL local rotation. The 135-dim representation is unchanged (3 transl + 22*6 rot6d),
# but the rot6d semantics change from parent-relative to world-frame absolute.
#
# Hypothesis: global rotation makes masked joint imputation easier because
# neighboring joints' rotations are in the same coordinate frame, enabling
# geometric interpolation without IK (similar to KIMODO's approach).
#
# Key differences from local rotation baseline:
#   - model.mean_std_dir -> _stats_global_rot (different normalization)
#   - model.rotation_space = 'global' (decode converts global->local for SMPL output)
#   - Pipeline: LocalToGlobalRotation inserted after LoadSmplx55
#   - Mask-aware noise inherited from _man parent
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py 8
#   bash tools/taiji_dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py

_base_ = './hymotion_m2m_completion_uncond_fm_man_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_uncond_fm_man_globalrot_046b'

model = dict(
    mean_std_dir='data/hymotion_m2m_data/_stats_global_rot',
    rotation_space='global',
)

# Must override full pipeline (MMEngine list merge = replace) to insert
# LocalToGlobalRotation after LoadSmplx55.
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
                type='PrepareM2MUniversalMask',
                key='motion',
                strategy_weights=dict(
                    m1_random_cell=0.25,
                    m2_random_block=0.15,
                    m3_temporal_contiguous=0.25,
                    m4_joint_contiguous=0.15,
                    m5_full_mask=0.05,
                    m6_keyframe_sparse=0.15,
                ),
                min_mask_ratio=0.05,
                max_mask_ratio=0.95,
                edit_repair_prob=0.15,
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
