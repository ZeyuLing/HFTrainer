# HyMotion M2M v3 (DSCF) — Unconditioned (motion-only) + Global Rotation.
#
# DSCF architecture without text conditioning (uncondition_mode=True).
# All text inputs are forced to null embeddings during training.
# This variant trains the motion condition pathway exclusively.
#
# Use case: motion completion, inpainting, interpolation — no text guidance.
# After training, can be combined with caption model via dual-CFG:
#   v_guided = v_uncond + s_text*(v_text - v_uncond) + s_cond*(v_full - v_text)
#
# Launch:
#   python tools/train.py configs/hymotion_m2m_v3/hymotion_m2m_v3_uncond_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m_v3/hymotion_m2m_v3_uncond_046b.py 8

_base_ = './_base_hymotion_m2m_v3_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v3_uncond_046b'

model = dict(
    pred_type='velocity',
    uncondition_mode=True,
    cond_mask_prob=0.0,
)

# Unconditioned pipeline: no text loading, only motion
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            dict(type='Compute198DimPosition', key='motion'),  # 135 -> 198
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
                corruptor_names=['jitter', 'joint_jump', 'sliding',
                                 'limb_candy_wrapper', 'wrist_candy_wrapper'],
                max_corruptions=2,
            ),
            dict(
                type='PackInputs',
                keys=['src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length', 'edit_mode'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
    ),
)
