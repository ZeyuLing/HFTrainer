# HyMotion M2M 0.46B with EnsureDimensionFilter to fix batch contamination
_base_ = ['./_base_hymotion_m2m_151dim_046b.py']

work_dir = 'work_dirs/hymotion_m2m_completion_151dim_uncond_fm_046b_with_filter'

# Reduce batch size to avoid OOM on T4 GPU
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    shuffle=True,
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        task_mode='auto',
        num_person=1,
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            # Compute 147-dim by appending end-effector positions
            dict(
                type='Compute147DimEndEffector',
                key='motion',
            ),
            # Compute 151-dim by appending foot contact binary indicators
            dict(
                type='Compute151DimFootContact',
                key='motion',
                bone_offsets_dir='data/bone_offsets',
                velocity_threshold=0.002,  # 0.002 m/frame threshold for contact detection
            ),
            # Filter out any samples that don't have exactly 151-dim
            # This removes contamination from 198-dim or other representations
            dict(
                type='EnsureDimensionFilter',
                key='motion',
                expected_dim=151,
            ),
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
        verbose=True,
        refetch=True,
    ),
)
