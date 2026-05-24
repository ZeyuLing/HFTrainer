# HyMotion M2M 0.46B — 151-dim with SINGLE-AGENT dataset (no multi-task complexity)
#
# This config bypasses the multi-task dataset to eliminate the 198-dim sample source.
# Uses MotionHubSingleAgentDataset with fixed 151-dim pipeline.
#
# Motion representation: smpl_22 with rotation_6d + end-effector positions + foot contact
# Layout: [3D translation, 132D rot6d (22×6), 12D end-effector positions (4 joints × 3), 4D foot contact (4 joints × 1)]
# Total dims: 151 (3 + 132 + 12 + 4)

_base_ = '_base_hymotion_m2m_151dim_046b.py'

# Override work_dir for this variant
work_dir = 'work_dirs/hymotion_m2m_completion_151dim_single_agent_046b'

# ----- Dataset: Use Single-Agent instead of Multi-Task -----
# This removes the task_mode='auto' which assigns tasks per-sample and causes dimensional inconsistency
_motion_dim = 151

data_loader = dict(
    train=dict(
        batch_size=8,  # Reduced from 32 to avoid OOM on T4 GPU
        num_workers=4,
        dataset=dict(
            type='MotionHubSingleAgentDataset',  # Changed from MotionhubMultiTaskMultiAgentDataset
            json_path='data/annotation/train_hymotion_400h_hq_20260403.json',
            dataset_name='HyMotion400h-HQ',
            motion_root='data',
            cache_dir='cache',
            max_motion_len=360,
            sampling_strategy='uniform',
            fps_scaling_range=(0.5, 2.0),
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
)
