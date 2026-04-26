# Ablation T2 Phase 1: Curriculum Training — Phase 1 (pure T2M)
# 验证：KIMODO 两阶段 curriculum 训练是否优于直接混合训练。
# Phase 1: 20 epoch 纯 M5=1.0（unconditional generation）
# Phase 2: 在 Phase 1 ckpt 基础上续训 20 epoch（标准 M1-M6）
#
# 此 config 为 Phase 1。Phase 2 见 ablation_t2_curriculum_p2.py

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_t2_curriculum_p1'

train_cfg = dict(max_epochs=20)

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
                    m1_random_cell=0.0,
                    m2_random_block=0.0,
                    m3_temporal_contiguous=0.0,
                    m4_joint_contiguous=0.0,
                    m5_full_mask=1.0,  # Phase 1: pure T2M
                    m6_keyframe_sparse=0.0,
                ),
                min_mask_ratio=0.05,
                max_mask_ratio=0.95,
            ),
            dict(
                type='PackInputs',
                keys=['src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
    ),
)
