# Ablation M1: T2M Only (纯 text-to-motion, M5=100%)
# 验证：混合训练是否会降低 T2M 质量。此实验只做纯 unconditional generation。
#
# 改动：mask 策略 → M5=1.0（全部 mask=1，即纯 unconditional 生成）
# 对比：Baseline-M2M (M5=5%)

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_m1_t2m_only'

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
                    m5_full_mask=1.0,  # 100% T2M
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
