# Ablation M3: T2M Heavy Mix (M5=50%, 其余各 10%)
# 验证：T2M 和 M2M 均衡训练是否能兼顾两种能力。
#
# 改动：mask 策略 → M5=50%, M1/M2/M3/M4/M6 各 10%
# 对比：M1 (100% T2M) vs Baseline-M2M (5% T2M) vs M3 (50% T2M)

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_m3_t2m_heavy'

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
                    m1_random_cell=0.10,
                    m2_random_block=0.10,
                    m3_temporal_contiguous=0.10,
                    m4_joint_contiguous=0.10,
                    m5_full_mask=0.50,  # 50% T2M
                    m6_keyframe_sparse=0.10,
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
