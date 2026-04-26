# HyMotion M2M v2 — Caption-conditioned + Global rotation.
#
# Uses pre-extracted Qwen3+CLIP embeddings. cond_mask_prob=0.3 for CFG.
# Global (world-frame) rotation with 198-dim global-rot stats.
#
# V100 32GB: B=24 (text tokens add 128 extra attention tokens).
#
# Launch:
#   python tools/train.py configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_046b.py 8

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_caption_global_046b'

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_global_rot',
    rotation_space='global',
)

train_dataloader = dict(
    batch_size=20,  # V100-32GB: text tokens (128×4096) add ~6GB; bs=20 peak ~30GB
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
                tier2_prob=0.4,
                editing_prob=0.15,
                corruptor_names=[
                    'jitter', 'joint_jump', 'sliding',
                    'limb_candy_wrapper', 'wrist_candy_wrapper',
                ],
                max_corruptions=2,
                tier2_weights={
                    'pure_gen': 0.40,       # T2M: 40% of Tier2 = 16% global (was 8%)
                    'inbetween': 0.15,
                    'prefix': 0.10,
                    'keyframes': 0.10,
                    'end_effector': 0.08,
                    'trajectory': 0.07,
                    'foot_ground': 0.05,
                    'edit_repair': 0.05,
                },
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
