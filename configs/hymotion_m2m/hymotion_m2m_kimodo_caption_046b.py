# HyMotion M2M v2 -- KIMODO-root mixed-task training.
#
# Canonical KIMODO caption config.  This intentionally includes the same real
# paired editing data as the SMPL-root full config (PerMo + MotionFix), while
# keeping the KIMODO root representation:
#
#   [0:3]      ADMM-smoothed pelvis translation
#   [3:135]    22 joints x 6D local rotations
#   [135:198]  21 body-joint position channels in the adjusted root frame
#
# The older caption-only KIMODO branch is no longer maintained as the default:
# KIMODO caption training should exercise text-only generation and real
# motion-conditioned editing in one stream.

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_permo_motionfix_mix_20260706'

# Warm-start from the latest usable KIMODO-root checkpoint, but reset optimizer
# state so the fixed PerMo/MotionFix editing stream and current loss config
# define the new training phase.
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4/checkpoint-epoch_890/',
    load_scope='model',
    exclude_bundle_keys=['mean', 'std'],
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    motion_cond_mask_prob=0.0,
    enable_special_game_feat=True,
    train_null_embeddings=True,
    train_special_game_embeddings=True,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',
    rotation_space='local',
    caption_freeze_strategy='encoders',
    text_encoder=dict(),
    losses_cfg=dict(
        keypoints3d_weight=10.0,
        velocity_loss_reduction='component_mean',
        spike_downweight_enabled=False,
    ),
)

train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        # 400h caption data plus real PerMo/MotionFix instruction editing pairs.
        anno_file='data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260704_permo_pathfix.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
            dict(
                type='LoadPreExtractedTextEmbedding',
                key='caption',
                allow_none=True,
                text_emb_augment_dir='qwen3_augmented',
            ),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            dict(type='Compute198DimPosition', key='motion'),
            dict(
                type='SmplTransToKimodoRootOnline',
                key='motion',
                admm_margin_m=0.06,
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
                type='PrepareM2Mv2Condition',
                key='motion',
                sampler_version='v3',
                editing_prob=0.0,
                corruptor_names=[],
            ),
            dict(
                type='LoadEditingSourceMotion',
                kimodo_root_cfg=dict(admm_margin_m=0.06),
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
