# HyMotion M2M v2 -- official-T2M aligned mixed-task training.
#
# This is the formal M2M phase after the aligned T2M-only M2M checkpoint is
# selected.  It keeps the robust rank-k / v3 mask sampler for arbitrary
# motion conditions, but pins pure T2M to the verified official HYMotion SFT
# data stream instead of the older 400h HQ JSON path.
#
# Target mix per sampled step:
#   official_t2m       50%
#   MotionFix edit     25%
#   PerMo edit         25%
#
# The weighted-sampler fractions intentionally sum to 1.0 so generic
# MotionHub generation records get zero probability.  The full M2M phase keeps
# pure text-to-motion frequent enough to preserve text following, while the
# remaining half exercises paired motion-conditioned editing.

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_strict198_official_t2m_permo_motionfix_mix_20260706'

_pack_keys = [
    'src_motion', 'tgt_motion', 'src_mask',
    'tgt_length', 'src_length', 'edit_mode',
    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
    'data_src', 'source', 'input_filename', 'text_emb_dir',
    'text_source_type', 'motion_path', 'caption_path', 'fps', 'caption',
]

# Start formal M2M directly from HY-Motion-1.0-Lite through the M2M bundle
# adapter.  Do not use the generic load_from path here: the M2M input/output
# projections use strict 201->198 pelvis-RIC-drop/channel-expanded adaptation
# and the learned null embeddings must be preserved.
load_from = None

model = dict(
    t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    t2m_freeze_strategy='none',
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    motion_cond_mask_prob=0.0,
    enable_special_game_feat=True,
    train_null_embeddings=True,
    train_special_game_embeddings=True,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
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
    _delete_=True,
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    weighted_sampler=dict(groups=[
        dict(name='official_t2m', match=['official_t2m'], frac=0.50),
        dict(name='motionfix', match=['MotionFix'], frac=0.25),
        dict(name='permo_edit', match=['PerMo-editing'], frac=0.25),
    ]),
    dataset=dict(
        type='MotionDatasetUnion',
        subset_prefixes=['official_t2m', 'permo', 'motionfix'],
        datasets=[
            dict(
                type='HYMotionOfficialT2MDataset',
                data_root='data/hymotion_data',
                input_record_file_dir='_input_record_files/sft_train_v1103_qwen3',
                motion_dir='motions_o6dp_v0922',
                motion_postfix='npy',
                require_motion_file=False,
                pipeline=[
                    dict(
                        type='LoadPreExtractedTextEmbedding',
                        key='caption',
                        allow_none=False,
                        text_emb_augment_dir='qwen3_augmented',
                        refetch_on_missing=True,
                        raw_text_prob=0.5,
                    ),
                    dict(type='LoadO6dp', key='motion', joints_num=22, transl_aug_prob=0.0),
                    dict(type='Compute198DimPosition', key='motion'),
                    dict(
                        type='CropMotionByTextTime',
                        keys='motion',
                        fps_key='fps',
                        min_frame=10,
                        max_frame=360,
                    ),
                    dict(
                        type='RandomCropPadding',
                        clip_len=360,
                        pad_mode='replicate',
                        allow_shorter=True,
                        allow_longer=False,
                        make_pad_mask=True,
                        pad_mask_key='pad_mask',
                    ),
                    dict(type='PrepareM2Mv2FullMask', key='motion'),
                    dict(
                        type='PackInputs',
                        keys=_pack_keys,
                        meta_keys=[],
                        set_dummy_value=True,
                        dummy_value=None,
                    ),
                ],
                refetch=True,
                max_refetch=100,
                verbose=True,
            ),
            dict(
                type='MotionhubMultiTaskMultiAgentDataset',
                motion_key='smplx',
                data_dir='data/motionhub',
                anno_file='data/annotation/permo_editing_train_smplh52_20260705.json',
                task_mode='auto',
                num_person=1,
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
                        v3_config=dict(traj_control_prob=0.12),
                        corruptor_names=[],
                    ),
                    dict(type='LoadEditingSourceMotion'),
                    dict(
                        type='PackInputs',
                        keys=_pack_keys,
                        meta_keys=[],
                        set_dummy_value=True,
                        dummy_value=None,
                    ),
                ],
                verbose=True,
                refetch=True,
            ),
            dict(
                type='MotionhubMultiTaskMultiAgentDataset',
                motion_key='smplx',
                data_dir='data',
                anno_file='data/MotionFix/motionfix_train.json',
                task_mode='auto',
                num_person=1,
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
                        v3_config=dict(traj_control_prob=0.12),
                        corruptor_names=[],
                    ),
                    dict(type='LoadEditingSourceMotion'),
                    dict(
                        type='PackInputs',
                        keys=_pack_keys,
                        meta_keys=[],
                        set_dummy_value=True,
                        dummy_value=None,
                    ),
                ],
                verbose=True,
                refetch=True,
            ),
        ],
    ),
)
