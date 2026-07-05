# HyMotion M2M v2 -- KIMODO-root full training.
#
# This config is intentionally aligned with hymotion_m2m_smpl_caption_046b.py:
# same HY-Motion official T2M stream, same PerMo/MotionFix editing streams,
# same 50/25/25 sampling mix, same text embedding policy, same special-token
# behavior, same losses, and same HY-Motion-1.0-Lite initialization.
#
# The only representation-level difference is the root trajectory convention:
# after the shared SMPL/O6DP load + 198-dim position construction, the motion is
# converted to KIMODO root by smoothing XZ translation and compensating position
# channels.  Editing source motions receive the same conversion.

_base_ = './hymotion_m2m_smpl_caption_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_official_t2m_permo_motionfix_mix_20260706'

_kimodo_root_cfg = dict(admm_margin_m=0.06)

_pack_keys = [
    'src_motion', 'tgt_motion', 'src_mask',
    'tgt_length', 'src_length', 'edit_mode',
    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
    'data_src', 'source', 'input_filename', 'text_emb_dir',
    'text_source_type', 'motion_path', 'caption_path', 'fps', 'caption',
]

model = dict(
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',
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
                        type='SmplTransToKimodoRootOnline',
                        key='motion',
                        **_kimodo_root_cfg,
                    ),
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
                        type='SmplTransToKimodoRootOnline',
                        key='motion',
                        **_kimodo_root_cfg,
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
                        v3_config=dict(traj_control_prob=0.12),
                        corruptor_names=[],
                    ),
                    dict(type='LoadEditingSourceMotion', kimodo_root_cfg=_kimodo_root_cfg),
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
                        type='SmplTransToKimodoRootOnline',
                        key='motion',
                        **_kimodo_root_cfg,
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
                        v3_config=dict(traj_control_prob=0.12),
                        corruptor_names=[],
                    ),
                    dict(type='LoadEditingSourceMotion', kimodo_root_cfg=_kimodo_root_cfg),
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
