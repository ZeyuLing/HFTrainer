# Eval-only config: run the REAL kimodo_caption_permo_resume model (E4 / ep890)
# through the deterministic overfit-100 all-tasks set, in the KIMODO-Root
# representation the model was trained on, so motion_annot_web/overfit_viewer
# can show GT-vs-Pred per task.
#
# Why a dedicated config
# ----------------------
# * The model expects KIMODO-Root 198-dim (ADMM-smoothed pelvis + adjusted
#   positions). The stock overfit-100 config uses plain SMPL-Root + _stats_198dim,
#   which would mismatch the model's normalization and representation.
# * We therefore inherit the kimodo resume config (correct model: healthy T2M
#   text encoders restored + frozen, KIMODO-root mean/std) and override ONLY the
#   dataloader with the deterministic per-task overfit-100 pipeline, inserting the
#   same SmplTransToKimodoRootOnline conversion used in kimodo training.
#
# Run (single GPU is plenty for 100 samples):
#   python3 scripts/eval/eval_m2m_v2_overfit100_alltasks.py \
#     --config configs/hymotion_m2m/eval_kimodo_resume_overfit100.py \
#     --checkpoint work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4/checkpoint-epoch_890 \
#     --output-dir output/evaluation/m2m_kimodo_resume_E4_overfit100/ep890 \
#     --max-samples 100 --save-npz

_base_ = './hymotion_m2m_kimodo_caption_permo_resume_046b.py'

# Keep the work_dir so --checkpoint=auto can also resolve the latest ckpt here.
work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4'

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        # Deterministic 100-case all-tasks set (carries overfit_task /
        # overfit_source_key meta used by the viewer for per-task grouping).
        anno_file='data/annotation/overfit_100_m2m_v2_all_tasks_20260528.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
            dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            # Compute198DimPosition MUST come before SmplTransToKimodoRootOnline.
            dict(type='Compute198DimPosition', key='motion'),
            # KIMODO Root conversion — matches the kimodo model's training repr.
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
            # Deterministic per-task mask/condition construction (text_only,
            # text_frame, text_upper, ... ) keyed off overfit_task.
            dict(type='PrepareM2Mv2OverfitCase', key='motion'),
            # Real PerMo/MotionFix editing source, converted to KIMODO root too.
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
                meta_keys=['motion_path', 'fps', 'overfit_task', 'overfit_source_key'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)
