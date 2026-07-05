# HyMotion M2M v2 -- SMPL-root full continuation from epoch 1990.
#
# This inherits the current official-T2M + PerMo + MotionFix mixed-task recipe
# from hymotion_m2m_smpl_caption_046b.py, then warm-starts the model weights
# from the latest SMPL-root full checkpoint produced by the older run.  The
# load is intentionally model-only so optimizer/scheduler state from the old
# dump config does not override the repaired training strategy.

_base_ = './hymotion_m2m_smpl_caption_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_official_t2m_permo_motionfix_mix_from1990_20260706'

model = dict(
    # The parent config can initialize directly from HY-Motion-1.0-Lite.  For
    # this continuation run, the full M2M checkpoint below already carries the
    # adapted M2M weights, so avoid loading T2M-Lite first and then overwriting.
    t2m_pretrained_path=None,
)

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_smpl_caption_editfix_from870_20260528/checkpoint-epoch_1990',
    load_scope='model',
    exclude_bundle_keys=['mean', 'std'],
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)
