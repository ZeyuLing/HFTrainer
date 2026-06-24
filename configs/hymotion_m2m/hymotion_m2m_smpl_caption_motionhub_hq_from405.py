"""HYMotion M2M SMPL caption training with PRISM/HYMotion-T2M data source.

This isolates the data-source hypothesis for poor pure-T2M evaluation:
continue from the latest no-edit specialist checkpoint, but replace the
M2M-only ``train_hymotion_400h_hq_20260403.json`` source with the same merged
high-quality source used by PRISM / HYMotion-T2M:

    data/annotation/train_hq_motionhub_hymotion.json

That annotation contains HYMotion data plus the high-quality MotionHub subset
(including HumanML3D-like entries), so its caption style is much closer to the
HumanML3D T2M evaluator than the simple-caption-only M2M data.

Launch:
    python3 tools/taiji_submit.py m2m_smpl_motionhub_hq_from405 \
        configs/hymotion_m2m/hymotion_m2m_smpl_caption_motionhub_hq_from405.py \
        --host_num 8
"""

_base_ = './hymotion_m2m_smpl_caption_cleandata_ablation.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_motionhub_hq_from405'

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_smpl_caption_cleandata_ablation/checkpoint-epoch_405',
    load_scope='model',
)

train_dataloader = dict(
    dataset=dict(
        anno_file='data/annotation/train_hq_motionhub_hymotion.json',
    ),
)
