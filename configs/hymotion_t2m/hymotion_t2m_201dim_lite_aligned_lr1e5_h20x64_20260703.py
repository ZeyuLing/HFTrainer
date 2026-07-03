# Restart HY-Motion 1.0 Lite from the released checkpoint after aligning the
# in-repo data path/text-feature/model-bundle differences with the official
# trainer. This keeps the 20260702 run intact for direct comparison.

_base_ = './hymotion_t2m_201dim_lite_h20x64_20260702.py'

work_dir = 'work_dirs/hymotion_t2m_201dim_lite_aligned_lr1e5_h20x64_20260703'

optimizer = dict(
    type='Adam',
    lr=1e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
    foreach=True,
)
