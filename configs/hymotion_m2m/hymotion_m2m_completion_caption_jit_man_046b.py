# HyMotion M2M 0.46B — Completion (caption-conditioned) + JiT + Mask-Aware Noise (V4 ablation).
#
# Based on caption_jit baseline, with mask_aware_noise=True.
# See uncond_fm_man for detailed description of the mask-aware noise mechanism.
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_man_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_man_046b.py 8
#   bash tools/taiji_dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_man_046b.py

_base_ = './hymotion_m2m_completion_caption_jit_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_caption_jit_man_046b'

trainer = dict(
    mask_aware_noise=True,
)
