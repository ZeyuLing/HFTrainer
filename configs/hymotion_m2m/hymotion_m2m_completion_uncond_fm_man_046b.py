# HyMotion M2M 0.46B — Completion (unconditioned) + Mask-Aware Noise (V4 ablation).
#
# Ablation of mask-aware noise: known regions in x_t stay clean during training,
# making inference-time replacement guidance train-consistent.
#
# Difference from uncond_fm baseline:
#   - trainer.mask_aware_noise = True
#   - x_t[known] = x_clean (not noised), x_t[generate] = (1-t)*noise + t*x_clean
#   - Loss only computed on generation regions (src_mask=1)
#   - VACE inactive channel still provides known values (redundant but harmless)
#
# Expected behavior:
#   - Loss is higher than baseline (only counts hard generation regions)
#   - Inference replacement guidance becomes effective
#   - Boundary quality between known/generated regions should improve
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py 8
#   bash tools/taiji_dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py

_base_ = './hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_uncond_fm_man_046b'

trainer = dict(
    mask_aware_noise=True,
)
