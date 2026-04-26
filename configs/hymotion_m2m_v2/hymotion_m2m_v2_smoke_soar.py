# SOAR smoke test: reuses the v2 smoke model/data but swaps in the SOAR
# post-trainer. Purpose: verify end-to-end SOAR gradient flow on a 0.37M
# param tiny model without needing the full 0.46B checkpoint.
#
# Launch:
#   python tools/train.py configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke_soar.py

_base_ = './hymotion_m2m_v2_smoke.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smoke_soar'

trainer = dict(
    type='HyMotionM2MSoarTrainer',
    val_num_steps=2,
    mask_aware_noise=True,
    # SOAR hyperparameters (minimal settings per plan §5.1 recommendation)
    soar_lambda=0.1,
    soar_num_aux=1,
    soar_K=50,
    soar_cfg_scale=1.0,
    soar_sigma_clamp=0.05,
)

# Shorten iters for the smoke test
train_cfg = dict(
    by_epoch=False,
    max_iters=20,
    val_interval=10000,
)
