# SOAR post-training on top of hymotion_m2m_v2_uncond_local_046b.
#
# Loads the latest SFT checkpoint (epoch_485 as of 2026-04-17) and applies
# SOAR correction loss to reduce exposure bias in generated regions. See
# docs/temp/soar_m2m_v2_post_training_plan.md for the full method.
#
# Why uncond_local first:
#   - No text encoder → simplest SOAR rollout (cfg_scale=1.0, reuse v_pred)
#   - Lowest VRAM (no text tokens) → easiest to fit SOAR's extra forward
#   - Local rotation → no LocalToGlobalRotation pipeline step
#
# Launch:
#   bash tools/dist_train.sh configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_uncond_local_046b_soar.py 8
#   python tools/taiji_submit.py m2m_v2_uncond_local_soar configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_uncond_local_046b_soar.py --host_num 8

_base_ = '../hymotion_m2m_v2_uncond_local_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_uncond_local_046b_soar'

# ----- Trainer: swap in SOAR post-trainer -----
trainer = dict(
    type='HyMotionM2MSoarTrainer',
    val_num_steps=10,
    mask_aware_noise=True,
    soar_lambda=0.1,
    soar_num_aux=1,
    soar_K=50,
    soar_cfg_scale=1.0,
    soar_sigma_clamp=0.05,
)

# ----- Post-training optimisation: smaller LR, fewer steps -----
optimizer = dict(
    type='AdamW',
    lr=2e-5,               # 5x smaller than SFT 1e-4 (plan §5.2)
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

# ----- Batch size: SOAR uses ~2x forward. Halve from SFT batch to fit VRAM. -----
train_dataloader = dict(
    batch_size=14,         # SFT uncond was 28; SOAR post-training halves to 14
)

# ----- Training schedule: 5K iter post-training, no epoch concept -----
train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=5000,
    val_interval=100000,
    max_grad_norm=1.0,
)

# ----- Checkpoint more often (post-training is short) -----
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=500, by_epoch=False,
                    max_keep_ckpts=5, save_last=True),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)

# ----- Load from latest SFT checkpoint (NOT T2M pretrained) -----
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_485',
    load_scope='model',     # weights only; reset optimizer/step
)
