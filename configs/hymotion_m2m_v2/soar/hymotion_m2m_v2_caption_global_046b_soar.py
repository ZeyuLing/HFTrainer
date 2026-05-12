# SOAR post-training on top of hymotion_m2m_v2_caption_global_046b.
# Latest SFT checkpoint as of 2026-04-17: epoch_548.

_base_ = '../hymotion_m2m_v2_caption_global_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_caption_global_046b_soar'

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

optimizer = dict(
    type='AdamW',
    lr=2e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_dataloader = dict(
    batch_size=10,
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=5000,
    val_interval=100000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=500, by_epoch=False,
                    max_keep_ckpts=5, save_last=True),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_global_046b/checkpoint-epoch_548',
    load_scope='model',
    # B2-ext fix: intermediate checkpoints have all-zero null embeddings.
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)
