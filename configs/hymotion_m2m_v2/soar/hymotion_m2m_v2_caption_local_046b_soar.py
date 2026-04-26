# SOAR post-training on top of hymotion_m2m_v2_caption_local_046b.
# Latest SFT checkpoint as of 2026-04-17: epoch_498.
#
# NOTE on CFG: the first-version SOAR trainer uses v_pred.detach() as the
# rollout velocity regardless of text conditioning (soar_cfg_scale=1.0). For
# a caption-conditioned model this is equivalent to "use the model's own
# conditional prediction as the rollout direction" — a reasonable baseline
# that matches plan §4.6 ("start with uncond, add CFG in ablation E6").

_base_ = '../hymotion_m2m_v2_caption_local_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_caption_local_046b_soar'

trainer = dict(
    type='HyMotionM2MSoarTrainer',
    val_num_steps=10,
    mask_aware_noise=True,
    soar_lambda=0.1,
    soar_num_aux=1,
    soar_K=50,
    soar_cfg_scale=1.0,     # v1: no CFG rollout. TODO: enable for ablation E6.
    soar_sigma_clamp=0.05,
)

optimizer = dict(
    type='AdamW',
    lr=2e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_dataloader = dict(
    batch_size=10,           # SFT caption was 20; halve for SOAR extra forward
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
    path='work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_498',
    load_scope='model',
)
