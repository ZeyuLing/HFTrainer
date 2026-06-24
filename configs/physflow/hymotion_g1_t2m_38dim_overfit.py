# Overfit sanity check for the G1-native T2M fine-tune: train AND infer on the
# SAME 100 clips. If the pipeline + 38-d representation are correct, the model
# should memorize each (caption -> G1 motion) pair and regenerate motions nearly
# identical to GT. Run on 1x8 V100.

_base_ = 'hymotion_g1_t2m_38dim.py'

work_dir = 'work_dirs/hymotion_g1_t2m_38dim_overfit_vel'

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        anno_file='data/annotation/train_g1_t2m_overfit100.json',
        # deterministic text variant -> clean (caption -> motion) memorization
        random_caption=False,
    ),
)

# slightly higher lr so 100 clips overfit quickly
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_cfg = dict(
    by_epoch=False,
    max_iters=6000,
    val_interval=100000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=20),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000,
                    max_keep_ckpts=3, save_last=True),
)
