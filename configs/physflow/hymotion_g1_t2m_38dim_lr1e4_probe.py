# Short optimizer-reset probe for the G1-native HYMotion T2M generator.
#
# The long resume config restores the full accelerator state, including the
# old AdamW learning rate/state.  This probe intentionally loads only model
# weights from the latest sane checkpoint and rebuilds the optimizer at the
# original HYMotion T2M fine-tune LR to test whether the 2e-5 continuation is
# simply too conservative for the reinitialised 38-D G1 head.

_base_ = 'hymotion_g1_t2m_38dim.py'

work_dir = 'work_dirs/hymotion_g1_t2m_38dim_lr1e4_probe'

auto_resume = False

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=10000,
    val_interval=100000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        max_keep_ckpts=4,
        save_last=True,
    ),
)

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_g1_t2m_38dim/checkpoint-iter_27000',
    load_scope='model',
)
