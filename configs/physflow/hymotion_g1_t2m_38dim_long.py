# Long continuation for the G1-native HYMotion T2M generator.
#
# The base config intentionally stopped at 40 epochs for an early warm-start
# probe.  This config resumes the same work_dir and leaves enough headroom for
# convergence checks instead of ending due to a short schedule.

_base_ = 'hymotion_g1_t2m_38dim.py'

work_dir = 'work_dirs/hymotion_g1_t2m_38dim'

auto_resume = True

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
        interval=1000,
        max_keep_ckpts=10,
        save_last=True,
    ),
)

# Safety fallback: if auto-resume cannot discover the latest work_dir checkpoint,
# fail over to the known completed checkpoint instead of restarting from the
# HY-Motion Lite warm-start.
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_g1_t2m_38dim/checkpoint-iter_25760',
    load_scope='full',
)
