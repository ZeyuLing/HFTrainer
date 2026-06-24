"""Pilot verification: HYMotion-G1 generator + ProtoMotions judge.

Goal: test whether online tracker feedback improves the generator's physical
trackability versus the same HYMotion-G1 start checkpoint.
"""

_base_ = "physflow_online_adv_g1_38dim.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_protomotions"

trainer = dict(
    judge_backend="protomotions",
    num_samples=4,
    diffusion_steps=30,
    anchor_weight=1.0,
    gt_weight=0.5,
    tracker_pool_dir=None,
    export_gt_to_pool=False,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(max_items=512),
)

optimizer = dict(type="AdamW", lr=5e-6, betas=[0.9, 0.99], weight_decay=0.0)

accelerator = dict(mixed_precision="no", gradient_accumulation_steps=4)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=400,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=10),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=100, max_keep_ckpts=6, save_last=True),
)

load_from = dict(
    _delete_=True,
    path="work_dirs/hymotion_g1_t2m_38dim/checkpoint-iter_99000",
    load_scope="model",
)
