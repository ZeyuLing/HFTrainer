"""HYMotion-G1 generator + HumanoidGPT judge config.

HumanoidGPT currently ships inference/deployment code and released ONNX only;
no full tracker training entrypoint is available in the local release. This
config therefore runs the generator half and exports qpos replay for later
tracker training once a trainable HumanoidGPT stack is integrated.
"""

_base_ = "verify_hymotion_g1_humanoidgpt_130k_safe.py"

work_dir = "work_dirs/physflow_coevo_humanoidgpt_g1"

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(max_items=None),
)

trainer = dict(
    judge_backend="hgpt",
    hgpt_freq=50,
    hgpt_input_fps=30,
    num_samples=4,
    diffusion_steps=30,
    anchor_weight=1.5,
    gt_weight=0.75,
    frontier_mode=True,
    sft_target="easiest",
    gt_pool_accept_mode="kinematic",
    tracker_pool_dir=None,
    tracker_qpos_pool_dir="work_dirs/physflow_coevo_humanoidgpt_g1/qpos_pool",
    tracker_qpos_pool_fps=30.0,
    export_gt_to_pool=True,
    gt_pool_freq=2,
    pool_max_motions=8000,
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=300,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=10),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=100,
        max_keep_ckpts=4,
        save_last=True,
    ),
)
