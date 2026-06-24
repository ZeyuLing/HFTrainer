_base_ = "verify_hymotion_g1_any2track_130k_relgate_0618.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_any2track_130k_relboot_0619"
auto_resume = True

trainer = dict(
    # Start from the frozen 130k generator, so strict positive margins can starve
    # reward-SFT before the model has a chance to move. This keeps the same-noise
    # base comparison but selects/weights accepted targets by net advantage.
    relative_mode="advantage",
    relative_min_advantage=-0.005,
    relative_select_by_advantage=True,
    relative_weight_by_advantage=True,
    relative_advantage_weight_scale=1.5,
    relative_advantage_weight_max=3.0,
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=6000,
    val_interval=999999,
    max_grad_norm=1.0,
)
