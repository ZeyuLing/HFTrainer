"""HYMotion-G1 generator + Any2Track reward from the frozen 130k base."""

_base_ = "verify_hymotion_g1_protomotions_130k_safe.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_any2track_130k_safe"

trainer = dict(
    judge_backend="any2track",
    any2track_input_fps=30,
    # Score the full 300-frame clip. The ckpt99000 pilot used 150 frames, which
    # can miss late failures and over-reward partial-trackable motions.
    any2track_max_steps=300,
    tracker_pool_dir=None,
)
