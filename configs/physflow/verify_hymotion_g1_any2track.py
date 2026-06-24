"""Pilot verification: HYMotion-G1 generator + Any2Track judge."""

_base_ = "verify_hymotion_g1_protomotions.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_any2track"

trainer = dict(
    judge_backend="any2track",
    any2track_input_fps=30,
    # Keep pilot scoring bounded; full metrics can remove this cap.
    any2track_max_steps=150,
    tracker_pool_dir=None,
)
