"""Pilot verification: HYMotion-G1 generator + HumanoidGPT judge."""

_base_ = "verify_hymotion_g1_protomotions.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_humanoidgpt_headless"

trainer = dict(
    judge_backend="hgpt",
    hgpt_freq=50,
    hgpt_input_fps=30,
    tracker_pool_dir=None,
)
