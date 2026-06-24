"""HYMotion-G1 generator + HumanoidGPT reward from the frozen 130k base."""

_base_ = "verify_hymotion_g1_protomotions_130k_safe.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_humanoidgpt_130k_safe"

trainer = dict(
    judge_backend="hgpt",
    hgpt_freq=50,
    hgpt_input_fps=30,
    tracker_pool_dir=None,
)
