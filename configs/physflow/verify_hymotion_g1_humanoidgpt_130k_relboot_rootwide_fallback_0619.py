_base_ = "verify_hymotion_g1_humanoidgpt_130k_relboot_rootwide_0619.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_humanoidgpt_130k_relboot_rootwide_fallback_0619"
auto_resume = True

trainer = dict(
    # HumanoidGPT needs the same relative-only bootstrap path after root gates are
    # widened, otherwise early absolute gates can still starve reward-SFT.
    accept_soft_fallback=True,
    accept_soft_fallback_require_relative=True,
)
