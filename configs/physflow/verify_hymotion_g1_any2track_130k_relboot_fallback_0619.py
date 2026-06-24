_base_ = "verify_hymotion_g1_any2track_130k_relboot_0619.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_any2track_130k_relboot_fallback_0619"
auto_resume = True

trainer = dict(
    # Any2Track rejects early samples by absolute fall/score gates even when
    # some same-noise candidates improve over the frozen base. Use relative-only
    # soft fallback for bootstrap signal; strict hard metrics stay logged.
    accept_soft_fallback=True,
    accept_soft_fallback_require_relative=True,
    relative_min_advantage=-0.01,
)
