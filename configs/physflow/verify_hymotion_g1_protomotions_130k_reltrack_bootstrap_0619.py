_base_ = "verify_hymotion_g1_protomotions_130k_reltrack_0619.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_protomotions_130k_reltrack_bootstrap_0619"
auto_resume = True

trainer = dict(
    # The strict reltrack run started from exactly the frozen base model, so the
    # same-noise candidate/base advantage was numerically zero and produced no
    # reward-SFT signal. Bootstrap accepts hard-trackable candidates that are not
    # materially worse than base, while still selecting/weighting by advantage.
    relative_min_advantage=-0.005,
    relative_select_by_advantage=True,
    relative_weight_by_advantage=True,
    relative_advantage_weight_scale=1.5,
    relative_advantage_weight_max=3.0,
)
