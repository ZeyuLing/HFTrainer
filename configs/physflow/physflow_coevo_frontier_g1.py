# PhysFlow LEARNABILITY-FRONTIER co-evolution (G1-native generator).
#
# This is the "direction B" redesign. The earlier co-evolution loop exported the
# EASIEST-to-track generations to the trainee pool (best-of-N picked the minimum
# tracking error), so the trainee only ever co-trained on motions it ALREADY
# solved -> no learning signal -> the tracker could not improve (and, mixed with
# slow GT, degraded). Here we DECOUPLE two judges and chase the regret frontier:
#
#   Q (quality_judge="frozen")  : the strong released G1 tracker. Certifies a
#       generation is physically valid / a competent robot CAN execute it.
#   T (trainee_judge="trainee") : the policy being improved. Measures the
#       motion's CURRENT difficulty for the trainee.
#
#   * SFT target  = regret-max VALID candidate (Q-trackable, hardest for T) so the
#     generator actively explores the trainee's failure frontier rather than
#     collapsing onto the trainee's comfort zone.
#   * Trainee pool = every FRONTIER candidate (Q-valid AND T struggles but does
#     not catastrophically fail: completion_T in (t_low, t_high)). These are the
#     motions that actually teach the trainee something new.
#   * Strict quality gate: Q-no-fall + completion + kinematic-velocity artifact
#     limits on the reference itself (impossible joint/root speeds rejected so we
#     never teach the trainee garbage).
#
# REQUIRED orchestrator flags (so BOTH judges are always in the ensemble and the
# trainer can read the per-judge breakdown):
#   --gen-config configs/physflow/physflow_coevo_frontier_g1.py
#   --judge-mode anchor  --anchor-alpha 0.8
# anchor-alpha only weights the *combined* score used as the round-0 cold-start
# fallback (no trainee yet); the frontier logic reads per_judge["frozen"/"trainee"]
# directly so the alpha value is not critical.

_base_ = 'physflow_online_adv_g1_38dim.py'

work_dir = 'work_dirs/physflow_coevo_frontier_g1'

# Full diverse prompt bank MINUS the held-out agile eval clips.
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(anno_file='data/annotation/train_g1_t2m_emb_minus_heldout.json'),
)

optimizer = dict(type='AdamW', lr=5e-6, betas=[0.9, 0.99], weight_decay=0.0)

trainer = dict(
    # ---- frontier (direction B) mechanism ----
    frontier_mode=True,
    quality_judge='frozen',
    trainee_judge='trainee',
    sft_target='regret',
    # trainee-completion band for the pool: motions T solves (>=0.9) carry no
    # learning signal; motions T cannot do at all (<0.2) are unlearnable / likely
    # off-distribution -> we keep only the (0.2, 0.9) learnable frontier.
    frontier_t_low=0.2,
    frontier_t_high=0.9,
    # ---- strict quality gate ----
    accept_min_completion=0.9,     # Q must track >=90% before a motion is "valid"
    accept_require_no_fall=True,
    accept_max_score=None,         # difficulty is no longer a validity criterion
    accept_min_joint_std=0.05,     # anti frozen-pose
    accept_max_root_disp_if_frozen=1.0,
    accept_frozen_joint_std=0.03,
    # kinematic artifact limits on the generated reference (generous; the frozen
    # judge no-fall is the primary physical certificate, these only catch gross
    # numerical artifacts). Calibrate against real-G1 stats if over-filtering.
    quality_max_joint_vel=30.0,    # rad/s peak per-joint finite-diff
    quality_max_root_vel=8.0,      # m/s peak root finite-diff
    # ---- trainee co-training pool ----
    export_gt_to_pool=True,
    gt_pool_freq=2,
    pool_max_motions=8000,
    tracker_pool_dir='work_dirs/physflow_coevo_frontier_g1/tracker_motion_pool',
)
