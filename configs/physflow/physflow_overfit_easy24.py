# PhysFlow OVERFIT — DECISIVE implementation-correctness test on an EASY subset.
#
# The 100-prompt overfit (physflow_overfit100, anchor=1.0) was STABLE (no
# freeze-collapse) but PLATEAUED at n_good~0.5 / reward_best~1.7. That plateau
# has two confounded causes: (a) the frozen judge tracker (g1-bones-deploy,
# trained on bones-seed locomotion) simply cannot track some of the diverse /
# agile HumanML3D prompts, and (b) the anchor needed to avoid collapse caps how
# far the policy can specialize.
#
# This config removes confounder (a): 24 PURE-LOCOMOTION prompts (walk / run /
# step / jog / pace forward-backward-circle) that the frozen tracker has ample
# headroom to track. If the implementation is correct, the loop should now
# OVERFIT cleanly -> n_good rises toward ~1.0, reward_best drops well below the
# ~1.7 diverse-set plateau, joint_std stays healthy. If it ALSO plateaus here,
# the limitation is the method/optimization rather than the tracker ceiling.

_base_ = './physflow_overfit100.py'  # inherits anchor_weight=1.0, lr=1e-5, accept gate

work_dir = 'work_dirs/physflow_overfit_easy24'

train_dataloader = dict(
    dataset=dict(
        corpus_file='configs/experiments/physflow_kimodo_g1/prompt_bank_locomotion_easy24.jsonl',
        feature_dir='data/kimodo_text_feature/kimodo_g1_llm2vec_locomotion_easy24',
        split='train',
        max_samples=24,
        min_frames=60,
        max_frames=120,
    ),
)

trainer = dict(
    tracker_pool_dir='work_dirs/physflow_overfit_easy24/tracker_motion_pool',
    pool_max_motions=300,
)

# 24 prompts x 8 GPUs ~= 3 steps/epoch; 1200 iters ~= 400 epochs (plenty to
# overfit a tiny easy set).
train_cfg = dict(
    by_epoch=False,
    max_iters=1200,
    val_interval=999999,
    max_grad_norm=1.0,
)
