# PhysFlow OVERFIT sanity check (Stage-1 RAFT, ~100 fixed prompts).
#
# Purpose: validate that the online best-of-N reward-SFT loop (generator +
# FROZEN judge tracker) is *correct and bug-free* by checking it can OVERFIT a
# tiny fixed prompt set -- i.e. on these 100 HumanML3D prompts the generator
# should drive the training-set physical-trackability reward down to near its
# achievable floor (reward_best_mean drops & saturates, completion->1, fall->0)
# WITHOUT articulation collapse (sel_joint_std_mean stays healthy, >~0.06).
# If the loop cannot even overfit 100 samples, the method/implementation has a
# real bug; reaching near-perfect on the train set is the prerequisite gate
# before scaling to the full co-evolution plan.
#
# Differences vs the formal v3 run (configs/physflow/physflow_online_adv_v3.py):
#   * data: 100-prompt fixed subset (prompt_bank_humanml3d_overfit100) with its
#     own cached text-feature dir (symlinked from the v1 cache, ids = hml_***).
#   * fitting capacity: anchor_weight 1.0 -> 0.3 and lr 5e-6 -> 2e-5 so the
#     policy can actually depart from base KIMODO toward the trackable optimum
#     on this small set (the anti-FREEZE accept gate is kept intact, so the
#     frozen-glide degenerate mode is still rejected at the SFT-target level).
#   * schedule: max_iters 3000 -> 2000, dense checkpoints/logging to watch the
#     reward curve.
#   * starts from BASE KIMODO (load_from=None), like v3.
#
# Launch (Taiji vermo image, single 8xV100 node, DDP via accelerate):
#   python3 tools/taiji_submit.py physflow_overfit100 \
#       --host_num 1 --gpu_name V100 \
#       --start-cmd "cd <repo> && bash tools/physflow_mn_start.sh \
#           configs/physflow/physflow_overfit100.py --auto-resume"

# NOTE (iter 2): the first overfit attempt (work_dirs/physflow_overfit100,
# anchor_weight=0.3, lr=2e-5) FROZE-COLLAPSED by step ~400: SFT keeps selecting
# the least-articulated still-acceptable best-of-N candidate, ratcheting
# joint_std down to the 0.05 floor until every candidate is rejected (n_good->0,
# joint_std ~0.003, reward_best ~0.6 = the trivially-trackable frozen optimum),
# and anchor=0.3 was too weak to pull the policy back out. This reproduces the
# exact v2->v3 collapse on 100 samples and confirms the anti-freeze gate fires
# (n_good->0). Fix: restore the v3 anchor_weight=1.0 (proven anti-ratchet) and
# drop lr 2e-5 -> 1e-5. Success here = STABLE training (n_good stays high,
# joint_std >~0.06, accepted completion->1 / fall->0), NOT reward_best->0
# (reward_best->0 IS the freeze degenerate optimum the gate must reject).

_base_ = './physflow_online_adv_v3.py'

work_dir = 'work_dirs/physflow_overfit100_anchor1'

# 100 fixed prompts + their cached LLM2Vec text features (built by reusing the
# v1 feature cache; manifest ids are hml_000..hml_099).
train_dataloader = dict(
    dataset=dict(
        corpus_file='configs/experiments/physflow_kimodo_g1/prompt_bank_humanml3d_overfit100.jsonl',
        feature_dir='data/kimodo_text_feature/kimodo_g1_llm2vec_overfit100',
        split='train',
        max_samples=100,
        min_frames=60,
        max_frames=120,
    ),
)

# Reduce the base anchor and raise the LR so the loop has enough capacity to
# actually fit this tiny set; keep every anti-collapse / anti-freeze accept
# threshold from v3 so we are still testing *the method*, not a degenerate one.
trainer = dict(
    anchor_weight=1.0,
    tracker_pool_dir='work_dirs/physflow_overfit100_anchor1/tracker_motion_pool',
    pool_max_motions=600,
)

optimizer = dict(type='AdamW', lr=1e-5, betas=[0.9, 0.99], weight_decay=0.0)

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

train_cfg = dict(
    by_epoch=False,
    max_iters=2000,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=5),
    checkpoint=dict(type='CheckpointHook', interval=100, max_keep_ckpts=12, save_last=True),
)

load_from = None
