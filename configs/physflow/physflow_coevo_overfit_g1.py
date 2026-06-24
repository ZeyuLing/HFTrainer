# PhysFlow co-evolution OVERFIT config (G1-native generator).
#
# Purpose: validate the FULL closed loop end-to-end on a tiny, fixed prompt set
# before any formal run --
#   GENERATOR (flow-matching RAFT vs judge) -> qpos decode -> JUDGE (MuJoCo+ONNX)
#   -> accept-filter -> reward SFT (+GT mix) -> pool export -> TRAINEE (PPO/IsaacGym)
#   -> JUDGE SYNC (export trainee ONNX, feed back as next round's judge).
#
# Driven by scripts/embodied/physflow_coevolve_orchestrator.py, which overrides
# train_cfg.max_iters / trainer.tracker_pool_dir / checkpoint interval and the
# per-round --work-dir / --load-from via CLI. Here we only pin the tiny prompt
# corpus and a slightly higher lr so the generator can visibly overfit the 8
# fixed prompts within a few dozen iters/round.

_base_ = 'physflow_online_adv_g1_38dim.py'

work_dir = 'work_dirs/physflow_coevo_overfit_g1'

# Tiny fixed prompt set (8 simple, in-place / highly-trackable motions) so the
# loop is fast and the reward signal is clean. train == eval (overfit).
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(anno_file='data/annotation/_coevo_overfit8.json'),
)

# Overfit: bump lr (4x the formal 5e-6) to move the few prompts faster.
optimizer = dict(type='AdamW', lr=2e-5, betas=[0.9, 0.99], weight_decay=0.0)

# Score/pool GT every step (tiny set) so the trainee pool fills quickly with the
# real anchors alongside accepted generations.
trainer = dict(
    gt_pool_freq=1,
)
