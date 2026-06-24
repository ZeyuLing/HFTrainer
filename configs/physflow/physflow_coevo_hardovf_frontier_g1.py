# PhysFlow AGILE hard-overfit, LEARNABILITY-FRONTIER (direction B).
#
# Same frontier mechanism as physflow_coevo_overfit_frontier_g1, but the prompt
# set is 24 of the MOST AGILE real clips (runs / jumps / stairs / fast turns /
# fall-recover), disjoint from the 80 held-out agile eval clips. On these the
# SOTA-warm-started trainee genuinely FAILS, so a learnability frontier exists:
# we expect n_frontier_mean > 0 and the trainee's completion on these clips to
# rise round-by-round -- the proof that the generator->frontier->trainee loop
# actually makes the tracker better (unlike the easy overfit8, which saturates).

_base_ = 'physflow_coevo_overfit_frontier_g1.py'

work_dir = 'work_dirs/physflow_coevo_hardovf_frontier_gtreplay_g1'

train_dataloader = dict(
    dataset=dict(anno_file='data/annotation/_coevo_hardovf_agile.json'),
)

trainer = dict(
    # Crucial fix: real GT hard clips should enter the trainee replay pool even
    # when the frozen tracker Q fails them. Q-validity remains mandatory for
    # generated motions; this relaxation applies only to GT-as-replay.
    gt_pool_accept_mode='kinematic',
)
