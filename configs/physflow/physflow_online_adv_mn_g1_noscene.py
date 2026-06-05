# PhysFlow multi-node run with scene-dependent HumanML3D prompts filtered out.
#
# Use this for the next formal generator/tracker co-training pass. It preserves
# the existing mn hyperparameters but replaces the train corpus with prompts that
# do not require chairs, stairs, props, vehicles, or other 3D scene fixtures.

_base_ = './physflow_online_adv_mn.py'

work_dir = 'work_dirs/physflow_online_adv_mn_g1_noscene'

trainer = dict(
    tracker_pool_dir='work_dirs/physflow_online_adv_mn_g1_noscene/tracker_motion_pool',
)

train_dataloader = dict(
    dataset=dict(
        corpus_file='configs/experiments/physflow_kimodo_g1/physflow_text_train_g1_noscene.jsonl',
    ),
)
