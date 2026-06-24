# PhysFlow OVERFIT with the Humanoid-GPT (CVPR'26 zero-shot) tracker as the
# physics-realism JUDGE, instead of the ProtoMotions g1-bones-deploy ONNX tracker.
#
# Same online best-of-N reward-weighted SFT loop as physflow_overfit100.py
# (KIMODO-G1 generator, FROZEN judge), but the per-candidate reward comes from
# rolling the motion out under Humanoid-GPT in MuJoCo (scored in HGPT's py3.11
# venv via a long-lived worker; see hftrainer/models/motion/physflow/hgpt_reward.py
# and the bundled Humanoid-GPT worker under hftrainer.models.motion.physflow.trackers).
#
# Goal: launch a REAL training run (not a smoke test) under the KIMODO-G1 + HGPT
# configuration and validate the method works -- reward_best_mean drops toward its
# floor, completion->1, fall->0, WITHOUT articulation collapse (sel_joint_std_mean
# stays >~0.06) -- then a paired base-vs-optimized eval under HGPT must show the
# optimized generator is more physically trackable.
#
# Local single-GPU launch (this dev node has a free Tesla T4 + KIMODO weights):
#   CUDA_VISIBLE_DEVICES=0 HF_HOME=checkpoints/kimodo HF_HUB_OFFLINE=1 \
#   TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
#   python3 tools/train.py configs/physflow/physflow_overfit100_hgpt.py

_base_ = './physflow_overfit100.py'

work_dir = 'work_dirs/physflow_overfit100_hgpt'

trainer = dict(
    judge_backend='hgpt',
    hgpt_freq=50,
    hgpt_input_fps=30,
    # HGPT path produces no ProtoMotions .motion, so there is nothing to export to
    # the trainee pool in this single-sided run; disable pooling to avoid confusion.
    tracker_pool_dir=None,
)
