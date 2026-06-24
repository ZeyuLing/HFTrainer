# PhysFlow online-adversarial fine-tune of the G1-native 38-d HyMotion
# flow-matching generator (path b').  Inherits the model + dataset definition
# from the supervised G1-T2M config and swaps in the PhysFlow online RAFT
# trainer + frozen MuJoCo/ONNX judge, mirroring physflow_online_adv_v3.py but
# in the 38-d G1 flow-matching space (no KIMODO, no SMPL retarget).
#
# Loop per step (PhysFlowG1Trainer):
#   sample N motions from the live policy (flow-matching ODE)
#   -> decode_g1_to_qpos -> qpos CSV -> frozen judge (MuJoCo+ONNX) score
#   -> pick best *acceptable* (no-fall + completion + anti-freeze) per prompt
#   -> reward-filtered flow-matching SFT + anchor MSE to frozen base
#   -> accepted motions accumulate in tracker_motion_pool (gen->trainee).

_base_ = 'hymotion_g1_t2m_38dim.py'

work_dir = 'work_dirs/physflow_online_adv_g1_38dim'

# ----- Model: reuse the G1-native generator, wrapped for the online loop -----
model = dict(
    type='PhysFlowG1Bundle',
    sample_steps=50,
    sample_guidance=1.0,
)

# ----- Trainer: online best-of-N reward SFT + accept filter + base anchor -----
trainer = dict(
    type='PhysFlowG1Trainer',
    _delete_=True,
    num_samples=4,
    diffusion_steps=50,          # flow-matching ODE steps per sample
    reward_weighted=False,
    enable_reward=True,
    # Optional G1/HYMotion style reward. Build once with:
    #   python3 scripts/embodied/build_g1_style_bank.py \
    #     --anno data/annotation/train_g1_t2m_emb_minus_heldout.json \
    #     --out data/g1_style_bank/train_minus_heldout_20k.npz --max-items 20000
    # Then set style_reward_bank to that path and tune style_reward_weight.
    style_reward_bank=None,
    style_reward_weight=0.0,
    keep_rollouts=False,
    judge_backend='protomotions',
    # anti-collapse controls (same rationale as physflow_online_adv_v3)
    accept_min_completion=0.9,
    accept_require_no_fall=True,
    accept_max_score=2.5,
    anchor_weight=1.0,
    # GT mixing: persistent supervised FM term toward real G1 motion. Gives a
    # stable cold-start signal while n_good~=0 and anchors against collapse;
    # at 0.5 it stays below the reward term's implicit 1.0 so self-generated
    # trackable targets still dominate once candidates start passing the gate.
    gt_weight=0.5,
    # GT-as-special-candidate: also stream judge-accepted GT clips into the
    # trainee pool (real-data mixing for the co-evolving tracker). gt_pool_freq=2
    # scores GT every other step to bound judge cost while keeping a steady real
    # -data trickle alongside the (currently sparse) accepted generations.
    export_gt_to_pool=True,
    gt_pool_freq=2,
    # anti-freeze: reject degenerate frozen-pose glides.
    accept_min_joint_std=0.05,
    accept_max_root_disp_if_frozen=1.0,
    accept_frozen_joint_std=0.03,
    # closed loop: accepted (trackable, non-frozen) motions -> trainee pool.
    tracker_pool_dir='work_dirs/physflow_online_adv_g1_38dim/tracker_motion_pool',
    pool_max_motions=4000,
)

# ----- Data: prompt corpus = the G1 dataset (dual CLIP+Qwen3 embeddings) -----
# We override only the loader-level knobs; the dataset definition (type, anno,
# embeddings) is inherited from the base config.  GT motion is ignored online.
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
)

# ----- Optimizer: low lr for online fine-tune -----
optimizer = dict(type='AdamW', lr=5e-6, betas=[0.9, 0.99], weight_decay=0.0)
lr_scheduler = None

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=3000,
    val_interval=999999,
    max_grad_norm=1.0,
)

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=4,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=150,
                    max_keep_ckpts=8, save_last=True),
)

# Warm-start from the supervised G1-native generator (NOT HY-Motion-Lite).
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_g1_t2m_38dim/checkpoint-g1base',
    load_scope='model',
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
