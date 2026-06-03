# PhysFlow online-adversarial FORMAL run v2 (Stage 1: RAFT / best-of-N reward SFT).
#
# v2 fixes the generator COLLAPSE observed in v1 (work_dirs/physflow_online_adv_v1):
#   v1 symptom: healthy for ~80 steps (reward_best 0.4-0.9, best<cand), then
#   sampling diversity collapsed (best==cand by step ~100) and the policy drifted
#   into an untrackable mode (reward_best ~6, every eval motion falls at frame ~48).
#   Root cause: pure best-of-N self-SFT with (a) no reward filtering -> it trained
#   toward the argmin even when ALL candidates fell, and (b) no anchor -> the
#   output distribution sharpened until best-of-N lost all signal.
#
# v2 changes (standard RAFT/ReST recipe):
#   * accept filter: only SFT toward candidates the robot actually executes
#     (no fall + completion >= 0.9); rejected prompts get zero SFT gradient.
#   * anchor regularizer: MSE(pred, base_KIMODO_pred) keeps the policy from
#     collapsing / drifting (preserves sampling diversity).
#   * lower lr + gradient accumulation for a smoother, larger effective batch.
#
# Periodic evaluation is decoupled (scripts/embodied/physflow_periodic_eval.py).
#
# Launch (single GPU, KIMODO py3.10 env, offline HF cache), inside tmux:
#   HF_HOME=checkpoints/kimodo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#   CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
#       configs/physflow/physflow_online_adv_v2.py

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/physflow_online_adv_v2'

_feature_dir = 'data/kimodo_text_feature/kimodo_g1_llm2vec_v1'
_corpus = 'configs/experiments/physflow_kimodo_g1/physflow_text_train.jsonl'

# ----- Model (KIMODO-G1 generator, denoiser trainable; frozen base anchor) -----
model = dict(
    type='PhysFlowBundle',
    kimodo_model='Kimodo-G1-RP-v1',
    checkpoint_dir='checkpoints/kimodo/hub',
    hf_home='checkpoints/kimodo',
    cfg_weight=[2.0, 2.0],
    cfg_type='separated',
    sample_diffusion_steps=20,
    offline=True,
)

# ----- Trainer (online best-of-N reward SFT + accept filter + base anchor) -----
trainer = dict(
    type='PhysFlowTrainer',
    num_samples=4,
    diffusion_steps=20,
    reward_weighted=False,
    enable_reward=True,
    keep_rollouts=False,
    # anti-collapse controls
    accept_min_completion=0.9,
    accept_require_no_fall=True,
    accept_max_score=None,
    anchor_weight=0.5,
    # closed loop: accepted (trackable) motions accumulate here for the trainee
    tracker_pool_dir='work_dirs/physflow_online_adv_v2/tracker_motion_pool',
    pool_max_motions=4000,
)

# ----- Data (full prompt corpus + cached text embeddings) -----
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=True,
    dataset=dict(
        type='PhysFlowPromptDataset',
        corpus_file=_corpus,
        feature_dir=_feature_dir,
        split='train',
        fps=30.0,
        min_frames=60,
        max_frames=150,
        max_samples=None,
    ),
)

optimizer = dict(type='AdamW', lr=5e-6, betas=[0.9, 0.99], weight_decay=0.0)
lr_scheduler = None

train_cfg = dict(
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
    checkpoint=dict(type='CheckpointHook', interval=150, max_keep_ckpts=8, save_last=True),
)

load_from = None
val_dataloader = None
val_evaluator = None
val_visualizer = None
