# PhysFlow online-adversarial FORMAL run v3 (Stage 1: RAFT / best-of-N reward SFT).
#
# v3 fixes the *articulation collapse* discovered by visual inspection of the v2
# generator (work_dirs/physflow_online_adv_v2/checkpoint-iter_3000):
#   v2 symptom: locomotion prompts ("walks", "jogging in place", "staggers") are
#   generated as a FROZEN standing pose dragged across the floor -- root_disp
#   ~5-6 m with joint temporal std ~0.013 rad (vs base KIMODO ~0.09-0.16 rad). The
#   base model articulates; our RAFT fine-tune froze the legs.
#   Root cause: a pure physical-trackability reward has a *degenerate optimum at
#   "don't move"* -- a static pose never falls, completes fully, and the tracker
#   matches its (constant) joints with near-zero error, so the accept filter
#   (no-fall + completion>=0.9) lets these glides through, the best-of-N argmin
#   (lowest tracking score) prefers them, and RAFT reinforces leg-freezing.
#   This is the mirror image of the v1 collapse (everything fell): v2 over-
#   corrected into the trivially-trackable degenerate mode.
#
# v3 changes:
#   * ANTI-FREEZE accept gate: reject candidates whose articulation (mean temporal
#     std of joint angles over the valid window) is below ``accept_min_joint_std``,
#     and reject pure-translation slides (large root displacement on near-frozen
#     joints). This removes the frozen-glide optimum from both the SFT targets and
#     the trainee pool.
#   * score ceiling: ``accept_max_score`` rejects high-tracking-error candidates.
#   * stronger base anchor (``anchor_weight`` 0.5 -> 1.0): the base KIMODO model
#     articulates, so pulling harder toward it preserves leg motion / diversity.
#   * fresh run from BASE (NOT resumed from the collapsed v2 iter_3000).
#   * ``sel_joint_std_mean`` is now logged every step -- watch it stay >~0.07; if
#     it trends toward 0 the policy is collapsing again.
#
# Periodic evaluation is decoupled (scripts/embodied/physflow_periodic_eval.py).
#
# Launch (single GPU, KIMODO py3.10 env, offline HF cache), inside tmux:
#   HF_HOME=checkpoints/kimodo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#   CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
#       configs/physflow/physflow_online_adv_v3.py

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/physflow_online_adv_v3'

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
    accept_max_score=2.5,
    anchor_weight=1.0,
    # anti-freeze: reject degenerate frozen-pose glides. Base locomotion has
    # joint_std ~0.09-0.16; collapsed glides ~0.013 -> a 0.05 floor separates them.
    accept_min_joint_std=0.05,
    accept_max_root_disp_if_frozen=1.0,
    accept_frozen_joint_std=0.03,
    # closed loop: accepted (trackable, non-frozen) motions accumulate here
    tracker_pool_dir='work_dirs/physflow_online_adv_v3/tracker_motion_pool',
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
