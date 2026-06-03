# PhysFlow online-adversarial FORMAL run (Stage 1: RAFT / best-of-N reward SFT).
#
# Same loop as the smoke config but at scale:
#   * full HumanML3D prompt corpus (11,241 train captions), shuffled;
#   * best-of-4 candidates per prompt scored by the FROZEN g1-bones-deploy judge;
#   * only the KIMODO-G1 diffusion denoiser is trained (x0 SFT toward best).
#
# Periodic evaluation is decoupled: a separate watcher
#   scripts/embodied/physflow_periodic_eval.py
# scores every saved checkpoint on the held-out test prompts and logs
#   work_dirs/physflow_online_adv_v1/physflow_eval_metrics.jsonl
# so a slow MuJoCo eval never stalls the training loop.
#
# Launch (single GPU, KIMODO py3.10 env, offline HF cache), inside tmux:
#   HF_HOME=checkpoints/kimodo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#   CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
#       configs/physflow/physflow_online_adv_v1.py

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/physflow_online_adv_v1'

_feature_dir = 'data/kimodo_text_feature/kimodo_g1_llm2vec_v1'
_corpus = 'configs/experiments/physflow_kimodo_g1/physflow_text_train.jsonl'

# ----- Model (KIMODO-G1 generator, denoiser trainable) -----
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

# ----- Trainer (online best-of-N reward SFT) -----
trainer = dict(
    type='PhysFlowTrainer',
    num_samples=4,
    diffusion_steps=20,
    reward_weighted=False,
    enable_reward=True,
    keep_rollouts=False,
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

optimizer = dict(type='AdamW', lr=1e-5, betas=[0.9, 0.99], weight_decay=0.0)
lr_scheduler = None

train_cfg = dict(
    by_epoch=False,
    max_iters=3000,
    val_interval=999999,
    max_grad_norm=1.0,
)

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    # Checkpoint every 150 iters -> the periodic-eval watcher picks each one up.
    checkpoint=dict(type='CheckpointHook', interval=150, max_keep_ckpts=8, save_last=True),
)

load_from = None
val_dataloader = None
val_evaluator = None
val_visualizer = None
