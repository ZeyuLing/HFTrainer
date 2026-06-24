# PhysFlow online-adversarial smoke config.
#
# Stage 1: online best-of-N reward-weighted SFT of the KIMODO-G1 generator.
#   * NO 8B text encoder: cached embeddings (data/kimodo_text_feature) feed the
#     diffusion denoiser directly.
#   * Frozen judge tracker (g1-bones-deploy) scores physics realism in MuJoCo.
#   * Only the diffusion denoiser is trainable.
#
# Launch (single GPU, KIMODO py3.10 env, offline HF cache):
#   HF_HOME=checkpoints/kimodo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#   CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
#       configs/physflow/physflow_online_adv_smoke.py

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/physflow_online_adv_smoke'

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
    num_samples=2,
    diffusion_steps=20,
    reward_weighted=False,
    enable_reward=True,
    keep_rollouts=False,
)

# ----- Data (prompts + cached text embeddings) -----
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    shuffle=True,
    dataset=dict(
        type='PhysFlowPromptDataset',
        corpus_file=_corpus,
        feature_dir=_feature_dir,
        split='train',
        fps=30.0,
        min_frames=60,
        max_frames=120,
        max_samples=16,
    ),
)

optimizer = dict(type='AdamW', lr=1e-5, betas=[0.9, 0.99], weight_decay=0.0)
lr_scheduler = None

train_cfg = dict(
    by_epoch=False,
    max_iters=2,
    val_interval=999999,
    max_grad_norm=1.0,
)

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1000, max_keep_ckpts=2, save_last=True),
)

load_from = None
val_dataloader = None
val_evaluator = None
val_visualizer = None
