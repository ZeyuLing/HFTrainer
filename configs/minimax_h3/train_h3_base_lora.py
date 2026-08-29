"""Experimental cached-feature LoRA fine-tuning for MiniMax-H3 Base."""

# ruff: noqa: C408 - MMEngine config files conventionally use dict(...).

_base_ = ["../_base_/default_runtime.py"]

root = __import__("os").environ.get("MINIMAX_H3_ROOT", "checkpoints/MiniMax-H3")
manifest = __import__("os").environ.get(
    "MINIMAX_H3_CACHE_MANIFEST", "data/minimax_h3/train.jsonl"
)
work_dir = __import__("os").environ.get(
    "HFTRAINER_WORK_DIR", "outputs/training/minimax_h3_lora"
)

custom_imports = dict(
    imports=[
        "hftrainer.models.minimax_h3",
        "hftrainer.trainers.minimax_h3",
        "hftrainer.datasets.synchronized_audio_video",
    ],
    allow_failed_imports=False,
)

# The conditioner and VAEs are deliberately absent: the dataset owns their
# cached outputs, leaving only the 33B denoiser resident during optimization.
model = dict(
    type="MiniMaxH3Bundle",
    variant="fl2va",
    transformer=dict(
        type="MiniMaxH3Transformer3DModel",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="transformer",
            torch_dtype="bf16",
            low_cpu_mem_usage=True,
            device="cpu",
            strict=True,
        ),
        trainable="lora",
        save_ckpt=True,
        checkpoint_format="lora",
        gradient_checkpointing=True,
        lora_cfg=dict(
            rank=16,
            alpha=16,
            dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
    ),
    scheduler=dict(
        type="MiniMaxH3Scheduler",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="scheduler",
        ),
        trainable=False,
        save_ckpt=False,
    ),
    audio_scheduler=dict(
        type="MiniMaxH3Scheduler",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="audio_scheduler",
        ),
        trainable=False,
        save_ckpt=False,
    ),
)

trainer = dict(
    type="MiniMaxH3Trainer",
    mode="t2va",
    video_loss_weight=1.0,
    audio_loss_weight=1.0,
    timestep_distribution="uniform",
    min_timestep=0.0,
    max_timestep=1.0,
)

train_dataloader = dict(
    dataset=dict(
        type="MiniMaxH3CachedFeatureDataset",
        manifest=manifest,
        verify_files=True,
    ),
    batch_size=1,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
)

optimizer = dict(type="AdamW", lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
lr_scheduler = dict(type="cosine", num_warmup_steps=100)

train_cfg = dict(by_epoch=False, max_iters=2000, val_interval=None)
auto_resume = True

accelerator = dict(
    mixed_precision="bf16",
    gradient_accumulation_steps=1,
)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=250,
        max_keep_ckpts=3,
        save_last=True,
    ),
    logger=dict(type="LoggerHook", interval=10),
)
