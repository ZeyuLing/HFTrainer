# configs/text2image/sd15_demo.py
# Smoke test config for SD1.5 on demo data (8 caption-image pairs)
# Usage: python tools/train.py configs/text2image/sd15_demo.py --work-dir work_dirs/sd15_smoke

_base_ = ['../_base_/default_runtime.py']

CKPT_PATH = 'checkpoints/stable-diffusion-v1-5'
DATA_ROOT = 'data/text2image/demo'

# ── ModelBundle ──
model = dict(
    type='SD15Bundle',
    tokenizer_path=CKPT_PATH,
    max_token_length=77,
    text_encoder=dict(
        type='CLIPTextModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            subfolder='text_encoder',
        ),
        trainable=False,
        save_ckpt=False,
    ),
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            subfolder='vae',
        ),
        trainable=False,
        save_ckpt=False,
    ),
    unet=dict(
        type='UNet2DConditionModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            subfolder='unet',
        ),
        trainable=True,
        save_ckpt=True,
    ),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            subfolder='scheduler',
        ),
        trainable=False,
        save_ckpt=False,
    ),
)

# ── Trainer ──
trainer = dict(
    type='SD15Trainer',
    prediction_type='epsilon',
    num_val_inference_steps=5,    # fast for smoke test
    val_prompts=['a red cat on a mat', 'a blue sky'],
)

# ── Data ──
train_dataloader = dict(
    type='HFImageFolderDataset',
    data_root=DATA_ROOT,
    image_size=512,
    max_samples=8,
    batch_size=1,
    num_workers=0,
    shuffle=True,
)

# ── Optimizer ──
optimizer = dict(type='AdamW', lr=1e-5, weight_decay=1e-2)
lr_scheduler = dict(type='constant_with_warmup', num_warmup_steps=2)

# ── Training Loop ──
train_cfg = dict(
    by_epoch=False,
    max_iters=10,    # smoke test: 10 steps
    val_interval=5,
    save_interval=10,
)

# ── Runtime ──
work_dir = 'work_dirs/sd15_smoke'
auto_resume = False

accelerator = dict(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=2,
    ),
    logger=dict(type='LoggerHook', interval=1),
)

val_visualizer = dict(
    type='FileVisualizer',
    save_dir='work_dirs/sd15_smoke/vis',
    max_samples=4,
)
