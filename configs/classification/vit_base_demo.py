# configs/classification/vit_base_demo.py
# Smoke test config for ViT-B/16 on demo data (20 samples, 4 classes)
# Usage: python tools/train.py configs/classification/vit_base_demo.py --work-dir work_dirs/vit_smoke

_base_ = ['../_base_/default_runtime.py']

CKPT_PATH = 'checkpoints/vit-base-patch16-224'
DATA_ROOT = 'data/classification/demo/images'
NUM_CLASSES = 4

# ── ModelBundle ──
model = dict(
    type='ViTBundle',
    num_labels=NUM_CLASSES,
    image_size=224,
    model=dict(
        type='ViTForImageClassification',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        ),
        trainable=True,
        save_ckpt=True,
    ),
)

# ── Trainer ──
trainer = dict(
    type='ClassificationTrainer',
    label_smoothing=0.1,
)

# ── Data ──
train_dataloader = dict(
    type='ImageFolderDataset',
    data_root=DATA_ROOT,
    split='train',
    image_size=224,
    max_samples=20,
    batch_size=4,
    num_workers=0,
    shuffle=True,
)

val_dataloader = dict(
    type='ImageFolderDataset',
    data_root=DATA_ROOT,
    split='train',  # use same data for smoke test
    image_size=224,
    max_samples=20,
    batch_size=4,
    num_workers=0,
    shuffle=False,
)

# ── Optimizer & Scheduler ──
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=1e-2)
lr_scheduler = dict(type='cosine_with_warmup', num_warmup_steps=2)

# ── Training Loop ──
train_cfg = dict(
    by_epoch=False,
    max_iters=10,      # smoke test: 10 steps
    val_interval=5,
)

# ── Runtime ──
work_dir = 'work_dirs/vit_smoke'
auto_resume = False

accelerator = dict(
    mixed_precision='no',
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

# ── Evaluation ──
val_evaluator = dict(
    type='AccuracyEvaluator',
    topk=(1,),
)

val_visualizer = dict(
    type='FileVisualizer',
    save_dir='work_dirs/vit_smoke/vis',
    max_samples=8,
)
