# configs/text2video/wan_demo.py
# Smoke test config for WAN T2V 1.3B on synthetic video data
# Usage: python tools/train.py configs/text2video/wan_demo.py --work-dir work_dirs/wan_smoke

_base_ = ['../_base_/default_runtime.py']

CKPT_PATH = 'checkpoints/Wan2.1-T2V-1.3B-Diffusers'

# ── ModelBundle ──
model = dict(
    type='WanBundle',
    tokenizer_path=CKPT_PATH + '/tokenizer',
    max_token_length=128,
    gradient_checkpointing=True,  # save activation memory during backward
    text_encoder=dict(
        type='UMT5EncoderModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH + '/text_encoder',
            torch_dtype='bf16',
        ),
        trainable=False,
        save_ckpt=False,
    ),
    vae=dict(
        type='AutoencoderKLWan',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH + '/vae',
            torch_dtype='bf16',
        ),
        trainable=False,
        save_ckpt=False,
    ),
    transformer=dict(
        type='WanTransformer3DModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH + '/transformer',
            torch_dtype='bf16',
        ),
        trainable=True,
        save_ckpt=True,
    ),
    scheduler=dict(
        type='FlowMatchEulerDiscreteScheduler',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH + '/scheduler',
        ),
        trainable=False,
        save_ckpt=False,
    ),
)

# ── Trainer ──
trainer = dict(
    type='WanTrainer',
    prediction_type='flow_matching',
    num_val_inference_steps=3,
    val_prompts=['a cat walking'],
    val_num_frames=4,
    val_height=32,
    val_width=32,
)

# ── Data ──
train_dataloader = dict(
    type='HFVideoDataset',
    data_root='data/text2video/demo',
    num_frames=4,       # very short for smoke test
    height=32,          # tiny for smoke test (V100 memory)
    width=32,
    synthetic=True,
    max_samples=4,
    batch_size=1,
    num_workers=0,
    shuffle=True,
)

# ── Optimizer ──
# Use torch.optim.Adafactor (stateless, no second moment) to avoid ~11 GB Adam state memory
# for the 1.3B transformer on V100-32GB. beta2_decay=-0.8 matches HF Adafactor default.
optimizer = dict(type='Adafactor', lr=1e-5, beta2_decay=-0.8, weight_decay=1e-2)
lr_scheduler = None  # Adafactor manages its own step size

# ── Training Loop ──
train_cfg = dict(
    by_epoch=False,
    max_iters=5,    # smoke test: 5 steps
    val_interval=5,
    save_interval=5,
)

# ── Runtime ──
work_dir = 'work_dirs/wan_smoke'
auto_resume = False

accelerator = dict(
    mixed_precision='bf16',
    gradient_accumulation_steps=1,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=1,
    ),
    logger=dict(type='LoggerHook', interval=1),
)

val_visualizer = dict(
    type='FileVisualizer',
    save_dir='work_dirs/wan_smoke/vis',
    max_samples=2,
)
