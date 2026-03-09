# configs/llm/llama_sft_demo.py (using TinyLlama)
# Smoke test config for TinyLlama SFT on 10 Alpaca samples
# Usage: python tools/train.py configs/llm/llama_sft_demo.py --work-dir work_dirs/llama_smoke

_base_ = ['../_base_/default_runtime.py']

CKPT_PATH = 'checkpoints/TinyLlama-1.1B-Chat-v1.0'
DATA_PATH = 'data/llm/demo/alpaca_sample.json'

# ── ModelBundle ──
model = dict(
    type='CausalLMBundle',
    max_length=256,
    model=dict(
        type='AutoModelForCausalLM',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            torch_dtype='auto',
        ),
        trainable=True,
        save_ckpt=True,
    ),
)

# ── Trainer ──
trainer = dict(
    type='CausalLMTrainer',
    val_max_new_tokens=32,
    do_sample=False,
)

# ── Data ──
train_dataloader = dict(
    type='AlpacaDataset',
    data_path=DATA_PATH,
    tokenizer_name_or_path=CKPT_PATH,
    max_length=256,
    max_samples=10,
    batch_size=2,
    num_workers=0,
    shuffle=True,
)

# ── Optimizer ──
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=1e-4)
lr_scheduler = dict(type='cosine', num_warmup_steps=2)

# ── Training Loop ──
train_cfg = dict(
    by_epoch=False,
    max_iters=10,   # smoke test: 10 steps
    val_interval=10,
    save_interval=10,
)

# ── Runtime ──
work_dir = 'work_dirs/llama_smoke'
auto_resume = False

accelerator = dict(
    mixed_precision='bf16',
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
    save_dir='work_dirs/llama_smoke/vis',
    max_samples=4,
)
