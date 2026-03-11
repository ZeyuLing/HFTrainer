"""TinyLlama LoRA SFT demo.

Usage:
    python3 tools/train.py configs/llm/llama_lora_demo.py
"""

_base_ = ['../_base_/default_runtime.py']

CKPT_PATH = 'checkpoints/TinyLlama-1.1B-Chat-v1.0'
DATA_PATH = 'data/llm/demo/alpaca_sample.json'

model = dict(
    type='CausalLMBundle',
    max_length=256,
    model=dict(
        type='AutoModelForCausalLM',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            torch_dtype='auto',
        ),
        trainable='lora',
        checkpoint_format='lora',
        lora_cfg=dict(
            task_type='CAUSAL_LM',
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules='all-linear',
        ),
    ),
)

trainer = dict(
    type='CausalLMTrainer',
    val_max_new_tokens=32,
    do_sample=False,
)

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

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.0)
lr_scheduler = dict(type='cosine', num_warmup_steps=2)

train_cfg = dict(
    by_epoch=False,
    max_iters=10,
    val_interval=10,
)

work_dir = 'work_dirs/llama_lora_smoke'
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
    save_dir='work_dirs/llama_lora_smoke/vis',
    max_samples=4,
)
