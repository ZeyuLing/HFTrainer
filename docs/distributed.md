# Distributed Training

All distributed training is managed by HuggingFace Accelerate. No custom distributed wrappers are needed.

## DDP (default)

```bash
# Auto-detect all GPUs
bash tools/dist_train.sh configs/text2video/wan_demo.py

# Specify GPU count
bash tools/dist_train.sh configs/text2video/wan_demo.py 8

# Or use accelerate launch directly
accelerate launch --num_processes=8 tools/train.py configs/text2video/wan_demo.py
```

## FSDP

```bash
accelerate launch --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    tools/train.py configs/text2video/wan_demo.py
```

Or generate a config file via `accelerate config`, then:

```bash
accelerate launch --config_file fsdp_config.yaml tools/train.py configs/text2video/wan_demo.py
```

## DeepSpeed ZeRO

```bash
accelerate launch --use_deepspeed \
    --deepspeed_config_file ds_config.json \
    tools/train.py configs/text2video/wan_demo.py
```

## Single GPU (no accelerate launch)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/text2video/wan_demo.py
```

## Accelerator Config in `.py`

You can also set Accelerate options in the experiment config:

```python
accelerator = dict(
    mixed_precision='bf16',
    gradient_accumulation_steps=4,
)
```

These are passed to the `Accelerator` constructor. Command-line flags from `accelerate launch` take precedence when both are specified.
