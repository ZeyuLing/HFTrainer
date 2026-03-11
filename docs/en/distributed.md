# Distributed Training

HF-Trainer delegates distributed execution to HuggingFace `accelerate`.

## Single Process

```bash
python3 tools/train.py configs/classification/vit_base_demo.py
```

## Multi-GPU

```bash
bash tools/dist_train.sh configs/text2video/wan_demo.py 8
```

Or call `accelerate` directly:

```bash
accelerate launch --num_processes=8 tools/train.py configs/text2video/wan_demo.py
```

## FSDP / DeepSpeed

Use standard `accelerate launch` options or an `accelerate config` file. The framework does not add another distributed wrapper layer on top.

## Practical Note

The regular `tools/train.py` path now keeps `LOCAL_RANK` unset in plain single-process runs, so local smoke tests do not accidentally initialize a process group.
