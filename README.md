# HF-Trainer

> Config-driven training for HuggingFace-native models, built on `accelerate`, with one shared task core for both training and inference.

HF-Trainer is for teams that like MMEngine-style `.py` configs, but want the runtime, checkpointing, and model ecosystem of HuggingFace rather than another custom engine.

## Why Use HF-Trainer

| If you want | HF-Trainer gives you |
| --- | --- |
| declarative experiments instead of ad-hoc scripts | MMEngine-style `.py` configs and registry-based construction |
| native HuggingFace runtime behavior | `accelerate` for DDP, FSDP, DeepSpeed, mixed precision, logging, and state save/load |
| direct use of `transformers`, `diffusers`, and `peft` | no extra wrapper model layer just to fit another framework |
| one place to write task logic | `ModelBundle` shared by `Trainer` and `Pipeline` |
| memory-aware training control | global AMP, per-module dtype, gradient checkpointing, freeze, and LoRA from config |
| selective fine-tuning and small checkpoints | per-module `trainable`, `save_ckpt`, `checkpoint_format`, and LoRA control |
| reliable restart on long-running jobs | `auto_resume`, model-only load, and full accelerator resume |

## Where It Fits Best

HF-Trainer is a good fit when you need a reusable training framework across multiple task families, but do not want to give up HuggingFace-native components or maintain two copies of the same task logic for train and infer.

It is especially useful when:

- your project mixes classification, diffusion, video, LLM, or adversarial training
- you need per-module freeze / save / LoRA behavior from config
- you care about resume semantics and distributed-runtime correctness
- you want a smaller framework surface than MMEngine-style sample containers and mode-switching forwards

## Runnable Today

| Task | Core Stack | Demo Config | Status |
| --- | --- | --- | --- |
| Classification | `ViTBundle` + `ClassificationTrainer` + `ClassificationPipeline` | `configs/classification/vit_base_demo.py` | verified |
| Text-to-image | `SD15Bundle` + `SD15Trainer` + `SD15Pipeline` | `configs/text2image/sd15_demo.py` | verified |
| Causal LM SFT | `CausalLMBundle` + `CausalLMTrainer` + `CausalLMPipeline` | `configs/llm/llama_sft_demo.py` | verified |
| Causal LM LoRA | `CausalLMBundle` + `CausalLMTrainer` + `CausalLMPipeline` | `configs/llm/llama_lora_demo.py` | verified |
| Text-to-video | `WanBundle` + `WanTrainer` + `WanPipeline` | `configs/text2video/wan_demo.py` | verified |
| GAN | `StyleGAN2Bundle` + `GANTrainer` + `StyleGAN2Pipeline` | `configs/gan/gan_demo.py` | runnable reference |
| DMD | `DMDBundle` + `DMDTrainer` + `DMDPipeline` | `configs/distillation/dmd_demo.py` | runnable reference |

## Start In 60 Seconds

Install:

```bash
pip install -e .
```

Prepare demo assets:

```bash
bash tools/download_checkpoints.sh
python3 tools/download_demo_data.py --task all
```

Run the most reliable first example:

```bash
python3 tools/train.py configs/classification/vit_base_demo.py
```

Run the verified LoRA path:

```bash
python3 tools/train.py configs/llm/llama_lora_demo.py
python3 tools/infer.py \
  --config configs/llm/llama_lora_demo.py \
  --checkpoint work_dirs/llama_lora_smoke/checkpoint-iter_10 \
  --merge-lora \
  --prompt "Name one primary color."
```

Run distributed training:

```bash
bash tools/dist_train.sh configs/text2video/wan_demo.py 8
```

Run the startup smoke suite:

```bash
python3 -m pytest -m smoke tests/smoke/test_task_startup.py
```

The smoke suite uses temporary reduced configs to verify that each task stack can start training and inference through the real CLI entry points. On smaller GPUs, the WAN case is skipped unless the device has enough memory for the local WAN checkpoint.

## Core Ideas

- `AccelerateRunner` builds the full runtime from one config and owns the training loop.
- `ModelBundle` holds task sub-modules and the shared atomic forward functions.
- `Trainer` owns training logic; `Pipeline` owns inference control flow.
- Hooks, evaluators, and visualizers stay simple and dict-based.

## Memory Control From Config

- global AMP: `accelerator.mixed_precision='no'|'fp16'|'bf16'`
- per-module load dtype: `from_pretrained.torch_dtype` or `from_pretrained.dtype`
- per-module post-load cast: `module_dtype='fp32'|'fp16'|'bf16'`
- activation memory reduction: `gradient_checkpointing=True` on sub-modules that expose the underlying HF hook
- optimizer/state reduction: `trainable=False`, `trainable='lora'`, and `accelerator.gradient_accumulation_steps`

If you need a strict policy like `vae=fp32` and `transformer=bf16`, prefer per-module dtypes and keep `accelerator.mixed_precision='no'`. Global AMP can still autocast eligible ops on top of your module weights.

See:

- [English Memory and Precision Guide](docs/en/memory.md)
- [简体中文 显存与精度指南](docs/zh-cn/memory.md)

## Model Integration Paths

HF-Trainer now exposes two explicit integration paths:

| Starting point | What you implement | What stays HuggingFace-native |
| --- | --- | --- |
| a model already supported by `transformers` / `diffusers` | a task bundle plus task training logic | `from_pretrained`, component classes, tokenizer / processor, and exported inference artifact |
| a custom or self-developed model | your own `nn.Module` plus a task bundle | config-driven construction, checkpointing, hooks, runner, and optional custom `save_pretrained` |

Rule of thumb:

- if HuggingFace already has the model class, keep using the official class inside the bundle and only add training-specific wiring
- if HuggingFace does not have the model class, use `ModelBundle.from_config(...)` directly and optionally implement bundle-level `from_pretrained/save_pretrained` later

## Documentation

| Topic | English | 简体中文 |
| --- | --- | --- |
| Docs Home | [Home](docs/en/index.md) | [首页](docs/zh-cn/index.md) |
| Installation | [Installation](docs/en/installation.md) | [安装说明](docs/zh-cn/installation.md) |
| Quick Start | [Quick Start](docs/en/quickstart.md) | [快速开始](docs/zh-cn/quickstart.md) |
| Integration Guide | [Integration](docs/en/integration.md) | [模型接入](docs/zh-cn/integration.md) |
| API Reference | [API Reference](docs/en/api_reference.md) | [API 参考](docs/zh-cn/api_reference.md) |
| Memory and Precision | [Memory](docs/en/memory.md) | [显存与精度](docs/zh-cn/memory.md) |
| LoRA | [LoRA](docs/en/lora.md) | [LoRA](docs/zh-cn/lora.md) |
| Architecture | [Architecture](docs/en/architecture.md) | [架构设计](docs/zh-cn/architecture.md) |
| Hook System | [Hook System](docs/en/design/hooks.md) | [Hook 系统](docs/zh-cn/design/hooks.md) |
| Distributed Training | [Distributed](docs/en/distributed.md) | [分布式训练](docs/zh-cn/distributed.md) |
| Experiment Directory | [Experiment Dir](docs/en/experiment_dir.md) | [实验目录](docs/zh-cn/experiment_dir.md) |
| Task Matrix | [Tasks](docs/en/tasks.md) | [任务矩阵](docs/zh-cn/tasks.md) |
| Design Docs | [Design Index](docs/en/design/index.md) | [设计文档](docs/zh-cn/design/index.md) |

## Public API Surface

The public API reference now documents the user-facing framework surface:

- runner: `AccelerateRunner`
- model core: `ModelBundle`
- training/inference base classes: `BaseTrainer`, `BasePipeline`
- runtime helpers: hooks, evaluators, visualizers, checkpoint utils
- CLI entry points: `tools/train.py`, `tools/infer.py`

Start here:

- [English API Reference](docs/en/api_reference.md)
- [简体中文 API 参考](docs/zh-cn/api_reference.md)

## Repository Layout

```text
configs/      runnable experiment configs
hftrainer/    framework package
tools/        train / infer / utility entry points
docs/         English + Chinese documentation
data/         demo datasets
checkpoints/  local pretrained checkpoints for demos
```

## Scope Notes

- `docs/en/` and `docs/zh-cn/` are the source-of-truth public docs. Root-level `docs/*.md` pages are compatibility entry pages.
- The GAN and DMD stacks are runnable reference implementations aligned to the framework path and core algorithm structure, not benchmark-tuned reproductions out of the box.
