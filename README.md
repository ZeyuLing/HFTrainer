<div align="center">

<img src="assets/hftrainer_logo.png" alt="HFTrainer logo" width="420" />

# HF-Trainer

**Config-driven training and inference for HuggingFace-native and official model stacks.**

One shared task core for training and inference, native `transformers` /
`diffusers` / `peft` integration, and thin pinned adapters when an upstream
project already owns the complete algorithm lifecycle.

<p>
  <a href="docs/en/index.md"><strong>Documentation</strong></a> •
  <a href="docs/en/quickstart.md"><strong>Quick Start</strong></a> •
  <a href="docs/en/integration.md"><strong>Integration</strong></a> •
  <a href="docs/en/api_reference.md"><strong>API Reference</strong></a> •
  <a href="docs/en/tasks.md"><strong>Task Matrix</strong></a> •
  <a href="https://github.com/ZeyuLing/HFTrainer/issues"><strong>Issues</strong></a>
</p>

<p>
  <a href="docs/en/index.md">English Docs</a> |
  <a href="docs/zh-cn/index.md">简体中文文档</a>
</p>

<p>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Accelerate" src="https://img.shields.io/badge/Accelerate-native-4f46e5">
  <img alt="HuggingFace" src="https://img.shields.io/badge/HuggingFace-transformers%20%7C%20diffusers-facc15?logo=huggingface&logoColor=black">
  <img alt="Config System" src="https://img.shields.io/badge/MMEngine-config%20%2B%20registry-0ea5e9">
</p>

</div>

## Why HF-Trainer

HF-Trainer is for teams that like MMEngine-style `.py` configs, but want the runtime behavior, model ecosystem, and export path of HuggingFace instead of another custom engine.

It is built for a specific workflow:

- keep experiment configuration declarative
- keep model classes and inference artifacts HuggingFace-native
- avoid writing one copy of task logic for training and another for inference
- fine-tune large models with per-module freeze, LoRA, dtype, and checkpoint control

## What You Get

| You want | HF-Trainer gives you |
| --- | --- |
| reproducible experiments instead of ad-hoc scripts | MMEngine-style `.py` configs and registry-based construction |
| native large-model runtime behavior | `accelerate` for DDP, FSDP, DeepSpeed, mixed precision, logging, and state save/load |
| direct use of HuggingFace components | native `transformers`, `diffusers`, and `peft` classes without framework-specific wrapper semantics |
| one place to implement task logic | `ModelBundle` shared by `Trainer` and `Pipeline` |
| safe adoption of a complete upstream stack | managed trainers and lazy adapters preserve the official loop instead of forking it |
| less framework glue for HF-native tasks | parent-level `from_config` / `from_pretrained` plus declarative bundle specs instead of per-bundle boilerplate |
| memory-aware fine-tuning | config-driven freeze, LoRA, per-module dtype, gradient checkpointing, and accumulation |
| reliable restart and export | `auto_resume`, model-only load, full accelerator resume, and task-native `save_pretrained(...)` |

## Runnable Today

| Task | Core Stack | Example Config | Status |
| --- | --- | --- | --- |
| Classification | `ViTBundle` + `ClassificationTrainer` + `ClassificationPipeline` | `configs/classification/vit_base_demo.py` | verified |
| Text-to-image | `SD15Bundle` + `SD15Trainer` + `SD15Pipeline` | `configs/text2image/sd15_demo.py` | verified |
| Causal LM SFT | `CausalLMBundle` + `CausalLMTrainer` + `CausalLMPipeline` | `configs/llm/llama_sft_demo.py` | verified |
| Causal LM LoRA | `CausalLMBundle` + `CausalLMTrainer` + `CausalLMPipeline` | `configs/llm/llama_lora_demo.py` | verified |
| Text-to-video | `WanBundle` + `WanTrainer` + `WanPipeline` | `configs/text2video/wan_demo.py` | verified |
| LTX-2.5 distilled audio-video inference | `LTXVideoBundle` + `LTXVideoPipeline` | `configs/ltx_video/infer_ltx_video_2_5_distilled.py` | API/config contract verified; 22B GPU run not performed |
| LTX-2.5 LoRA training + dev inference | managed `LTXVideoTrainer` + `LTXVideoPipeline` | `configs/ltx_video/train_ltx_video_2_5_lora.py` | API/config contract verified; 22B GPU run not performed |
| GAN | `StyleGAN2Bundle` + `GANTrainer` + `StyleGAN2Pipeline` | `configs/gan/gan_demo.py` | verified reference |
| DMD | `DMDBundle` + `DMDTrainer` + `DMDPipeline` | `configs/distillation/dmd_demo.py` | verified reference |

`verified reference` means the training / inference path is smoke-validated and runnable, but the default project is positioned as a framework reference implementation rather than a benchmark-tuned reproduction.

For LTX-2.5, “API/config contract verified” is deliberately narrower: tests
exercise registry construction, split-checkpoint roles, managed-trainer
dispatch, preprocessing command construction, and the pinned official Python
API through fakes. The gated 22B weights were not allocated in the repository
test environment, so this status does not claim model quality, throughput, or
training convergence.

## Installation

```bash
pip install -e .
```

`pyproject.toml` is the sole dependency source of truth. LTX-Video remains an
optional, source-pinned integration:

```bash
# Inference only, training only, or both
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"
python -m pip install -e ".[ltx-video]"
```

The LTX extras enforce the PyTorch API floor needed by the pinned source, but
they do **not** choose a CUDA wheel/index for your machine. For a production
GPU runtime, prepare the pinned official LTX checkout with its `uv sync`
workflow first, then install HFTrainer into that isolated environment. The
step-by-step commands are in the LTX guide linked below.

The LTX extras use the official
[Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) repository at commit
`400fd31054597515f47125691032c04b1c3ee24e`, because the current trainer/API
combination is not represented by the older PyPI package line.

Prepare local demo assets:

```bash
bash tools/download_checkpoints.sh
python3 tools/download_demo_data.py --task all
```

## Get Started

Run the simplest verified training path:

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

Run LTX-2.5 distilled inference after accepting the gated model terms and
downloading its split checkpoint pack:

```bash
export LTX25_CHECKPOINT_ROOT="$PWD/checkpoints/LTX-2.5"

hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_distilled.py \
  --prompt "A paper boat drifts through a rain-filled street at dusk." \
  --output outputs/ltx_video_2_5/distilled.mp4
```

Preprocess data and launch the managed official LoRA trainer:

```bash
hftrainer-ltx-preprocess data/ltx_video_2_5/dataset.json \
  --ltx-repo third_party/LTX-2 \
  --resolution-buckets 960x544x49 \
  --model-path "$LTX25_CHECKPOINT_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --text-encoder-path "$LTX25_CHECKPOINT_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --output-dir data/ltx_video_2_5/.precomputed

export LTX25_PREPROCESSED_DATA="$PWD/data/ltx_video_2_5/.precomputed"
hftrainer-train configs/ltx_video/train_ltx_video_2_5_lora.py
```

The preprocessing wrapper expects an official LTX-2 checkout at the same
pinned commit. The full gated-download, license, Linux/CUDA/VRAM, distilled vs
dev+LoRA, and inference instructions are in the
[LTX-Video 2.5 guide](docs/en/models/ltx_video_2_5.md) and the
[中文指南](docs/zh-cn/models/ltx_video_2_5.md).

Run the startup smoke suite:

```bash
python3 -m pytest -m smoke tests/smoke/test_task_startup.py
```

The smoke suite uses reduced temporary configs to verify that each task stack can start training and inference through the real CLI entry points.

## Core Design

HF-Trainer keeps the framework surface small:

- `AccelerateRunner` builds the full runtime from one config and owns the loop
- `build_runner_from_cfg` selects either `AccelerateRunner` or a registered
  managed trainer whose official upstream stack owns the complete loop
- `ModelBundle` holds task sub-modules and shared atomic forward functions
- `Trainer` assembles training-time control flow and optimization
- `Pipeline` assembles inference-time control flow without duplicating task internals

This is the main reason the project exists: training and inference stay aligned without forcing users into a non-HuggingFace inference API.

Imports are lightweight by default. `import hftrainer` creates the registries
without importing every model library; call `hftrainer.register_all_modules()`
for the built-in catalogue, or use config-level `custom_imports` to register
only one vertical slice. Optional LTX packages are imported only when an LTX
backend is constructed.

## Memory Control From Config

Supported today:

- global AMP via `accelerator.mixed_precision='no'|'fp16'|'bf16'`
- per-module loader dtype via `from_pretrained.torch_dtype` or `dtype`
- per-module post-load cast via `module_dtype='fp32'|'fp16'|'bf16'`
- activation memory reduction via `gradient_checkpointing=True`
- optimizer/state reduction via `trainable=False`, `trainable='lora'`, and `accelerator.gradient_accumulation_steps`

Important caveat:

- if you need a strict policy like `vae=fp32` and `transformer=bf16`, prefer per-module dtype settings and keep `accelerator.mixed_precision='no'`
- global AMP can still autocast eligible ops on top of module weights

See:

- [English Memory and Precision Guide](docs/en/memory.md)
- [简体中文 显存与精度指南](docs/zh-cn/memory.md)

## Integration Paths

HF-Trainer exposes three clear ways to adopt the framework:

| Starting point | What you implement | What stays HuggingFace-native |
| --- | --- | --- |
| an existing `transformers` / `diffusers` model | a task bundle plus task training logic | `from_pretrained`, official component classes, tokenizer / processor, and exported inference artifact |
| a custom or self-developed model | your own `nn.Module` plus a task bundle | config-driven construction, checkpointing, hooks, runner, and optional custom `save_pretrained` |
| a complete official algorithm stack | a thin bundle/pipeline adapter and, when needed, a managed trainer | upstream model math, preprocessing, optimizer, checkpoint, validation, and resume semantics |

Rule of thumb:

- if HuggingFace already has the model class, keep the official class inside the bundle and only add training wiring
- if HuggingFace already has the artifact layout, declare `HF_PRETRAINED_SPEC` / `HF_SAVE_PRETRAINED_SPEC` on the bundle instead of hand-writing loader/export methods
- if HuggingFace does not have the model class, use `ModelBundle.from_config(...)` and add custom `from_pretrained/save_pretrained` logic only when you need a stable exported artifact
- if upstream already owns a tightly coupled loop, pin and delegate to it
  instead of partially copying the algorithm into `AccelerateRunner`

## Documentation

| Topic | English | 简体中文 |
| --- | --- | --- |
| Docs Home | [Home](docs/en/index.md) | [首页](docs/zh-cn/index.md) |
| Installation | [Installation](docs/en/installation.md) | [安装说明](docs/zh-cn/installation.md) |
| Quick Start | [Quick Start](docs/en/quickstart.md) | [快速开始](docs/zh-cn/quickstart.md) |
| LTX-Video 2.5 | [LTX-Video 2.5](docs/en/models/ltx_video_2_5.md) | [LTX-Video 2.5](docs/zh-cn/models/ltx_video_2_5.md) |
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

The public API reference covers the user-facing framework surface:

- runner: `AccelerateRunner` and managed-trainer dispatch
- model core: `ModelBundle`
- training / inference base classes: `BaseTrainer`, `BasePipeline`
- runtime helpers: hooks, evaluators, visualizers, checkpoint utils
- CLI entry points: `hftrainer-train`, `hftrainer-infer`,
  `hftrainer-ltx-infer`, and `hftrainer-ltx-preprocess`

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
tests/        startup smoke tests and focused unit tests
```

Model code, task runtime, and data code are intentionally separated:

```text
hftrainer/models/<model_name>/
  bundle.py
  ...
hftrainer/trainers/<task_name>/
  ...
hftrainer/pipelines/<task_name>/
  ...
hftrainer/datasets/<task_name>/
  ...
```

Each runnable config declares focused `custom_imports`; this keeps optional
vertical slices such as `models/ltx_video`, `pipelines/ltx_video`, and
`trainers/ltx_video` out of the core import path.

Datasets follow an MMEngine-style split:

```text
dataset.load_data_list()  -> raw records
dataset.pipeline          -> decoding / tokenize / resize / pack transforms
collate_fn                -> batch assembly
```

## Scope Notes

- `docs/en/` and `docs/zh-cn/` are the source-of-truth public docs
- root-level `docs/*.md` pages are compatibility entry pages
- the GAN and DMD stacks are runnable framework references, not benchmark-tuned reproductions out of the box
- LTX-2.5 is pinned to one official source revision and contract-tested; full
  22B GPU inference/training still requires gated weights and a suitable Linux
  CUDA environment

## Acknowledgements

HF-Trainer is built around three complementary ecosystems:

- MMEngine for config-driven experiment construction and registry ergonomics
- HuggingFace for model classes, inference artifacts, and runtime interoperability
- Lightricks for the native LTX-2.5 model, pipeline, and training stack used by
  the optional adapter
