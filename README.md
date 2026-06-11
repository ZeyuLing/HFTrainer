# HF-Trainer

A unified, config-driven training framework built on the HuggingFace ecosystem. Combines MMEngine's declarative config system with Accelerate-native distributed training, and uses a `ModelBundle` abstraction to share forward logic between training and inference.

## Features

- **Config-Driven, Registry-Based** -- `.py` config files with `_base_` inheritance (MMEngine style)
- **Accelerate-Native** -- DDP, FSDP, DeepSpeed, mixed precision, gradient accumulation via HuggingFace Accelerate
- **ModelBundle = Shared Core** -- Trainer and Pipeline share the same forward functions, written once
- **Per-Module Control** -- Each sub-module independently controls `trainable`, `save_ckpt`, and LoRA via config
- **HuggingFace-First** -- Directly uses `diffusers`, `transformers`, and `peft` with no extra wrappers
- **Unified Checkpoint** -- Single `load_from` + `load_scope` API; `auto_resume=True` for cluster preemption recovery

## Installation

```bash
git clone <repo-url> && cd hf_trainer
pip install -e .
```

## Quick Start

```bash
# ViT classification smoke test (no GPU required)
python tools/train.py configs/classification/vit_base_demo.py

# Distributed training with 8 GPUs
bash tools/dist_train.sh configs/text2video/wan_demo.py 8
```

## Supported Tasks

| Task | ModelBundle | Trainer | Pipeline | Example Models |
|---|---|---|---|---|
| Classification | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | ViT, DeiT, Swin |
| Text-to-Image | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | SD1.5, SDXL |
| LLM SFT | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | TinyLlama, LLaMA, Qwen |
| Text-to-Video | `WanBundle` | `WanTrainer` | `WanPipeline` | WAN 1.3B/14B |

## Motion Representations & Conversions

`hftrainer.motion` is the standalone motion-domain library (no `ref_repo`
dependency for the core chain). It defines the motion representations we use
(`HML263`, `MS272`, SMPL `motion_135/138/198/…`, KIMODO SOMA, Unitree G1) and the
validated conversions between them — HumanML3D-263 ↔ SMPL ↔ SOMA ↔ G1 ↔
MotionStreamer-272, with skeleton/mesh/robot visualization.

> **Always convert through `hftrainer.motion.representation.convert`** — never
> hand-pick a low-level helper. The rot6d **COLUMN vs ROW** convention is the #1
> source of silent bugs.

```python
import os; os.environ["HFTRAINER_SKIP_AUTOREGISTER"] = "1"   # import-light
from hftrainer.motion.representation import convert

joints = convert.hml263_to_joints(m263)                       # 263 -> (T,22,3)
m135   = convert.hml263_to_motion135(m263, device="cuda")     # 263 -> SMPL motion_135 (ROW, IK)
m272   = convert.hml263_to_motion272(m263)                    # full chain -> MS272 evaluator space
```

See the detailed motion docs:

- [`docs/motion/representations.md`](docs/motion/representations.md) — representation table, conversion map, the rot6d-convention trap, and pre-rendered before/after-retarget clips
- [`docs/motion/api.md`](docs/motion/api.md) — API-level reference (every public function/class, signatures + conventions)
- [`hftrainer/motion/README.md`](hftrainer/motion/README.md) — library overview; specs in [`representation/specs.py`](hftrainer/motion/representation/specs.py)
- Browser demo: `scripts/demo/hml263_multi_repr_demo.py` + `motion_annot_web/repr_convert_demo/app.py`

## Documentation

For full documentation, see the [docs/](docs/) directory:

- [Installation](docs/installation.md) -- Install the package and download pretrained checkpoints
- [Quick Start](docs/quickstart.md) -- Smoke tests and inference examples
- [Distributed Training](docs/distributed.md) -- DDP, FSDP, DeepSpeed, single GPU
- [Experiment Directory](docs/experiment_dir.md) -- Work directory layout, checkpoint management, auto-resume
- [Architecture](docs/architecture.md) -- How the framework is structured
- [Supported Tasks](docs/tasks.md) -- Task table and val output conventions
- [Comparison](docs/comparison.md) -- Comparison with MMEngine and HuggingFace Trainer
- [Design Docs](docs/design/index.md) -- In-depth design rationale (bilingual Chinese/English)

To serve the docs locally with MkDocs:

```bash
pip install mkdocs-material
mkdocs serve
```

## License

Apache 2.0
