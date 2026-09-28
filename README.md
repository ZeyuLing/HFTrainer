<div align="center">

<img src="assets/hftrainer_logo.png" alt="HFTrainer logo" width="420" />

# HFTrainer

**A config-driven training and inference framework with repository-owned model implementations.**

One implementation tree per model family, one shared bundle for training and
inference, and no runtime delegation to an external model framework.

<p>
  <a href="docs/en/index.md"><strong>Documentation</strong></a> •
  <a href="docs/en/quickstart.md"><strong>Quick Start</strong></a> •
  <a href="docs/en/integration.md"><strong>Integration Guide</strong></a> •
  <a href="docs/zh-cn/index.md"><strong>中文文档</strong></a> •
  <a href="https://github.com/ZeyuLing/HFTrainer/issues"><strong>Issues</strong></a>
</p>

</div>

## Why HFTrainer

Research repositories often combine a model package, a training package, an
inference package, and a separate adapter package. The resulting experiment
may run, but its numerical behavior and artifact format depend on whichever
versions happen to be installed.

HFTrainer uses a stricter boundary:

- model math lives in `hftrainer/models/<implementation>/network/`;
- the bundle owns components, artifact loading, and atomic forward operations;
- the trainer owns losses and update order;
- the pipeline owns inference orchestration and public inputs/outputs;
- configs refer only to classes registered from this repository;
- LoRA, schedulers, tokenizers, and checkpoint loading used by a model are
  implemented locally as part of that execution path.

PyTorch, Accelerate, MMEngine, safetensors, NumPy, Pillow, and similar
infrastructure libraries are still normal dependencies. The hard rule is that
HFTrainer model execution does not import or dynamically resolve another model
implementation such as `transformers`, `diffusers`, `peft`, or separately
installed LTX packages.

This rule is enforced by source-tree AST checks and fresh-process import hooks,
not only documented as a convention.

## Implementations

[Model cards](docs/en/models/index.md) · [Demo data availability](docs/en/models/demo-data.md) · [Public training evidence](docs/en/models/evidence/public_data/index.md)

| Model | Task | Public training data | Verification scope |
| :--- | :--- | :--- | :--- |
| [ViT-B/16](docs/en/models/vit.md) | Classification | Beans | 20 steps, held-out accuracy, native reload |
| [TinyLlama 1.1B](docs/en/models/llama.md) | Plain-text LM | WikiText-2 | 20-step LoRA, held-out line-wise perplexity |
| [Stable Diffusion 1.5](docs/en/models/sd15.md) | Text to image | Smithsonian butterflies | 20-step UNet update, base image demo |
| [DMD / DMD2 initialization](docs/en/models/dmd.md) | One-step image distillation | Butterfly captions | 20-step generator/fake-score updates |
| [StyleGAN2-ADA](docs/en/models/stylegan2.md) | Unconditional images | Smithsonian butterflies | 20-step G/D updates; official FFHQ output parity |
| [Wan 2.1 T2V 1.3B](docs/en/models/wan.md) | Text to video | CREMA-D video | Reduced-resolution training; 480p base preview |
| [LTX-Video 2.5 Dev](docs/en/models/ltx_video_2_5.md) | Video + audio | CREMA-D audio-video | 20-step LoRA; independent official base comparison |
| [MiniMax-H3 Base](docs/en/models/minimax_h3.md) | Video + audio | CREMA-D audio-video | 20-step LoRA; independent official T2VA comparison |

“Reference aligned” refers to isolated development-time numerical checks. The
reference packages are not installed as product dependencies and are not
imported by HFTrainer at runtime.

Wan now uses the complete causal VAE and UMT5 component graph for the published
Wan 2.1 T2V 1.3B checkpoint. Full-weight component outputs match after native
export/reload; tiny numerical checks compare against pinned references. This
does not establish full-resolution video quality or compatibility with every
Wan release. StyleGAN2 additionally includes the native NVIDIA ADA graph and
pretrained FFHQ 1024 generator/discriminator; the original compact demo remains
available separately.

LTX-2.5 is large and gated. Full dev-model GPU execution and independent official
sample parity were verified on 2026-09-28; short demo runs do not establish
training convergence or every supported mode. Standard text encoding,
training, and video generation are wired locally; image-conditioned *prompt
enhancement* is explicitly rejected until the Gemma vision tower is also
implemented locally. Image conditioning for the LTX video model is a separate
pipeline feature.

MiniMax-H3 is substantially larger and its public model agreement contains
territorial exclusions and use restrictions. The repository does not bundle
its weights or tokenizer/config assets. Full-weight T2VA now runs on one H200 by loading the 32B conditioner and 33B
denoiser sequentially. Sample parity and LoRA train/resume/infer are verified;
other conditioning modes, multi-GPU performance and convergence remain separate
validation tasks. See the model cards for one-command recipes, raw demo loss
curves, environment requirements and remaining codec differences.

## Installation

```bash
python -m pip install -e .
```

The base installation intentionally does not install an external model
framework. LTX uses additional media/runtime utilities:

```bash
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"
# or both
python -m pip install -e ".[ltx-video]"

# MiniMax-H3 media I/O and checkpoint-download helpers
python -m pip install -e ".[minimax-h3]"
```

Choose the correct CUDA-enabled PyTorch wheel for the target machine before
installing LTX extras. LTX-2.5 requires the capabilities guarded by the local
runtime and a suitable Linux/CUDA environment for the real 22B workflow.

## Quick Start

Install `huggingface_hub` alongside HFTrainer. The public-data launcher downloads
pinned data and native checkpoints, verifies dataset checksums, prepares caches
when needed, and starts the normal trainer:

```bash
python -m pip install "huggingface_hub>=0.34,<2"
python tools/run_public_demo.py vit
# Also: llama sd15 dmd stylegan2 wan ltx_video minimax_h3
```

All examples use real public datasets and run 20 optimizer steps. Use
`--checkpoint path/to/complete_bundle` to reuse local weights, `--prepare-only`
to stop before training, or `--work-dir path/to/run` for a separate experiment.
See the model card for hardware, media dependencies and gated-model access.

Inference accepts a **checkpoint path or Hub ID**; its `config.json` selects
the registered local model and pipeline. A separate Python config is optional:

```bash
python tools/infer.py --model ZeyuLing/HFTrainer-Stable-Diffusion-1.5 \
  --device cuda --seed 42 --num-steps 30 \
  --prompt "A butterfly on a flower in natural daylight." \
  --output outputs/butterfly.png
```

Model cards pin exact weight revisions and provide each task's inputs and
outputs. No remote model code is executed.

## Video models

Both video models use the same checkpoint-only entry point. Accept their Hub
agreements and authenticate with `hf auth login` first; H3 requires manual
approval. Install the model-specific extras and FFmpeg as documented in the
cards. The measured environment is one H200; H3 also needs substantial host RAM.

```bash
python tools/infer.py --model ZeyuLing/HFTrainer-LTX-2.5-22B-Dev \
  --device cuda --seed 42 --height 512 --width 768 --num-frames 121 \
  --num-steps 30 --prompt "A woman speaks calmly to the camera in soft daylight." \
  --output outputs/ltx.mp4

python tools/infer.py --model ZeyuLing/HFTrainer-MiniMax-H3-Base-FL2VA \
  --device cuda --seed 42 --height 768 --width 768 --num-steps 50 \
  --prompt "A woman speaks calmly to the camera in soft daylight." \
  --output outputs/h3.mp4
```

The H3 path places its conditioner and denoiser on the GPU sequentially. LTX
retains the dev model's official second-stage LoRA. Historical independent
reference comparisons, numerical limits and playable demos are linked from
the cards. No hosted Regenerate-2K or Context-IR implementation is claimed.

## Checkpoint availability

All eight model families have **processed pretrained HFTrainer bundles on
Hugging Face**. Each English model card provides a pinned revision, one-command
download, training/inference configs, source licenses and validation scope.

| Model | Processed checkpoint | Access |
| :--- | :--- | :--- |
| ViT | [ZeyuLing/HFTrainer-ViT-Base-Patch16-224](https://huggingface.co/ZeyuLing/HFTrainer-ViT-Base-Patch16-224) | Public |
| TinyLlama | [ZeyuLing/HFTrainer-TinyLlama-1.1B-Chat](https://huggingface.co/ZeyuLing/HFTrainer-TinyLlama-1.1B-Chat) | Public |
| Stable Diffusion 1.5 | [ZeyuLing/HFTrainer-Stable-Diffusion-1.5](https://huggingface.co/ZeyuLing/HFTrainer-Stable-Diffusion-1.5) | Public |
| DMD (DMD2 initialization) | [ZeyuLing/HFTrainer-DMD2-SD15-Initialization](https://huggingface.co/ZeyuLing/HFTrainer-DMD2-SD15-Initialization) | Public |
| StyleGAN2-ADA FFHQ 1024 | [ZeyuLing/HFTrainer-StyleGAN2-ADA-FFHQ-1024](https://huggingface.co/ZeyuLing/HFTrainer-StyleGAN2-ADA-FFHQ-1024) | Public |
| Wan 2.1 T2V 1.3B | [ZeyuLing/HFTrainer-Wan2.1-T2V-1.3B](https://huggingface.co/ZeyuLing/HFTrainer-Wan2.1-T2V-1.3B) | Public |
| LTX 2.5 22B Dev | [ZeyuLing/HFTrainer-LTX-2.5-22B-Dev](https://huggingface.co/ZeyuLing/HFTrainer-LTX-2.5-22B-Dev) | Accept agreement |
| MiniMax H3 Base FL2VA | [ZeyuLing/HFTrainer-MiniMax-H3-Base-FL2VA](https://huggingface.co/ZeyuLing/HFTrainer-MiniMax-H3-Base-FL2VA) | Manual access approval |

These are real pretrained weights exported or packaged through our native
bundle contract. They are not random-initialization artifacts or claims of
new training. DMD uses the author's trained DMD2 SD1.5 non-GAN checkpoint as
initialization; it is not an original-DMD-paper reproduction. Short-run fine-tuned weights are separate from these pretrained bases. Public
demo datasets are linked below; historical private clips remain unpublished.

Training and inference share `cfg.model.from_pretrained`, with
`save_pretrained` and `tools/export_model.py` providing the export path.
[Configs and pinned revisions](configs/pretrained/checkpoints.json) ·
[Conversion and integrity evidence](docs/en/models/evidence/checkpoints/index.md).
These are HFTrainer artifacts hosted on the Hub; no compatibility with another
framework's automatic model loader is implied.

## Public data and training evidence

| Dataset | Tasks / models | Public download |
| :--- | :--- | :--- |
| Beans | ViT classification | [hftrainer_beans](https://huggingface.co/datasets/ZeyuLing/hftrainer_beans) |
| WikiText-2 | TinyLlama plain-text LM | [hftrainer_wikitext2](https://huggingface.co/datasets/ZeyuLing/hftrainer_wikitext2) |
| Smithsonian butterflies | SD1.5, DMD, StyleGAN2 | [hftrainer_smithsonian_butterflies](https://huggingface.co/datasets/ZeyuLing/hftrainer_smithsonian_butterflies) |
| CREMA-D mini | H3/LTX audio-video, Wan video-only | [hftrainer_crema_d_mini](https://huggingface.co/datasets/ZeyuLing/hftrainer_crema_d_mini) |

Data is downloaded to `data/hftrainer_*`. Each dataset preserves source
revision, licensing, transformations, fixed splits and checksums; published
files were downloaded again and verified. CREMA-D splits are actor-disjoint.
The earlier private dance clips are not required by these recipes.

Each English model card follows the same structure: original paper/code,
released settings with config and checkpoint URLs, one-command training and
inference, measured loss/metrics, produced demos, limitations and BibTeX.
**Short training proves execution, not convergence or generation quality.**
Base-model demos are identified separately from fine-tuning results.

[Dataset provenance](docs/en/models/demo-data.md) ·
[Model cards and results](docs/en/models/index.md) ·
[Model-card standard](docs/en/models/standard.md)

## Repository Standard

```text
hftrainer/
  models/
    <implementation_id>/
      network/          model math and model-specific primitives
      bundle.py         components, atomic forwards, artifact boundary
      checkpoint.py     implementation artifact schema when needed
  trainers/
    <implementation_id>/
      trainer.py        losses and update order
  pipelines/
    <implementation_id>/
      pipeline.py       inference graph and public I/O
  tasks/
    <reusable_task>/    only genuinely reusable task contracts
  datasets/
    <data_contract>/    records, transforms, and batching
  evaluation/
    <task_contract>/    reusable metrics
configs/
  <implementation_id>/
```

The same `implementation_id` is used across model, trainer, pipeline, and
config whenever behavior belongs to one concrete method. A task directory is
used only when the code is truly reusable across implementations. For example,
ViT uses the reusable `image_classification` task trainer/pipeline, while
SD1.5, DMD, Wan, StyleGAN2, and LTX keep method-specific trainers and
pipelines.

Do not add task aliases such as `models/classification` or mix task names and
paper/model names at the same model-package level.

## Adding a Model

An integration is complete only when all of the following are true:

1. Core model code is present under `hftrainer/models/<implementation>/network`.
2. Config component types resolve through `MODEL_COMPONENTS` to classes whose
   module starts with `hftrainer.models.`.
3. `bundle.py` imports only local model components and owns atomic operations.
4. The bundle provides strict local artifact loading/saving; missing keys,
   shape mismatches, and low coverage are not silently accepted.
5. Trainer and pipeline reuse bundle operations instead of maintaining another
   model copy.
6. Tiny forward, loss, backward, inference, and artifact round-trip tests pass.
7. The forbidden-dependency test still passes in a process that actively blocks
   external model packages.
8. An English model card follows the [shared standard](docs/en/models/standard.md),
   with launch commands, data availability, measured evidence and explicit limits.

See the [English integration guide](docs/en/integration.md) or
[中文模型接入指南](docs/zh-cn/integration.md).

## Verification

```bash
python -m pytest -q
python -m compileall -q hftrainer tools
python -m build
```

The focused tests include:

- source-tree forbidden-import scanning;
- blocked-package fresh-process imports;
- local tokenizer and LoRA contracts;
- tiny model forward/loss/backward/generation tests;
- artifact checksum, schema, tamper, and round-trip tests;
- all shipped config imports and registry resolution;
- LTX split-checkpoint, preprocessing, packaged trainer, and inference contracts.
- MiniMax-H3 packed-layout, scheduler, transformer, Qwen/tokenizer, dual-VAE,
  cached-training, synchronized-inference, and strict-artifact contracts.

## Documentation

| Topic | English | 简体中文 |
| --- | --- | --- |
| Home | [docs/en/index.md](docs/en/index.md) | [docs/zh-cn/index.md](docs/zh-cn/index.md) |
| Installation | [docs/en/installation.md](docs/en/installation.md) | [docs/zh-cn/installation.md](docs/zh-cn/installation.md) |
| Quick start | [docs/en/quickstart.md](docs/en/quickstart.md) | [docs/zh-cn/quickstart.md](docs/zh-cn/quickstart.md) |
| Integration standard | [docs/en/integration.md](docs/en/integration.md) | [docs/zh-cn/integration.md](docs/zh-cn/integration.md) |
| Architecture | [docs/en/architecture.md](docs/en/architecture.md) | [docs/zh-cn/architecture.md](docs/zh-cn/architecture.md) |
| ModelBundle | [docs/en/design/model_bundle.md](docs/en/design/model_bundle.md) | [docs/zh-cn/design/model_bundle.md](docs/zh-cn/design/model_bundle.md) |
| [LTX-Video 2.5](docs/en/models/ltx_video_2_5.md) | [docs/en/models/ltx_video_2_5.md](docs/en/models/ltx_video_2_5.md) | [docs/zh-cn/models/ltx_video_2_5.md](docs/zh-cn/models/ltx_video_2_5.md) |
| MiniMax-H3 | [docs/en/models/minimax_h3.md](docs/en/models/minimax_h3.md) | [docs/zh-cn/models/minimax_h3.md](docs/zh-cn/models/minimax_h3.md) |

## Acknowledgements and Provenance

HFTrainer uses MMEngine for configuration/registries and Accelerate for
distributed runtime orchestration. Model execution is repository-owned.

Some local implementations were validated against public reference
implementations during development. Those references are not runtime
dependencies. LTX is a modified pinned source snapshot and retains its own
license, notices, and use restrictions as described above.
MiniMax-H3 model materials likewise retain their Community License terms;
HFTrainer ships the complete agreement, required notice, and pinned
modification record but no pretrained model artifacts.
