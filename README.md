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

| Implementation | Training | Inference | Local core | Current verification |
| --- | --- | --- | --- | --- |
| ViT classification | yes | yes | ViT, image processor, artifact loader | tiny forward/loss/backward; reference logits aligned; `.bin`/safetensors round-trip |
| LLaMA causal LM | full + LoRA | yes | LLaMA, KV cache, generation, BPE/WordPiece/Unigram tokenizer | tiny forward/loss/backward/generation; reference logits aligned; sharded artifact support |
| Stable Diffusion 1.5 | yes | yes | CLIP, byte-BPE tokenizer, VAE, conditional UNet, DDPM/DDIM/PNDM | checkpoint schema aligned; tiny numerical parity and train/infer round-trip |
| DMD | yes | one-step | reuses the local SD1.5 core | distribution matching, fake-score and teacher paths tested |
| StyleGAN2 | yes | yes | generator and discriminator | tiny adversarial path and artifact round-trip |
| Wan T2V | yes | yes | UMT5 path, tokenizer, video VAE, 3D transformer, flow scheduler | local tiny train/denoise/artifact tests; see compatibility note below |
| LTX-Video 2.5 | LoRA | distilled and dev+LoRA | pinned source reorganized into local model/trainer/pipeline layers; local Gemma text runtime and LoRA | API/config/checkpoint contracts and tiny Gemma path tested; full 22B GPU run not performed here |
| MiniMax-H3 Base 768p | experimental full/LoRA | T2VA, FL2VA, Ref2VA synchronized A/V | 50-layer Omni Transformer, Qwen3-VL-32B conditioner, visual/audio VAEs, processor and dual flow schedulers | tiny numerical/component/train/pipeline/artifact contracts; full 33B+32B GPU run not performed here |

“Reference aligned” refers to isolated development-time numerical checks. The
reference packages are not installed as product dependencies and are not
imported by HFTrainer at runtime.

Wan currently validates the framework-owned T2V contract and common checkpoint
names, but it does not claim bitwise compatibility with every historical Wan
release. Low checkpoint coverage is rejected instead of silently leaving most
weights randomly initialized. Its current compact VAE may require explicit
conversion for some full upstream artifacts.

LTX-2.5 is large and gated. The repository tests do not claim 22B model quality,
throughput, convergence, or end-to-end GPU execution. Standard text encoding,
training, and video generation are wired locally; image-conditioned *prompt
enhancement* is explicitly rejected until the Gemma vision tower is also
implemented locally. Image conditioning for the LTX video model is a separate
pipeline feature.

MiniMax-H3 is substantially larger and its public model agreement contains
territorial exclusions and use restrictions. The repository does not bundle
its weights or tokenizer/config assets. Tiny and frozen-reference tests cover
the local implementation and artifact boundary; they do not claim full-model
quality, throughput, convergence, or a successful 33B denoiser plus 32B
conditioner allocation on this development machine.

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

Train ViT:

```bash
python tools/train.py configs/vit/vit_base_demo.py
```

Train and run a local LLaMA LoRA:

```bash
python tools/train.py configs/llama/llama_lora_demo.py

python tools/infer.py \
  --config configs/llama/llama_lora_demo.py \
  --checkpoint work_dirs/llama_lora_smoke/checkpoint-iter_10 \
  --merge-lora \
  --prompt "Name one primary color."
```

Run SD1.5 inference:

```bash
python tools/infer.py \
  --config configs/sd15/sd15_demo.py \
  --checkpoint work_dirs/sd15_smoke/checkpoint-iter_10 \
  --prompt "A paper boat in a rain-filled street at dusk." \
  --output outputs/inference/paper_boat.png
```

The inference CLI reads `cfg.pipeline.type` and `cfg.inference.task`. It never
guesses inference behavior from a trainer or implementation class name.

## LTX-Video 2.5

Point the configs at an accepted and downloaded split checkpoint pack:

```bash
export LTX25_CHECKPOINT_ROOT="$PWD/checkpoints/LTX-2.5"

hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_distilled.py \
  --prompt "A paper boat drifts through a rain-filled street at dusk." \
  --output outputs/ltx_video_2_5/distilled.mp4
```

Preprocess a dataset using the packaged local script, then train:

```bash
hftrainer-ltx-preprocess data/ltx_video_2_5/dataset.jsonl \
  --resolution-buckets 960x544x49 \
  --model-path "$LTX25_CHECKPOINT_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --text-encoder-path "$LTX25_CHECKPOINT_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --output-dir data/ltx_video_2_5/.precomputed

export LTX25_PREPROCESSED_DATA="$PWD/data/ltx_video_2_5/.precomputed"
hftrainer-train configs/ltx_video/train_ltx_video_2_5_lora.py
```

No second LTX checkout or installed `ltx-core`, `ltx-pipelines`, or
`ltx-trainer` package is used. The modified pinned snapshot is organized under
HFTrainer's own layers.

Important: the included LTX source is governed by the
[LTX-2.x Community License Agreement](hftrainer/models/ltx_video/LICENSE.ltx-2.x),
which contains use restrictions and commercial-license conditions. See the
[upstream and modification record](hftrainer/models/ltx_video/UPSTREAM.md) and
[third-party notices](THIRD_PARTY_NOTICES.md) before use or redistribution.

## MiniMax-H3

After reading and accepting the upstream MiniMax H3 Community License
Agreement, download the frozen checkpoint revision and run one of the local
pipelines:

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include "model_index.json" "modular_model_index.json" \
    "processor/*" "tokenizer/*" "text_encoder/*" \
    "vae/*" "audio_vae/*" "scheduler/*" "audio_scheduler/*" \
    "transformer/*" "transformer_ref/*" \
  --local-dir checkpoints/MiniMax-H3

export MINIMAX_H3_ROOT="$PWD/checkpoints/MiniMax-H3"

hftrainer-infer \
  --config configs/minimax_h3/infer_h3_base_fl2va.py \
  --mode t2va \
  --prompt "A paper boat drifts downstream with synchronized water and bird sounds." \
  --duration 5 \
  --output outputs/minimax_h3/t2va.mp4
```

The FL2VA checkpoint partition also supports first and/or last frames. Ordered
image/video/audio references use the separate
`configs/minimax_h3/infer_h3_base_ref2va.py` recipe. Experimental transformer
LoRA training consumes cached Qwen/VAE features through
`configs/minimax_h3/train_h3_base_lora.py`; the matching
`infer_h3_base_fl2va_lora.py` recipe loads and merges its adapter checkpoint.
MiniMax has not published a complete official training recipe, so this is not
presented as recipe parity.

Read the [English MiniMax-H3 guide](docs/en/models/minimax_h3.md),
[中文指南](docs/zh-cn/models/minimax_h3.md), the
[upstream record](hftrainer/models/minimax_h3/UPSTREAM.md), the
[Apache-2.0 reference-code license](hftrainer/models/minimax_h3/LICENSE.apache-2.0),
and the
[complete model agreement](hftrainer/models/minimax_h3/LICENSE.minimax-h3)
before use or redistribution.

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
| LTX-Video 2.5 | [docs/en/models/ltx_video_2_5.md](docs/en/models/ltx_video_2_5.md) | [docs/zh-cn/models/ltx_video_2_5.md](docs/zh-cn/models/ltx_video_2_5.md) |
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
