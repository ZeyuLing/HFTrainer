# Installation

## Core requirements

- Python 3.10 or newer
- a PyTorch build appropriate for your CPU/CUDA platform

Install the editable core environment from the repository root:

```bash
python -m pip install -e .
```

`pyproject.toml` is the only dependency and package-metadata source of truth.
`requirements.txt` delegates to the editable project, and `setup.py` is only a
compatibility shim for old packaging tools; neither file owns version ranges.

## Optional groups

```bash
# TensorBoard logging
python -m pip install -e ".[logging]"

# Documentation authoring
python -m pip install -e ".[docs]"

# Tests and package-build checks
python -m pip install -e ".[dev]"
```

## LTX-Video 2.5

HFTrainer contains its modified, pinned LTX model, trainer, preprocessing, and
pipeline source. LTX is not part of the base dependency set because the 22B
workflow needs additional media and scientific-computing utilities. Select
inference, training, or both:

```bash
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"
python -m pip install -e ".[ltx-video]"

# Optional experiment tracking / Hub publication
python -m pip install -e ".[ltx-video-integrations]"

# Optional EXR/HDR media paths
python -m pip install -e ".[ltx-video-hdr]"
```

These extras install only supporting libraries such as PyAV, Einops, SciPy,
Pydantic, Rich, torchaudio, pandas, and Pillow-HEIF. W&B/Hub publication and
EXR/HDR handling remain separate opt-in groups. None of the groups installs
`ltx-core`, `ltx-pipelines`, `ltx-trainer`, or another model framework. No
second LTX checkout is required.

The extras require `torch>=2.8`; choose the CUDA-specific PyTorch wheel for the
target machine before installing them. The supported full training path is
Linux with NVIDIA CUDA. See the complete
[LTX-Video 2.5 guide](models/ltx_video_2_5.md) for the source/license boundary,
gated model access, exact commands, validation limits, and hardware planning.

## Model dependency boundary

Model implementations, tokenizers, sampling schedulers, LoRA layers, artifact
loaders, trainers, and pipelines execute from `hftrainer.*`. The project uses
general infrastructure libraries such as PyTorch, Accelerate, MMEngine,
safetensors, NumPy, and Pillow, but does not require an installed external
model implementation package. Installing such a package must not change which
model code a config resolves.

## Console commands

An editable or wheel installation exposes:

```text
hftrainer-train
hftrainer-infer
hftrainer-ltx-infer
hftrainer-ltx-preprocess
```

The corresponding `python tools/...` entry points remain available for source
tree workflows.

## Demo assets

Download checkpoints used by the small built-in demos:

```bash
bash tools/download_checkpoints.sh
```

Download or prepare demo data:

```bash
python tools/download_demo_data.py --task all
```

LTX-2.5 weights are gated and intentionally excluded from these demo helpers.
Download them only after accepting the model license and access terms.
