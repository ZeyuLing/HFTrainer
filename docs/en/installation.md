# Installation

## Core requirements

- Python 3.10 or newer
- a PyTorch build appropriate for your CPU/CUDA platform
- Git when installing a source-pinned optional integration such as LTX-Video

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

LTX is intentionally not part of the core installation. Select inference,
training, or both:

```bash
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"
python -m pip install -e ".[ltx-video]"
```

The extras install `ltx-core`, `ltx-pipelines`, and/or `ltx-trainer` directly
from the reviewed official Lightricks/LTX-2 commit
`400fd31054597515f47125691032c04b1c3ee24e`. This pin is deliberate: the PyPI
package line does not expose the current trainer/API combination used by the
adapter.

The extras enforce `torch>=2.8` because the pinned source imports
`torch.compiler.nested_compile_region`, an API missing from PyTorch 2.7.x.
They do not select the CUDA-specific PyTorch index for your GPU. LTX training
should run on Linux with NVIDIA CUDA; use the official checkout's `uv sync`
workflow for the recommended isolated runtime, then install HFTrainer into
that environment. See the complete [LTX-Video 2.5 guide](models/ltx_video_2_5.md)
for exact commands, gated model access, licensing, and hardware planning.

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
