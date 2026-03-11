# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+
- `accelerate`, `transformers`, `diffusers`, `datasets`, `mmengine`
- `torchvision` for image/video utilities used by the demo tasks

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install tensorboard
```

LoRA support depends on `peft`. The currently validated range is:

```bash
pip install "accelerate>=1.1,<2" "peft>=0.17,<0.18"
```

## Demo Assets

Download checkpoints used by the demo configs:

```bash
bash tools/download_checkpoints.sh
```

Download or prepare demo data:

```bash
python3 tools/download_demo_data.py --task all
```

The repository already contains a tiny local demo dataset under `data/` and local checkpoint placeholders under `checkpoints/`, so you can inspect the structure even before downloading everything.
